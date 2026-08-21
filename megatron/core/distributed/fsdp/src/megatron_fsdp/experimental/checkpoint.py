# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PyTorch Distributed Checkpoint (DCP) save/load for the experimental Megatron-FSDP path.

After :func:`fully_shard`, a module's parameters rest as ``DTensor`` views over the optimizer
(``main_weight``) buffers, and the optimizer's ``exp_avg``/``exp_avg_sq`` states are ``DTensor`` s
on the same device mesh. The standard DCP state-dict helpers
(:func:`torch.distributed.checkpoint.state_dict.get_model_state_dict` /
:func:`~torch.distributed.checkpoint.state_dict.get_optimizer_state_dict`) expose those as FQN-keyed
DTensors and initialize the (empty) optimizer state on load, so we do not reimplement that here.

The one Megatron-FSDP-specific step is :func:`preprocess_state_dict_for_uneven_dtensor`. A
``FsdpParameterGroup`` packs several parameters into one flat buffer with least-common-multiple row
padding, so a parameter's per-rank shard does not tile like torch's canonical ``Shard(0)`` (a rank
may own several rows of one parameter and none of the next). The helper attaches each DTensor's true
per-shard chunk offsets so DCP writes and reshards it correctly; without it the default planner
assumes canonical ``Shard(0)`` offsets and silently corrupts the checkpoint.
"""

import os
from contextlib import contextmanager
from copy import deepcopy

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.tensor import DTensor

from ..uneven_dtensor import preprocess_state_dict_for_uneven_dtensor
from .parameter_group import sync_model_weights_from_main_weights

__all__ = ["save_checkpoint", "load_checkpoint"]


def _optimizer_state_schema(state: dict) -> dict:
    """Convert one optimizer state to rank-portable construction metadata."""
    schema = {}
    for key, value in state.items():
        if isinstance(value, DTensor):
            schema[key] = ("dtensor", value.dtype)
        elif torch.is_tensor(value):
            schema[key] = ("tensor", value.detach().cpu(), value.device.type)
        else:
            schema[key] = ("value", deepcopy(value))
    return schema


def _optimizer_state_from_schema(schema: dict, param: DTensor) -> dict:
    """Construct an empty local optimizer state from another rank's schema."""
    state = {}
    for key, spec in schema.items():
        kind, value, *metadata = spec
        if kind == "dtensor":
            local_value = torch.empty_like(param.to_local(), dtype=value)
            state[key] = DTensor.from_local(
                local_tensor=local_value,
                device_mesh=param.device_mesh,
                placements=param.placements,
                shape=param.shape,
                stride=param.stride(),
            )
        elif kind == "tensor":
            device_type = metadata[0]
            device = param.device if device_type == param.device.type else torch.device(device_type)
            state[key] = value.to(device=device)
        else:
            state[key] = deepcopy(value)
    return state


@contextmanager
def _complete_empty_local_optimizer_state(optimizer: torch.optim.Optimizer):
    """Temporarily expose state for empty shards to PyTorch state-dict helpers."""
    missing_by_group = [
        [param for param in group["params"] if param not in optimizer.state]
        for group in optimizer.param_groups
    ]
    local_schemas = [
        (
            _optimizer_state_schema(optimizer.state[param])
            if (param := next((p for p in group["params"] if p in optimizer.state), None))
            is not None
            else None
        )
        for group in optimizer.param_groups
    ]
    invalid_missing = any(
        not isinstance(param, DTensor) or param.to_local().numel() != 0
        for missing_params in missing_by_group
        for param in missing_params
    )
    local_metadata = {
        "has_missing": any(missing_by_group),
        "invalid_missing": invalid_missing,
        "schemas": local_schemas,
    }
    if torch.distributed.is_initialized():
        gathered_metadata = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered_metadata, local_metadata)
    else:
        gathered_metadata = [local_metadata]

    if not any(metadata["has_missing"] for metadata in gathered_metadata):
        yield
        return
    if any(metadata["invalid_missing"] for metadata in gathered_metadata):
        raise ValueError("Missing optimizer state for a parameter with non-empty local storage.")

    group_schemas = []
    for group_index in range(len(optimizer.param_groups)):
        schema = next(
            (
                metadata["schemas"][group_index]
                for metadata in gathered_metadata
                if metadata["schemas"][group_index] is not None
            ),
            None,
        )
        if schema is None:
            raise ValueError("Cannot infer optimizer state for an empty parameter group.")
        group_schemas.append(schema)

    synthetic_params = []
    try:
        for missing_params, schema in zip(missing_by_group, group_schemas):
            for param in missing_params:
                optimizer.state[param] = _optimizer_state_from_schema(schema, param)
                synthetic_params.append(param)
        yield
    finally:
        for param in synthetic_params:
            optimizer.state.pop(param, None)


def _get_preprocessed_optimizer_state_dict(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> dict:
    """Get optimizer state with every rank participating in uneven-DTensor collectives."""
    with _complete_empty_local_optimizer_state(optimizer):
        state_dict = get_optimizer_state_dict(model, optimizer)
        preprocess_state_dict_for_uneven_dtensor(state_dict)
    return state_dict


def _remove_empty_local_optimizer_state(optimizer: torch.optim.Optimizer) -> None:
    """Remove synthetic optimizer state for DTensors with no local storage."""
    for group in optimizer.param_groups:
        for param in group["params"]:
            if isinstance(param, DTensor) and param.to_local().numel() == 0:
                optimizer.state.pop(param, None)


def _init_optimizer_state(
    optimizer: torch.optim.Optimizer, *, skip_empty_local_shards: bool = False
) -> None:
    """Allocate optimizer state so a DCP load has DTensors to fill.

    :func:`get_optimizer_state_dict` initializes empty optimizer state via torch's
    ``_init_optim_state``, but that assigns a parameter-dtype gradient. A Megatron-FSDP sharded
    parameter advertises the FSDP gradient dtype through ``grad_dtype``, which differs from the
    (main-weight) parameter dtype under mixed precision, and rejects a mismatched gradient. So
    initialize the state here with a ``grad_dtype``-matched zero gradient; the subsequent load
    overwrites it. This is a no-op once the state exists (for example after a training step).

    TODO: this function becomes unnecessary once torch's ``_init_optim_state`` honors a parameter's
    ``grad_dtype`` when it allocates the placeholder gradient (``torch.zeros_like(param)`` in
    ``torch/distributed/checkpoint/state_dict.py``); an upstream issue is being filed.
    """
    if optimizer.state:
        return
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is None:
                grad_dtype = getattr(param, "grad_dtype", None) or param.dtype
                param.grad = torch.zeros_like(param, dtype=grad_dtype)
    if not skip_empty_local_shards:
        optimizer.step()
    else:
        full_param_groups = optimizer.param_groups
        full_param_lists = [group["params"] for group in full_param_groups]
        active_param_groups = []
        try:
            for group, full_params in zip(full_param_groups, full_param_lists):
                group["params"] = [param for param in full_params if param.to_local().numel() > 0]
                if group["params"]:
                    active_param_groups.append(group)
            optimizer.param_groups = active_param_groups
            optimizer.step()
        finally:
            optimizer.param_groups = full_param_groups
            for group, full_params in zip(full_param_groups, full_param_lists):
                group["params"] = full_params
    optimizer.zero_grad()


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: str | os.PathLike,
    *,
    extra_state: dict | None = None,
) -> None:
    """Save a ``fully_shard``-wrapped model and its optimizer as a DCP checkpoint.

    Args:
        model: A module tree that has been sharded with :func:`fully_shard`.
        optimizer: Optimizer stepping the sharded parameters.
        checkpoint_dir: Destination directory for the DCP checkpoint.
        extra_state: Optional non-model state to include in the same DCP checkpoint.
    """
    model_state_dict = get_model_state_dict(model)
    optimizer_state_dict = _get_preprocessed_optimizer_state_dict(model, optimizer)
    preprocess_state_dict_for_uneven_dtensor(model_state_dict)

    state_dict = {"model": model_state_dict, "optimizer": optimizer_state_dict}
    if extra_state:
        overlap = state_dict.keys() & extra_state.keys()
        if overlap:
            raise ValueError(f"extra_state contains reserved keys: {sorted(overlap)}")
        state_dict.update(extra_state)
    dcp.save(state_dict, checkpoint_id=checkpoint_dir)


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: str | os.PathLike,
    *,
    sync_model_weights: bool = True,
    extra_state: dict | None = None,
    skip_empty_optimizer_shards: bool = False,
) -> dict:
    """Load a DCP checkpoint into a ``fully_shard``-wrapped model and its optimizer.

    The model and optimizer must already be sharded with the same layout used at save time (the same
    module structure and mesh); DCP reshards the on-disk data to this rank's shards.
    :func:`~torch.distributed.checkpoint.state_dict.get_optimizer_state_dict` initializes the
    (empty) optimizer state so DCP has DTensors to load into in place, and the ``set_*`` helpers
    reinstall the loaded state.

    Args:
        model: A module tree sharded with :func:`fully_shard`, whose weights receive the load.
        optimizer: Optimizer whose state receives the load.
        checkpoint_dir: Source directory of the DCP checkpoint.
        sync_model_weights: Refresh compute weights from the loaded main weights afterwards.
        extra_state: Optional non-model state template to load from the same DCP checkpoint.
        skip_empty_optimizer_shards: Omit empty local DTensors from the synthetic optimizer step.

    Returns:
        The loaded entries requested by ``extra_state``.
    """
    _init_optimizer_state(optimizer, skip_empty_local_shards=skip_empty_optimizer_shards)
    with _complete_empty_local_optimizer_state(optimizer):
        model_state_dict = get_model_state_dict(model)
        optimizer_state_dict = get_optimizer_state_dict(model, optimizer)
        preprocess_state_dict_for_uneven_dtensor(optimizer_state_dict)
        preprocess_state_dict_for_uneven_dtensor(model_state_dict)

        state_dict = {"model": model_state_dict, "optimizer": optimizer_state_dict}
        if extra_state:
            overlap = state_dict.keys() & extra_state.keys()
            if overlap:
                raise ValueError(f"extra_state contains reserved keys: {sorted(overlap)}")
            state_dict.update(extra_state)
        dcp.load(state_dict, checkpoint_id=checkpoint_dir)
        set_model_state_dict(model, model_state_dict)
        set_optimizer_state_dict(
            model, optimizer, optimizer_state_dict, options=StateDictOptions(strict=False)
        )
    if skip_empty_optimizer_shards:
        _remove_empty_local_optimizer_state(optimizer)
    if sync_model_weights:
        sync_model_weights_from_main_weights(model.parameters())
    return {key: state_dict[key] for key in (extra_state or {})}
