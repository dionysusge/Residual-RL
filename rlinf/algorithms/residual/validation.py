# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import hashlib
from collections.abc import Iterable

import torch


def parameter_checksum(module: torch.nn.Module) -> str:
    """Return a deterministic SHA-256 checksum over names and parameter bytes."""
    digest = hashlib.sha256()
    for name, parameter in module.named_parameters():
        digest.update(name.encode("utf-8"))
        value = parameter.detach().cpu().contiguous()
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def assert_base_vla_frozen(
    base_model: torch.nn.Module,
    optimizers: Iterable[torch.optim.Optimizer] = (),
) -> None:
    """Fail if base parameters are trainable or present in any optimizer."""
    base_parameters = list(base_model.parameters())
    if any(parameter.requires_grad for parameter in base_parameters):
        raise AssertionError("Base VLA has parameters with requires_grad=True.")
    base_ids = {id(parameter) for parameter in base_parameters}
    for optimizer in optimizers:
        optimizer_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        if base_ids & optimizer_ids:
            raise AssertionError("Base VLA parameters are present in an optimizer.")
