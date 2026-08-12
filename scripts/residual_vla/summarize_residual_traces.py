#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Summarize rollout-local residual trace shards without loading RLinf."""

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_dir", type=Path)
    args = parser.parse_args()
    records = []
    for shard in sorted(args.trace_dir.rglob("part_*.pt")):
        records.extend(torch.load(shard, map_location="cpu", weights_only=False))
    if not records:
        raise SystemExit("No residual trace shards found.")
    residual = torch.cat(
        [
            record["scaled_residual"]
            .detach()
            .float()
            .reshape(-1, *record["scaled_residual"].shape[-2:])
            for record in records
        ]
    )
    base = torch.cat(
        [
            record["base_action_chunk"]
            .detach()
            .float()
            .reshape(-1, *record["base_action_chunk"].shape[-2:])
            for record in records
        ]
    )
    preclip = torch.cat(
        [
            record["preclip_action"]
            .detach()
            .float()
            .reshape(-1, *record["preclip_action"].shape[-2:])
            for record in records
        ]
    )
    executed = torch.cat(
        [
            record["executed_action"]
            .detach()
            .float()
            .reshape(-1, *record["executed_action"].shape[-2:])
            for record in records
        ]
    )
    residual_rms = residual.square().mean().sqrt()
    base_rms = base.square().mean().sqrt()
    summary = {
        "records": len(records),
        "residual_abs_mean": residual.abs().mean().item(),
        "residual_rms": residual_rms.item(),
        "residual_max": residual.abs().max().item(),
        "per_action_dim_rms": residual.square().mean((0, 1)).sqrt().tolist(),
        "per_chunk_position_rms": residual.square().mean((0, 2)).sqrt().tolist(),
        "preclip_saturation_rate": ((preclip < -1) | (preclip > 1))
        .float()
        .mean()
        .item(),
        "executed_saturation_rate": ((executed == -1) | (executed == 1))
        .float()
        .mean()
        .item(),
        "base_action_rms": base_rms.item(),
        "residual_base_rms_ratio": (residual_rms / base_rms.clamp_min(1e-12)).item(),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
