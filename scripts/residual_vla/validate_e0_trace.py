#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Enforce the action-path part of the E0 hard gate from trace shards."""

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_dir", type=Path)
    args = parser.parse_args()
    count = 0
    max_error = 0.0
    for shard in sorted(args.trace_dir.rglob("part_*.pt")):
        for record in torch.load(shard, map_location="cpu", weights_only=False):
            count += 1
            base = record["base_action_chunk"]
            for key in ("preclip_action", "executed_action"):
                candidate = record[key]
                if not torch.equal(base, candidate):
                    error = (base - candidate).abs().max().item()
                    max_error = max(max_error, error)
                    if error > 1e-7:
                        raise SystemExit(
                            f"E0_FAIL: {key} max_abs_error={error:.9g} > 1e-7"
                        )
    if count == 0:
        raise SystemExit("E0_FAIL: no trace records found")
    print(f"E0_ZERO_RESIDUAL_PASS records={count} max_action_diff={max_error:.9g}")


if __name__ == "__main__":
    main()
