#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Select a Plus adaptation task from paired base-screening CSV rows."""

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--min-standard", type=float, default=0.9)
    parser.add_argument("--min-shifted", type=float, default=0.5)
    parser.add_argument("--max-shifted", type=float, default=0.9)
    args = parser.parse_args()
    with args.csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    standard = {}
    shifted = []
    for row in rows:
        key = (row["suite"], row["task"])
        if row["benchmark"].lower() in {"standard", "libero_standard"}:
            standard[key] = row
        elif row["benchmark"].lower() in {"plus", "libero_plus"}:
            shifted.append(row)
    candidates = []
    for row in shifted:
        key = (row["suite"], row["task"])
        base = standard.get(key)
        if base is None:
            continue
        standard_rate = float(base["success_rate"])
        shifted_rate = float(row["success_rate"])
        if not (
            standard_rate >= args.min_standard
            and args.min_shifted <= shifted_rate <= args.max_shifted
        ):
            continue
        mean_length = float(row["mean_episode_length"])
        candidates.append(
            {
                "suite": row["suite"],
                "task": row["task"],
                "perturbation": row["perturbation"],
                "standard_success": standard_rate,
                "shifted_success": shifted_rate,
                "mean_episode_length": mean_length,
                # Prefer strong preservation, then success near the middle of
                # the requested headroom interval, then shorter tasks.
                "rank_key": (-standard_rate, abs(shifted_rate - 0.7), mean_length),
            }
        )
    if not candidates:
        raise SystemExit("No candidate satisfies the configured selection rules.")
    candidates.sort(key=lambda item: item["rank_key"])
    selected = dict(candidates[0])
    selected.pop("rank_key")
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
