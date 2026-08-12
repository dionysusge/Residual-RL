#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Compute a deterministic manifest hash for a checkpoint file or directory."""

import argparse
import hashlib
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    root = args.path.resolve()
    files = (
        [root]
        if root.is_file()
        else sorted(path for path in root.rglob("*") if path.is_file())
    )
    digest = hashlib.sha256()
    for path in files:
        relative = path.name if root.is_file() else path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
    print(f"sha256 {digest.hexdigest()}  {root}")


if __name__ == "__main__":
    main()
