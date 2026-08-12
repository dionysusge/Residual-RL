# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import os
import tempfile
from pathlib import Path
from typing import Any

import torch


class AtomicResidualTraceWriter:
    """Write rollout-local, offline-readable trace shards atomically."""

    def __init__(self, root: str | Path, run_id: str, rank: int) -> None:
        self.directory = Path(root) / run_id / f"rank_{rank}"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.part = 0

    def write(self, records: list[dict[str, Any]]) -> Path:
        """Persist one shard; callers choose chunking to avoid per-step IPC."""
        target = self.directory / f"part_{self.part:06d}.pt"
        fd, temporary = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=self.directory
        )
        os.close(fd)
        try:
            torch.save(records, temporary)
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        self.part += 1
        return target
