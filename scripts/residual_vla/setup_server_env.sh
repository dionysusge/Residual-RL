#!/usr/bin/env bash
set -euo pipefail

RLINF="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${RLINF}"

# The locked installer creates a project-local environment. liberoplus also
# installs the standard RLinf LIBERO package; no shell or global pip config is
# modified by this wrapper.
bash requirements/install.sh embodied \
  --model openpi \
  --env liberoplus \
  --venv .venv \
  --python 3.11.14 \
  --install-rlinf

. .venv/bin/activate
python - <<'PY'
import importlib.metadata as metadata
import platform
import torch

packages = ("torch", "mujoco", "rlinf-libero", "rlinf-liberoplus", "openpi-client")
print("python", platform.python_version())
print("cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("gpu_count", torch.cuda.device_count())
for package in packages:
    try:
        print(package, metadata.version(package))
    except metadata.PackageNotFoundError:
        print(package, "NOT_INSTALLED")
PY
