#!/usr/bin/env bash
# Creates .venv with PyTorch (CUDA) + timm etc. Skips if the env already works.
# Override the wheel index with TORCH_INDEX (default cu128; e.g. https://download.pytorch.org/whl/cu126).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}
check() { "$PY" -c "import torch, timm, yaml, scipy, matplotlib, PIL; assert torch.cuda.is_available(); print('torch', torch.__version__, 'timm', timm.__version__, torch.cuda.get_device_name())" ; }
if [ -x "$PY" ] && check 2>/dev/null; then echo "[setup] .venv ok"; exit 0; fi
if command -v uv >/dev/null 2>&1; then
  [ -x "$PY" ] || uv venv --python 3.11 .venv
  uv pip install --python "$PY" torch torchvision --index-url "$TORCH_INDEX"
  uv pip install --python "$PY" -r stage1/requirements.txt
else
  [ -x "$PY" ] || python3 -m venv .venv
  "$PY" -m pip install -q --upgrade pip
  "$PY" -m pip install torch torchvision --index-url "$TORCH_INDEX"
  "$PY" -m pip install -r stage1/requirements.txt
fi
check
