#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="${1:-${repo_dir}/cuda/venv-r2t2}"
livestack_dir="${2:-/home/ubuntu/livestack}"

uv venv "${venv_dir}" --python python3.12
uv pip install --python "${venv_dir}/bin/python" \
  -r "${repo_dir}/cuda/requirements-r2t2.txt"
uv pip install --python "${venv_dir}/bin/python" \
  -e "${livestack_dir}/node-py"
uv pip install --python "${venv_dir}/bin/python" maturin

(
  cd "${livestack_dir}/shared-py"
  VIRTUAL_ENV="${venv_dir}" \
  PYO3_PYTHON="${venv_dir}/bin/python" \
    "${venv_dir}/bin/python" -m maturin develop --release
)

"${venv_dir}/bin/python" "${repo_dir}/cuda/runtime_preflight.py"
