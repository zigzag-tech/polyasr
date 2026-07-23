#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="${POLYASR_VENV:-${repo_dir}/cuda/venv}"
tmp_dir="${POLYASR_TMPDIR:-${repo_dir}/.tmp}"

# CUDA wheels can require multiple gigabytes of staging space. Do not depend
# on /tmp, which is commonly a much smaller tmpfs on GPU hosts.
mkdir -p "${tmp_dir}"
export TMPDIR="${tmp_dir}"
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-120}"
export PIP_RETRIES="${PIP_RETRIES:-10}"

if [[ ! -x "${venv_dir}/bin/python" ]]; then
    python3 -m venv "${venv_dir}"
fi

"${venv_dir}/bin/python" -m pip install -r "${repo_dir}/cuda/pytorch-cu129.txt"
"${venv_dir}/bin/python" -m pip install \
    --index-url "${POLYASR_PYPI_INDEX:-https://pypi.org/simple}" \
    -c "${repo_dir}/cuda/constraints-cu129.txt" \
    -r "${repo_dir}/cuda/requirements.txt"
"${venv_dir}/bin/python" -m pip check
"${venv_dir}/bin/python" "${repo_dir}/cuda/runtime_preflight.py"
