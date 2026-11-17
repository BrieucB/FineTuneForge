#!/usr/bin/env bash
# All accelerator imports see a CPU-only environment. No Slurm command may run.
set -euo pipefail
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${FTF_CPU_PYTHON:?Set FTF_CPU_PYTHON to a verified CPU-test interpreter}
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
deny_dir=$(mktemp -d)
trap 'rm -rf "$deny_dir"' EXIT
for command in sbatch srun salloc; do
    printf '#!/bin/sh\nprintf "Slurm commands are forbidden during CPU tests\\n" >&2\nexit 125\n' > "$deny_dir/$command"
    chmod 700 "$deny_dir/$command"
done
export PATH="$deny_dir:$PATH"
cd "$repo_dir"
"$python_bin" -m pytest "$@"
