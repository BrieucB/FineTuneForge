#!/usr/bin/env bash
# Execute the bounded, offline training-attention check inside its reserved job.
set -euo pipefail

if (( $# != 6 )); then
    printf '%s\n' 'Usage: run-study-training-attention-check.sh PYTHON REPO CONFIG OUTPUT LEDGER ATTEMPT' >&2
    exit 2
fi
: "${SLURM_JOB_ID:?accounted Slurm allocation required}"
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ ]]
[[ "${SLURM_JOB_ACCOUNT:-}" == eu-26-17 && "${SLURM_JOB_PARTITION:-}" == qgpu ]]

export PF_SCRATCH=/scratch/project/eu-26-17/eubrieucb/posteriorforge
check_python="$1"
[[ "$check_python" == "$PF_SCRATCH/envs/study-training-gpu/bin/python" && -x "$check_python" ]]
check_repo=$(cd -- "$2" && pwd -P)

export HF_HOME="$PF_SCRATCH/cache/huggingface" TORCH_HOME="$PF_SCRATCH/cache/torch" XDG_CACHE_HOME="$PF_SCRATCH/cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TABPFN_DISABLE_TELEMETRY=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 NVIDIA_TF32_OVERRIDE=0 TORCH_COMPILE_DISABLE=1
export TABPFN_ALLOW_CPU_LARGE_DATASET=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TMPDIR="$PF_SCRATCH/work/study-training-attention-check-$SLURM_JOB_ID"
mkdir -p -- "$TMPDIR"

unset PYTHONHOME
export PYTHONPATH="$check_repo/src"
export PYTHONSAFEPATH=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
PYTHONPYCACHEPREFIX=$(mktemp -d -- "$TMPDIR/pycache.XXXXXXXX")
export PYTHONPYCACHEPREFIX

exec "$check_python" "$check_repo/scripts/check-study-training-attention.py" \
    --config "$3" --output "$4" --ledger "$5" --attempt "$6"
