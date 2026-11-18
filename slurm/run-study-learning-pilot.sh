#!/usr/bin/env bash
# Run with the prepared study-training-gpu interpreter inside the reserved job.
set -euo pipefail

if (( $# < 5 )); then
    printf '%s\n' 'Usage: run-study-learning-pilot.sh PYTHON REPO CONFIG OUTPUT LEDGER [--resume-from PATH] [--stop-after-update N]' >&2
    exit 2
fi
: "${SLURM_JOB_ID:?accounted Slurm allocation required}"
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ ]]
[[ "${SLURM_JOB_ACCOUNT:-}" == eu-26-17 && "${SLURM_JOB_PARTITION:-}" == qgpu ]]

pilot_python="$1"
pilot_repo=$(cd -- "$2" && pwd -P)
pilot_config="$3"
pilot_output="$4"
pilot_ledger="$5"
shift 5

export PF_SCRATCH="${PF_SCRATCH:-/scratch/project/eu-26-17/eubrieucb/posteriorforge}"
export HF_HOME="$PF_SCRATCH/cache/huggingface" TORCH_HOME="$PF_SCRATCH/cache/torch" XDG_CACHE_HOME="$PF_SCRATCH/cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TABPFN_DISABLE_TELEMETRY=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 NVIDIA_TF32_OVERRIDE=0 TORCH_COMPILE_DISABLE=1
export TABPFN_ALLOW_CPU_LARGE_DATASET=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TMPDIR="$PF_SCRATCH/work/study-learning-pilot-$SLURM_JOB_ID"
mkdir -p -- "$TMPDIR"

unset PYTHONHOME
export PYTHONPATH="$pilot_repo/src"
export PYTHONSAFEPATH=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
PYTHONPYCACHEPREFIX=$(mktemp -d -- "$TMPDIR/pycache.XXXXXXXX")
export PYTHONPYCACHEPREFIX

exec "$pilot_python" -m posteriorforge.study.entrypoint study train-cuda-pilot \
    --config "$pilot_config" --output "$pilot_output" --ledger "$pilot_ledger" "$@"
