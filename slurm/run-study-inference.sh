#!/usr/bin/env bash
# Resources and budget reservation are exclusively supplied by StudyBudgetLedger.
set -euo pipefail
: "${SLURM_JOB_ID:?accounted Slurm allocation required}"
[[ "${SLURM_JOB_ACCOUNT:-}" == eu-26-17 ]]
[[ $# == 6 ]]
qualification_python="$1"
qualification_repo="$2"
export PYTHONPATH="$qualification_repo/src"
export PF_SCRATCH="${PF_SCRATCH:-/scratch/project/eu-26-17/eubrieucb/posteriorforge}"
export HF_HOME="$PF_SCRATCH/cache/huggingface" TORCH_HOME="$PF_SCRATCH/cache/torch"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TABPFN_DISABLE_TELEMETRY=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TABPFN_ALLOW_CPU_LARGE_DATASET=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
exec "$qualification_python" "$qualification_repo/scripts/engineer-study-inference.py" run --repo "$qualification_repo" --inputs "$3" --input-sha256 "$4" --output "$5" --phase "$6"
