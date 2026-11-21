#!/usr/bin/env bash
# Account resources exclusively through StudyBudgetLedger.submit.
set -euo pipefail
: "${SLURM_JOB_ID:?accounted allocation required}"
[[ "${SLURM_JOB_ACCOUNT:-}" == eu-26-17 ]]
[[ $# == 7 ]]
qualification_python="$1"
qualification_repo="$2"
export PYTHONPATH="$qualification_repo/src"
export PF_SCRATCH="${PF_SCRATCH:-/scratch/project/eu-26-17/eubrieucb/posteriorforge}"
export HF_HOME="$PF_SCRATCH/cache/huggingface"
export TORCH_HOME="$PF_SCRATCH/cache/torch"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TABPFN_DISABLE_TELEMETRY=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TABPFN_ALLOW_CPU_LARGE_DATASET=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
exec "$qualification_python" "$qualification_repo/scripts/qualify-study-cuda.py" probes --repo "$qualification_repo" --inputs "$3" --input-sha256 "$4" --output "$5" --campaign-checkpoint "$6" --checkpoint-manifest-sha256 "$7"
