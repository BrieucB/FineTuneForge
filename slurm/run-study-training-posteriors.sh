#!/usr/bin/env bash
# Resource requests and accounting are owned by StudyBudgetLedger.submit.
set -euo pipefail
[[ "${SLURM_JOB_ACCOUNT:-}" == eu-26-17 && -n "${SLURM_JOB_ID:-}" ]]
[[ "${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-0}}" == 1 && $# == 6 ]]
posterior_python="$1"; posterior_repo="$2"; posterior_inputs="$3"
posterior_sha="$4"; posterior_commit="$5"; posterior_output="$6"
export PF_SCRATCH="${PF_SCRATCH:-/scratch/project/eu-26-17/eubrieucb/posteriorforge}"
export PYTHONPATH="$posterior_repo/src"
export HF_HOME="$PF_SCRATCH/cache/huggingface" TORCH_HOME="$PF_SCRATCH/cache/torch" XDG_CACHE_HOME="$PF_SCRATCH/cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TABPFN_DISABLE_TELEMETRY=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TABPFN_ALLOW_CPU_LARGE_DATASET=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TMPDIR="$PF_SCRATCH/work/study-training-posterior-$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
exec "$posterior_python" "$posterior_repo/scripts/qualify-training-posteriors.py" run --inputs "$posterior_inputs" --input-sha256 "$posterior_sha" --repo "$posterior_repo" --source-commit "$posterior_commit" --output "$posterior_output"
