#!/usr/bin/env bash
# Resource and budget arguments belong to StudyBudgetLedger.submit.
set -euo pipefail
: "${SLURM_JOB_ID:?accounted Slurm allocation required}"
: "${SLURM_JOB_ACCOUNT:?Slurm account required}"
[[ "$SLURM_JOB_ACCOUNT" == eu-26-17 ]]
[[ "${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-0}}" == 1 ]]
[[ $# == 5 ]]
qualification_python="$1"
qualification_repo="$2"
qualification_inputs="$3"
qualification_sha="$4"
qualification_output="$5"
export PF_SCRATCH="${PF_SCRATCH:-/scratch/project/eu-26-17/eubrieucb/posteriorforge}"
export PYTHONPATH="$qualification_repo/src"
export HF_HOME="$PF_SCRATCH/cache/huggingface"
export TORCH_HOME="$PF_SCRATCH/cache/torch"
export XDG_CACHE_HOME="$PF_SCRATCH/cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TABPFN_DISABLE_TELEMETRY=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TMPDIR="$PF_SCRATCH/work/study-qualification-$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
exec "$qualification_python" "$qualification_repo/scripts/qualify-study-cuda.py" run --repo "$qualification_repo" --inputs "$qualification_inputs" --input-sha256 "$qualification_sha" --output "$qualification_output"
