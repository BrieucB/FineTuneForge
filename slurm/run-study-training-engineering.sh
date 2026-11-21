#!/usr/bin/env bash
# Resource requests and accounting are owned by StudyBudgetLedger.submit.
set -euo pipefail
: "${SLURM_JOB_ID:?accounted Slurm allocation required}"
: "${SLURM_JOB_ACCOUNT:?Slurm account required}"
[[ "$SLURM_JOB_ACCOUNT" == eu-26-17 ]]
[[ "${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-0}}" == 1 ]]
if [[ "${FTF_ENGINEERING_PHASE:-run}" == fusion-preflight ]]; then [[ $# == 9 ]]; else [[ $# == 6 ]]; fi
engineering_python="$1"; engineering_repo="$2"; engineering_inputs="$3"
engineering_sha="$4"; engineering_commit="$5"; engineering_output="$6"
export PF_SCRATCH="${PF_SCRATCH:-/scratch/project/eu-26-17/eubrieucb/posteriorforge}"
export PYTHONPATH="$engineering_repo/src"
export HF_HOME="$PF_SCRATCH/cache/huggingface" TORCH_HOME="$PF_SCRATCH/cache/torch" XDG_CACHE_HOME="$PF_SCRATCH/cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TABPFN_DISABLE_TELEMETRY=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TABPFN_ALLOW_CPU_LARGE_DATASET=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TMPDIR="$PF_SCRATCH/work/study-training-engineering-$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
if [[ "${FTF_ENGINEERING_PHASE:-run}" == fusion-preflight ]]; then
    export TORCHINDUCTOR_CACHE_DIR="$PF_SCRATCH/cache/inductor-fusion-$SLURM_JOB_ID"
    mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
    exec "$engineering_python" "$engineering_repo/scripts/engineer-study-training.py" fusion-preflight --repo "$engineering_repo" --inputs "$engineering_inputs" --input-sha256 "$engineering_sha" --source-commit "$engineering_commit" --output "$engineering_output" --checkpoint "$7" --checkpoint-sha256 "$8" --checkpoint-source-commit "$9"
fi
exec "$engineering_python" "$engineering_repo/scripts/engineer-study-training.py" "${FTF_ENGINEERING_PHASE:-run}" --repo "$engineering_repo" --inputs "$engineering_inputs" --input-sha256 "$engineering_sha" --source-commit "$engineering_commit" --output "$engineering_output"
