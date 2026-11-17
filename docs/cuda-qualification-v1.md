# First CUDA engineering qualification

Built with PriorLabs-TabPFN.

This is the explicitly authorized first GPU qualification stage for the variable-size
hierarchical study. It does not enable production training or final-test selection.
The existing CPU entrypoints remain CPU-only; `scripts/qualify-study-cuda.py` is a
separate, source-bound qualification entrypoint.

The pilot uses four independent processes, each with one visible A100 and the full
model. The lanes are frozen inference, LoRA-r4, LoRA-r16 and full fine-tuning. This
parallelizes independent checks, without claiming distributed training or tensor
parallelism. All eight regression ensemble members remain enabled. Neural execution
is deterministic FP32 with TF32 and AMP disabled; probability calculations and
physical decoding retain float64.

The frozen lane compares all eight raw member predictions, a declared CDF grid,
known-value density scores and seven central/tail quantiles against CPU execution.
The trained lanes each perform two optimizer updates with effective batch16, then
repeat through an interruption after update1. Model weights, optimizer state, future
Python/NumPy/TorchCPU/CUDA/episode RNG draws and data position must agree bitwise.
All methods receive the same32 fresh context/query banks. These engineering fixtures
are separate from the agreed200-update learning-rate trials; no rate is selected.

Frozen and briefly updated rank4 weights each generate exactly50,000 complete
parameter vectors on the declared four-unknown analytical development control using
256 context worlds. Raw arrays, recovery state, traces and failures remain outside
Git. A neural-output figure overlays exact analytical references without claiming
that the transformer passed the full study screen. Invalid decoding fails and is
never clipped, retried or replaced automatically.

Rank16 and full-fine-tuning workers also probe last-parameter conditionals at
representative medium/large sizes and contexts1024/4096. An OOM is retained as an
unsupported shape, not repaired by silently shrinking the input. These are inference
memory probes, not a full large-problem training/inference qualification.

Prepare public dependencies with the separate hashed GPU lock, cache the pinned base
checkpoint, commit the source, and archive that exact commit into an immutable job
snapshot. Run `prepare` on CPUs, retain its manifest SHA256, and submit only through
`StudyBudgetLedger.submit`. Compute workers verify snapshot source/lock bytes and
all input/model hashes, use cached dependencies offline, and retain every attempt.

The previous proposed four-hour pilot ceiling is removed by the user's execution
instruction. Each Slurm allocation still requires a finite site-supported walltime
and a durable reservation against the separate10,000 GPU-hour project cap. Charge
actual allocated GPUs times elapsed time, including idle workers, failures and
retries. The old200-hour ledger remains untouched. No automatic scheduler requeue.

The whole I07 scientific/reference qualification, BF16 qualification,200-update
learning-rate trials, production campaigns and final transfer comparisons remain
separate work after this engineering pilot's actual evidence is reviewed.

## Inference-only memory and adapter qualification

The follow-up `scripts/engineer-study-inference.py` entrypoint fixes the exact
75-unknown, 200-observation, 4,096-world conditional that failed in the first pilot.
It also compares frozen, unmerged rank4 and merged rank4 deployments on the same
four-unknown control, with exactly 50,000 complete vectors per request. Three
repeated requests at 256 worlds support median/range timing; the four 4,096-world
controls are single repetitions. These are engineering fixtures using the retained
two-update adapter, not training pilots or a transfer experiment.

Two changes are opt-in and leave the training path unchanged:

- `study.cuda_attention.install_tiled_attention` preserves small calls on the stock
  path and tiles large inference calls over batch and query rows. Each query still
  attends to every key. Its default 512 MiB limit bounds one score tensor, not
  total GPU memory. FP32 math attention, all eight members and the complete context
  remain enabled; gradient-enabled calls use the original implementation.
- `study.inference_deployment.load_frozen_deployment(mode="merged_lora")` starts
  from an authenticated fresh checkpoint and calls Hugging Face PEFT's ordinary
  merge operation, then resolves TabPFN's packed-weight compatibility layers into
  ordinary weights. Model and checkpoint source identities remain separate from
  the actual merged deployment hash. Original checkpoints are never overwritten.

Run `prepare` on CPUs against a committed source snapshot and retain its manifest
SHA256. `run --phase profile` launches four independent local GPU workers;
`run --phase context4096` launches the four complete-posterior controls. Both must
run within an accounted Slurm allocation submitted through the study ledger.
`slurm/run-study-inference.sh` is the offline launcher. No optimizer updates,
training-pilot launch or full 75-dimensional posterior is enabled by these phases.

The CPU-only `scripts/report-inference-engineering.py --root RUN --output FRESH`
authenticates complete samples and matching inputs before comparing distributions.
It retains absolute analytical posterior errors, including failures, and reports
preservation separately from scientific accuracy. Use `--partial` only for a report
that explicitly marks unfinished 4,096-world controls as pending.

PyTorch CPU/CUDA traces, setup/merge costs, per-stage timings, attention tile
counters, completed arrays and source/model/input hashes remain in the raw run.
Independent worker processes accelerate separate checks; their GPU memory is not
pooled. Capacity probes and a conditional-cost forecast do not establish complete
large-problem sampling latency or multi-GPU scaling.
