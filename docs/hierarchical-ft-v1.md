# Hierarchical fine-tuning: CPU software stage

Built with PriorLabs-TabPFN.

FineTuneForge adds a variable-size study namespace while preserving the historical
`posteriorforge` package, commands, source evidence, and 200-hour ledger. Scope and
acceptance criteria live in [issue #9](https://github.com/BrieucB/FineTuneForge/issues/9).
The new study has a separate 10,000 allocated GPU-hour planning ceiling; it is not
authorization to run jobs. Its ledger is separate from historical accounting, and
`posteriorforge study submit` currently refuses execution. The current implementation
includes file-backed CPU diagnostic orchestration for
bounded training, complete-posterior sampling, evaluation, and selection sealing. There
have been no GPU jobs, full training campaign, or actual 50,000-draw neural posterior
run. Production GPU execution and scientific qualification remain pending. CPU checks
do not qualify GPU behavior, scientific reference convergence, or posterior accuracy.

Each experiment/model is limited to P <= 6 local unknown parameters. This is not a cap on
the union across families for a shared specimen: it may involve more than six distinct
physical quantities overall. The global limits are 96 unique unknowns and 450 combined
measurements plus unknowns.

Before production training, the protocol requires consistent full-tail scoring and
sampling for both the original pretrained-weight baseline and the LoRA/full-fine-tuning
variants. Full-tail production qualification I07 remains pending. The legacy finite-v3
scoring/sampling path stays isolated as a regression control; the old benchmark is
unchanged. Do not treat implementing a candidate CPU primitive as qualifying the law or
training policy.

## Reproduce the CPU environments

`environments/study-training-cpu.lock` is a complete hashed CPU-only dependency
set with Torch 2.6.0+cpu, TabPFN 2.0.9, PEFT and Transformers. The independent
`study-reference-cpu.lock` uses JAX/NumPyro on CPU. These environments do not
modify the historical inference or reference environments and do not qualify
GPU execution. Prepare them on a network-enabled login/preparation host, then
use cached inputs offline. For example, with the pinned bootstrap uv executable:

```bash
export PF_SCRATCH=/path/to/private/posteriorforge
export UV_CACHE_DIR="$PF_SCRATCH/cache/uv"
/usr/bin/python3.11 -m venv "$PF_SCRATCH/envs/study-training-cpu"
"$PF_SCRATCH/envs/bootstrap/bin/uv" pip sync \
  --python "$PF_SCRATCH/envs/study-training-cpu/bin/python" --require-hashes \
  --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
  environments/study-training-cpu.lock
/usr/bin/python3.11 -m venv "$PF_SCRATCH/envs/study-reference-cpu"
"$PF_SCRATCH/envs/bootstrap/bin/uv" pip sync \
  --python "$PF_SCRATCH/envs/study-reference-cpu/bin/python" --require-hashes \
  environments/study-reference-cpu.lock
FTF_CPU_PYTHON="$PF_SCRATCH/envs/study-training-cpu/bin/python" \
  bash scripts/test-study-cpu.sh tests/test_study_*.py
FTF_CPU_PYTHON="$PF_SCRATCH/envs/study-reference-cpu/bin/python" \
  bash scripts/test-study-cpu.sh tests/test_study_reference.py tests/test_study_metrics.py
```

The test wrapper masks accelerators and places refusing Slurm command shims first
on PATH. CI checks both hashed study CPU locks (training and reference) and never
downloads a pretrained model.
Optional real-checkpoint tests require the already prepared pinned checkpoint;
CPU gradient tests are not a training campaign or posterior-accuracy result.

## Implemented interfaces

The additive source entrypoint is `python -m posteriorforge.study`; the installed
`posteriorforge study` namespace preserves the historical `python -m posteriorforge`
entrypoint. Planning and request files are configuration-driven. The implemented
commands and arguments are:

```text
train-plan [--config FILE]
train --config FILE --output DIR --cpu-diagnostic [--resume-from CHECKPOINT] [--stop-after-update N]
posterior-plan --config FILE
posterior --config FILE --output DIR --cpu-diagnostic [--resume]
evaluation-plan --config FILE --output FILE
evaluate --config FILE --output DIR --cpu-diagnostic [--resume]
seal-selection --records FILE --manifest-sha256 SHA --plan-sha256 SHA \
  --test-catalogue-sha256 SHA --output FILE
inspect-model --base-checkpoint FILE [--campaign-checkpoint DIR]
```

Training and posterior execution require explicit CPU-diagnostic opt-in. Training
requests are file-backed `finetuneforge.training_request.v1` objects; posterior
requests are `finetuneforge.posterior_request.v1`. Evaluation-plan and evaluation
requests are also file-backed and bind their input files by SHA256. The schemas
and execution policy are defined in `study/workflows.py`; evaluation records and
selection locks bind the plan, model, bank, query, and catalogue identities. These
commands do not submit scheduler jobs.

Training orchestration uses fresh context banks and fresh independent one-case
queries on every visit, with identical data streams per method for each seed and
phase. Quick validation is fixed at 40 cases, two banks and four context sizes
(320 scores, no posterior draws). The pilot is 200 updates and selects one learning
rate per method from the mean endpoint score across both seeds. Main training restarts
from pretrained weights with a fresh adapter; pilot updates and costs are separate.
The bounded CPU diagnostic is limited to 10 updates. The gated plans contain
12 pilot campaigns and six main campaigns. Main update-count candidates are
1,000, 3,000 and 10,000; none of these full campaigns has run.

The posterior path requires exactly 50,000 complete vectors. Invalid decoding fails
the request without retry or clipping; recovery binds source, model, bank and query
identities. Model loading accepts the pinned base checkpoint with a separately
cached safetensors full-model deployment or PEFT adapter directory. Evaluation can
seal diagnostic inputs when no qualified primary exists, without making qualification
or accuracy claims. The historical `plan`, `catalogue`, `prepare-bank`, `prior-check`,
`budget` and disabled `submit` commands remain available.

Each `ProblemSpec` records parameter priors and roles, experiment-family noise levels,
controls, and specimen/sharing identities. Positive log-normal parameters use physical
means and SDs. Parents precede descendants; parameter/observation order and all known
units are included in the fingerprint. Populations are independent conditional on the
declared hierarchy; posterior dependence is retained. `prepare-bank` writes authenticated
complete paired arrays; `prior-check` writes exactly 50,000 prior vectors with zero
forward calls. These are diagnostic samples, not a posterior pass. Bank counts are not
forward-call counts: actual distinct local evaluations, valid deterministic reuse, and
cumulative prefix costs are recorded separately.

Production qualification remains incomplete. There is no GPU engine qualification,
full training campaign, actual 50,000-draw neural posterior run, qualified production
reference, or final scientific outcome. Full-tail production qualification I07 remains
pending. CPU staging or toy sampler behavior does not establish model quality, GPU
memory/runtime, reference convergence, or posterior accuracy. Legacy finite-v3
behavior remains a separate regression control.

## Ownership and remaining qualification

At most three disjoint workers operate alongside the coordinator. Science,
reference/evaluation and model/training files have separate owners. Shared
contracts, CLI/configuration, locks, integration, commits and any future compute
reservations belong to the coordinator. Dependent edits require an explicit
handoff; workers never submit duplicate campaigns.

[The study epic](https://github.com/BrieucB/FineTuneForge/issues/9) contains the
complete scope and linked issue acceptance criteria. Physical/reference
convergence, actual 50,000-draw neural requests, GPU memory/runtime, multi-GPU
scaling, full training, sealed-test comparisons and model publication remain
unrun. A personal Hugging Face account is needed only before model release.

For a public wheel smoke reproduction, `environments/study-public-cpu.lock` pins
only NumPy, SciPy and pytest plus their dependencies. This checks that the public
scientific/CLI core installs independently of the model or private repositories.
The builder uses `study-training-cpu.lock`; a wheel can then be installed with
`--no-deps` into the minimal hashed environment. Model-dependent and NumPyro
checks belong to the two dedicated environments above.

The CPU-only `study.updates.stagewise_cpu_update` primitive releases each
conditional's gradient graph before processing the next one, then takes one
optimizer step for the complete effective batch. Its loss matches the existing
role-balanced full-graph calculation, including each coordinate's Jacobian.
Every realized query case has equal weight; setup frequency follows catalogue
sampling. This provides a tested accumulation path, not a production campaign
or a measured GPU memory/runtime improvement.


The examples in `configs/examples/hierarchical-ft-v1/` contain placeholder paths
and checksums. Replace them with prepared files and identities from
`study inspect-model`; JSON paths resolve relative to the request file.
`configs/studies/hierarchical-ft-v1-orchestration.json` records the settled
orchestration policy and its combined scientific/protocol fingerprint.

Checkpoint registry `training_seed` labels identify replica 0/1. The actual model
initialization and data-generation seeds are separately retained in checkpoint
and stream receipts. A frozen model's checkpoint identity is the immutable base
checkpoint-file SHA256; a trained model's identity is its campaign manifest SHA256.

`scripts/check-study-orchestration-cpu.py --output /fresh/output/path` produces an
exact analytical control with 50,000 complete vectors, inferred measurement noise,
a PNG and JSON evidence. Run it with the CPU environment and `PYTHONPATH=src`.
It uses an explicitly identified analytical backend, not transformer weights.
Its numerical agreement tests orchestration; it cannot establish transformer
accuracy, transfer, simulation savings or GPU throughput.

For a frozen-model inference request, `model` contains only `base_checkpoint`
and `model_sha256`. A trained request additionally supplies `campaign_checkpoint`
and its `checkpoint_manifest_sha256`. Evaluation registries also record a frozen
base's checkpoint identity, but that is distinct from supplying a campaign
manifest to the inference loader.
