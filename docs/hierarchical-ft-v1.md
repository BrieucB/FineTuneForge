# Hierarchical fine-tuning: CPU software stage

Built with PriorLabs-TabPFN.

FineTuneForge adds a variable-size study namespace while preserving the historical
`posteriorforge` package, commands, source evidence, and 200-hour ledger. Scope and
acceptance criteria live in [issue #9](https://github.com/BrieucB/FineTuneForge/issues/9).
The new study has a separate 10,000 allocated GPU-hour planning ceiling; it is not
authorization to run jobs. Its ledger is separate from historical accounting, and
`posteriorforge study submit` currently refuses execution. The current phase is code and
CPU checks: there are no GPU jobs, production training, complete 50,000-draw neural
posterior runs, qualified production references, or claimed accuracy gains. CPU tests
check implementation components; they do not qualify GPU behavior, scientific reference
convergence, or posterior accuracy.

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

```bash
PYTHONPATH=src "$PF_SCRATCH/envs/study-training-cpu/bin/python" -m posteriorforge.study plan \
  --config configs/studies/hierarchical-ft-v1.json
PYTHONPATH=src "$PF_SCRATCH/envs/study-training-cpu/bin/python" -m posteriorforge.study catalogue \
  --split validation --count 2 --output /fresh/path/definitions.json
```

After installation, the `posteriorforge study ...` console command uses this additive
entrypoint. For source checkouts, invoke `python -m posteriorforge.study ...`
as above. The historical `python -m posteriorforge` module entry remains unchanged.

Planning emits exact request counts and unknown measured costs without submitting
work. The CLI also offers `inspect-spec`, `catalogue`, `prepare-bank`, `prior-check`,
`budget`, and the disabled `submit` command. Catalogue generates definitions only, not
observations or inference results. Each `ProblemSpec` records parameter priors and roles, experiment-family noise levels,
controls, and specimen/sharing identities. Positive log-normal parameters use
physical means and SDs. Parents precede descendants; parameter/observation order
and all known units are included in the fingerprint. Populations are independent
conditional on the declared hierarchy; posterior dependence is retained.

`study prepare-bank` writes authenticated complete paired arrays from a spec.
`study prior-check` writes exactly 50,000 prior vectors with zero forward calls;
these are samples to assess against a reference, not a claimed posterior pass.
Both commands reject final-test specs at this CPU stage. Final test remains
sealed until the selection/backend protocol is locked. Bank counts are not
forward-call counts: the generator retains actual distinct local evaluations,
valid deterministic reuse, and cumulative prefix costs separately.

Reference, evaluation, PEFT and training primitives are Python APIs under the
study namespace. Some components have CPU checks, but the full orchestration and
production qualification remain incomplete. There is no qualified production
reference, training campaign, GPU engine, or final scientific outcome. Full-tail
production qualification I07 remains pending. No component check, configuration
flag, or old benchmark evidence substitutes for the required qualification.

The full-tail policy must be applied consistently to scoring and sampling for the
pretrained-weight baseline and both adaptation variants before production training.
Legacy finite-v3 behavior remains available only as a separate regression control.
Do not infer a settled distribution identifier from these instructions; use the
identifier recorded by the implementation when it is finalized.

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
