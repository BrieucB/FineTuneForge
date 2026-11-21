# Rounding-tolerant MLP fusion qualification

Status: planning only, 2026-11-18. This document authorizes no implementation,
GPU allocation, training pilot or production execution. Fusion remains disabled.
The user selected a separate rounding-tolerant qualification after reviewing the
strict preflight failure. This proposal is tracked by #16 and #19 in draft PR #23.

## Purpose and boundaries

Test whether the existing TorchInductor MLP execution path can reduce complete
LoRA update time while preserving useful gradients, optimizer behavior and the
posterior. The comparator is **eager execution with the qualified per-layer
parametrization cache**, so fusion must demonstrate value beyond caching.

Historical strict failures remain unchanged. Their maximum gradient difference
is not a basis for choosing a new tolerance. In particular, the retry with
Triton floating-point contraction disabled changed generated PTX but retained
the observed errors. The remaining cause is unresolved.

Retain the pinned checkpoint, architecture, PEFT implementation, all eight
members, FP32 neural computation, FP64 probability/decoding, math SDPA, full
shared-grid tail law, AdamW settings, clipping and effective batch of 16.
AMP, TF32, CUDA graphs, member batching, attention rewrites, optimizer fusion
and changes to the probability law are outside this qualification.

Freeze the existing 12-MLP compiler path: Inductor/default, `fullgraph=True`,
`dynamic=False`, `TRITON_DEFAULT_FP_FUSION=1` and CUDA graphs off. Record actual
generated-kernel options and source/environment identities. No compiler sweep
or private optimizer setting is part of this block.

The first block qualifies rank 4 on the declared small shapes only. Rank 16,
additional learning rates and larger shapes require the same checks before
fusion may be used there. This is a staged route to the already agreed r4/r16
study, not a reduction of its scientific comparison. Full fine-tuning remains
its original control; no compiled full-fine-tuning claim is made by this block.

## 1. Freeze the numerical contract before confirmation

Use a separate calibration split, then sign/hash a policy before opening fresh
confirmation results. Keep the old `atol=2e-7, rtol=2e-5` tensor comparisons as
historical and ongoing diagnostics. Never rewrite an old failure as a pass.

PyTorch 2.6 gives float32 comparison defaults of `atol=1e-5, rtol=1.3e-6`.
Evaluate those as a **candidate screening rule**, not as a scientific guarantee
or a complete acceptance criterion. Floating-point operation order can affect
results, as described in [PyTorch numerical accuracy](https://docs.pytorch.org/docs/2.6/notes/numerical_accuracy.html)
and [the pinned testing defaults](https://docs.pytorch.org/docs/2.6/testing.html).

The calibration deliverable must define exact numerical caps and aggregation
rules for the following, with a written rationale independent of the old failed
maxima. There is no claim that safe new backward/state caps are known today.
If calibration cannot justify and freeze them, stop before GPU confirmation.

| Check | Contract |
| --- | --- |
| Inventories and state | Exact parameter names, shapes, dtypes, base-weight freezing, counters, ordered data, RNG identities and module flags; finite values and no missing trainable gradients |
| Expected structural zeros | Preserve the zero-initialized LoRA-B to LoRA-A gradient dependency; distinguish structural zeros from numerical cancellation |
| Forward outputs | Keep logits `atol=1e-4, rtol=1e-3`; full-grid conditional CDF error at most `1e-4`; quantiles `atol=rtol=1e-3`; conditional NLL absolute difference at most `1e-3` |
| Gradients | Predeclare per-layer/branch absolute and norm-based caps before and after accumulation and clipping; report every entry, including near-zero coordinates |
| AdamW sensitivity | Compare first/second moments and actual parameter increments, normalized by learning rate and reference-step size with a declared floor; examine coordinates whose optimizer denominator is near `eps=1e-8` separately |
| Effective adapter | Compare `(alpha/r) B A`, not only its two factor matrices |
| Functional drift | Re-evaluate the unchanged forward/CDF checks after every matched update and at the final checkpoint |

A small gradient error alone is insufficient. At a fresh AdamW step with zero
weight decay, the increment is `-lr*g/(abs(g)+eps)`: changing `g=0` to `g=eps`
changes the increment by `lr/2`. The calibration therefore needs actual update
comparisons and retained near-zero diagnostics.

Use eager/eager and compiled/compiled repeats as execution controls. Applying
the same optimizer to identical frozen gradients must be an exact control.
Use isolated FP64 shadow arithmetic on fixed MLP operands, including backward,
to investigate expected FP32 error; this does not make the full model FP64 or
provide a gold-standard full-model gradient. Check calibration inputs span
initial and nonzero adapters. Do not hide failures behind an overall average,
mask small gradients, change AdamW epsilon, or rescale losses to pass a gate.

## 2. Separate calibration, engineering confirmation and scientific tests

Use the new seed domain `fusion-rounding-v1`, master seed `20261118`, with
separate calibration, confirmation-training, query, bank, initialization and
posterior substreams. Bind all arrays and manifests before candidate results.
These are new **diagnostic** cases; the study's validation and sealed test
catalogues remain untouched. All new synthetic forward calls are counted.

Historical zero-sensor data and prior gradient failures are development evidence.
Create fresh D4 sensor problems for confirmation: unknown population mean,
unknown population spread, one specimen property and unknown family noise.
Use ten readings per measurement vector and complete-hierarchy banks.

Two confirmation setups use different, fixed information levels:

- Strong sensor: ten known coefficients equal to 1.
- Weak sensor: ten alternating coefficients `+0.03` and `-0.03`.

Both have model `y_j = c_j*a + error_j` with independent Gaussian errors.
Use these fixed diagnostic priors in the declared parameter units:

- `mu ~ Normal(0, 0.6^2)`.
- `tau ~ LogNormal(log(0.2), s^2)` and `sigma ~ LogNormal(log(0.1), s^2)`,
  with `s = log(2)/Phi_inverse(0.95)`; the 90% prior interval is the median
  divided/multiplied by 2. These are proper distributions, not truncations.
- `a | mu,tau ~ Normal(mu, tau^2)`; both positive scales remain unknown.

The existing `PriorSpec` lognormal API takes physical mean and SD. Convert
median `m` and log-SD `s` to `mean=m*exp(s^2/2)` and
`sd=mean*sqrt(expm1(s^2))`; do not pass these medians as physical means.

Query parameters and noise are newly drawn, not chosen after candidate outputs. Use an independently seeded query/bank for each setup. Seal both cases
and an additional posterior random seed for the strong case before evaluation.

Use fresh C16 teacher-forced training episodes and C256 posterior banks initially.
One local call returns the ten-value measurement vector. Context and query
rows remain separate; true query values enter teacher-forced scores only.
The first GPU block uses one rank-4 initialization seed and the highest agreed
LoRA pilot learning rate, `3e-4`. These are bounded engineering updates, not LR
selection. Before later campaign use, repeat qualification for the other agreed
learning rate `1e-4`, both main training seeds, and rank 16. The main study still
requires the agreed matched examples and update allowance across methods.

## 3. Complete updates, restart and performance

From identical initial weights, data and RNG states, run eager/cache and
compiled/cache in independent worker processes with identical environments.
Probe every conditional at initial weights and an authenticated existing
nonzero checkpoint, retaining each checkpoint's own identity.

Then run **three complete matched updates** from the fresh initialization.
Every update uses 16 complete problems and all four conditional stages, followed
by exactly one optimizer step. Compare accumulated gradients, clipping, moments,
actual weight changes and probability checks on a separate development
monitor set. Final posterior confirmation queries remain sealed until the fixed
three-update checkpoints exist; they never select checkpoints or tolerances.
Preserve matching data
order; neither arm gains extra training data or updates.

Replay updates two and three after a fresh-process restart from update one.
Within each fixed execution policy/build/kernel plan require bitwise agreement
of weights, optimizer, future RNG, counters, data cursor and deployment outputs.
Cross-policy equality uses only the newly frozen rounding contract. A failure
is retained and blocks promotion; do not silently weaken restart requirements.

Keep the compiler scope valid through backward/checkpoint recomputation.
Authenticate export/reload, module restoration, model/optimizer keys and the
absence of stale weight-dependent caches. Record graph specialization and
actual kernel evidence; fail on graph breaks or unintended eager fallback.
Expected external matrix calls are not evidence of fallback by themselves.

Measure three **unprofiled**, synchronized warm update repetitions from the same
saved checkpoint and fixed next batch, restoring state for each repetition.
Keep these timing branches outside the three-update learning trajectory and
exclude them from posterior checkpoint selection. Record all their data/compute
costs. Capture CPU/CUDA traces separately from timed updates.

Report empty-cache compile/startup, first update, cache-hit restart, warm update
median/range, preparation, compute, peak allocated/reserved memory, kernel
counts and transfers/synchronization. Calculate compile-cost break-even in
updates. There is no fixed required speedup; retain the eager path if gains do
not justify complexity, startup or memory. A conditional microbenchmark is not
a complete-update performance result.

## 4. Posterior comparison and independent references

Use each arm's authenticated checkpoint after exactly three learning updates.
Deploy both through the **same eager merged inference path**. This isolates the
training effect; compiling posterior inference would require a separate study.

For each arm sample the two sealed confirmation cases, then repeat the strong
case at its predeclared additional posterior seed: **six requests in total,
50,000 complete vectors per request**. Within each pair use common random
uniforms, the same bank, parameter order, preprocessing and full-support law.
An invalid vector fails the request and preserves evidence; no clipping,
replacement, partial-row acceptance or smaller debugging sample count.

For the linear sensor, condition on spread and noise, integrate the Gaussian
population-mean/specimen-property block analytically, and refine a 2D quadrature
in log spread/log noise. Check transformed priors and Jacobians. Retain an
independent calculation and refinement witnesses before a reference is qualified.
Reference CDF precision remains at most `0.005`; the existing normalized mean
and SD precision limits remain `0.02`. References may use more internal samples
than 50,000. Account for CPU/reference costs and any allocated GPU work.

The current absolute-gate implementation requires authenticated empirical
`ReferenceRun` refinement/independent runs and valid uncertainty/ESS metadata.
Quadrature alone is an independent oracle, not an accepted receipt for that
API. Produce conforming refined numerical runs (for example the existing SMC
or NUTS reference paths), qualify them and cross-check against quadrature before
using the unchanged absolute gate. Never invent ESS from quadrature nodes or
pretend an unqualified oracle satisfies the existing reference contract. A new
analytic-reference adapter would be separate reviewed scope, not an implicit
change in this qualification.

Freeze the projection units and 32 joint directions independently of candidate
results. Use case-specific qualified reference SDs. The historical
`inference_engineering.compare_samples` hardcodes the zero-sensor case and is
**not suitable** for these new informative cases. A future comparator must
preserve the agreed thresholds without inheriting those scales.

Keep the investigation margins unchanged: `0.01` CDF difference, `0.05`
reference SD of mean shift and `5%` relative SD change. Retain marginal and
32-projection diagnostics, support and complete-row checks. Predeclare paired
whole-vector Monte Carlo uncertainty analysis: 1,000 paired bootstrap replicates
with its own fixed seed, simultaneous one-sided 95% upper uncertainty across
all case/seed/axis/projection metric maxima, and no pooling that hides a failed
case. These are approximate uncertainty diagnostics, not a universal coverage
guarantee. Include reference uncertainty; use the smaller qualified reference
SD when bounding SD-normalized mean shifts. Use only the additional seed already
listed above, not posthoc favorable reseeding. A bound that crosses a margin is
inconclusive even when its point estimate passes. Nominal independent-sample KS
p-values do not establish paired equivalence. No automatic repeated attempts
until a pass occurs.

Apply the original absolute gates separately to both variants: CDF error at
most `0.10`, absolute mean bias at most `0.2` reference SD and posterior SD ratio
within `[0.8,1.2]`, with the existing 32-projection gates. No new failure is
accepted on a case that the eager reference solves. An unresolved independent
reference prevents a scientific pass.

Report execution and science as separate statuses. Preserving a biased posterior
can support bounded execution agreement while scientific qualification remains
failed. Existing noise-posterior failure is retained. This small engineering
study cannot establish transfer, calibration across arbitrary hierarchies or
reduced simulator requirements.

## 5. Staging, cost forecast and artifacts

No reservations are created by this proposal. After implementation and numerical
policy review, an illustrative first block on A100 40 GB is:

| Sequential block | Allocation/walltime | Reserved GPU-hours |
| --- | --- | ---: |
| Rounding and shape preflight | 1 GPU, 15 minutes | 0.25 |
| Matched updates, fresh-process replay and warm timings | 2 GPUs, 45 minutes | 1.50 |
| Six C256 posterior requests | 2 GPUs, 20 minutes | 0.67 |
| Total first rank-4 block | Stop at any failed prerequisite | 2.42 |

These finite Slurm reservations are planning values, not predictions or a new
study cap. Existing small-case cached updates take roughly three minutes and
warm C256 requests roughly two minutes; the new data and compiler need fresh
timings. The new training inputs have ten readings, so the old tiny-case update
rate is not a measured forecast for this geometry. Use the first preflight to
reforecast the complete update/replay/timing block before reserving it; if it
does not fit 45 minutes, revise its walltime and cost rather than cutting work.
GPU workers are independent complete-model replicas, not distributed training.
References and any longer probes must be forecast before submission.

The complete rank/learning-rate/two-initialization extension is eight such blocks,
so roughly **10–20 allocated GPU-hours** is a provisional small-shape forecast,
including cold setup, replay and timing; revise it from the first measured block.
Do not submit the whole extension at once. No budget is moved or spent now.
Reserve and reconcile each later allocation against the existing 10,000-hour
study ledger, including failures and idle workers. Preserve the legacy ledger.

C4096 sampling, larger unknown counts, additional hierarchy shapes, production
CUDA campaign integration and full learning-rate pilots remain outside the first
block. Probe and reforecast those shapes before complete requests; never reduce
50,000 draws to make an oversized request fit. Scientific promotion to the
full study requires representative shape qualification, not extrapolation from D4.

Planned artifacts: a signed/hash-bound numerical policy; disjoint data manifests;
source, environment, checkpoint and compiler identities; per-layer gradients,
optimizer sensitivity and effective-adapter metrics; complete update/restart and
export receipts; 50,000-row arrays; case-specific reference witnesses; posterior
figures; cold/warm timings and traces; reconciled allocation costs; and separate
execution/scientific verdicts. Large raw artifacts stay outside Git with hashes.
Every prior failed diagnostic remains available.

The immediate next implementation step, only after execution is requested, is
the CPU calibration/metric harness and case-specific reference/comparison path.
GPU confirmation follows only once its numerical contract is frozen. This plan
does not enable fusion, lower posterior-quality gates or start training pilots.
