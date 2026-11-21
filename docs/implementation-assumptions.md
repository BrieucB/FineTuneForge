# FineTuneForge implementation assumptions

This is the register for the hierarchical fine-tuning study. Record every new
or revised implementation assumption here before it drives a code change,
benchmark, or qualification decision. Give it an ID, date, scope, status,
and source or evidence. Use **open** for choices or operating ranges that
still need verification. Keep superseded records and link their replacements.
The [study contract](hierarchical-ft-v1.md) and [study epic](https://github.com/BrieucB/FineTuneForge/issues/9)
contain the complete scientific protocol.

## A01 · Conditional sampling for budgeted rank-4 pilots

Approved 2026-11-20. Use **50% role-stratified conditional sampling** for
budgeted rank-4 LoRA learning pilots. Run the full set when a comparison
explicitly calls for an exhaustive training control.

- Use all 16 realized cases per optimizer update. Within each case, sample
  uniformly without replacement in each of the three role groups
  (population/batch, physical, and noise). For a group with `n` conditionals,
  select `k = ceil(n / 2)`, so every group contributes and the realized
  fraction can exceed 50%.
- Keep the case's original conditional order and all true predecessor values
  for each selected stage. Evaluate all eight ensemble members and score the
  same full-support density, including the decoder Jacobian.
- Weight each selected stage's loss by `n / k`, using the original role
  denominator and global case count. Sum rank-local gradients before the one
  global clip and AdamW update. The resulting pre-clipping gradient targets
  the exhaustive objective in expectation.
- Use `role_stratified_half_episode_sha256_shuffle.v1` in the CUDA pilot
  request. For each role, seed a private Python shuffle with the SHA-256 of
  `conditional-pilot-v1/<episode request ID>/<role index>`. Role indices
  follow the order above. Keep the first `k` indices in each shuffle and
  restore the original parameter order for scoring. The episode identity
  includes the stream and data cursor.
- Retain the rate and version in the request and checkpoint identity. Keep
  the emitted selection plan, including stage indices and original and
  selected role counts, in the checkpoint history and update receipt.
  Score all conditionals in held-out evaluation.

The matched-budget qualification used FP32 rank-4 LoRA at `3e-4` and
three independent paired training seeds. It scored 20 fresh setups at each of
16, 64, 256, and 1,024 context rows, with equal weight across contexts and
setups. At the 12-GPU-hour checkpoint horizon the sampled-minus-exhaustive
mean negative log density was `-0.00482`, with a one-sided 95% upper bound
of `+0.00332` against the predeclared `+0.005` margin. The sampled arms
completed five updates each, versus two, two, and three for the exhaustive
arms. The bounds for 256 and 1,024 rows were `+0.00822` and `+0.01059`.
Quality at those sizes and use in the full scientific campaign are open.

The [public replication receipt](../evidence/hierarchical-ft-v1-conditional-sampling/conditional-sampling-replication-results.json)
includes the paired score differences, training and validation accounting,
and the SHA256 of the archived detailed receipt.
Training jobs were `5468572`–`5468577`. Validation jobs were `5468670`,
`5468676`, and `5468679`. The score comparison used the last completed
checkpoint within each budget. The allocated time includes initialization
and the cost of any unfinished update, including an in-flight conditional
that crossed the training deadline.

### Native restart verification

The CUDA command path reproduced the selected stages, held-out score,
weights, and optimizer state in the CPU integration test. On one A100
(job `5468890`), the pinned TabPFN backend reproduced the selection, loss,
gradient norm, all 422 model tensors, and all 732 AdamW state tensors
after a checkpoint restore. The small fixture used 16 context rows and
selected 48 of 80 conditionals per update because of within-role rounding.
The [public A100 restart receipt](../evidence/hierarchical-ft-v1-conditional-sampling/native-sampled-restart-a100.json)
records timing, the checkpoint and source hashes, the original receipt's SHA256,
and the first attempt, which stopped during the verifier's device comparison.

### Wide 1,024-row restart verification

Two independent rank-4 fixtures used one retained 1,024-row case and
fifteen 16-row controls per measured update. Each restart ran in a
fresh process on another A100. Both cases reproduced the sampled stages,
loss, gradient norm, random-number states, 422 model tensors, and 732
AdamW tensors exactly. The selected wide cases contained 21 of 42 and
35 of 69 conditionals. Their peak PyTorch allocations were 9.61 and
18.65 GiB, and peak reservations were 24.38 and 38.90 GiB. The wider
fixture left 0.61 GiB between peak PyTorch reservation and the device's
reported capacity. The four jobs reconciled to 1.586 allocated GPU-hours.

The [wide-case receipt](../evidence/hierarchical-ft-v1-conditional-sampling/wide-sampled-a100-restart-results.json)
retains all four job IDs, timings, source and checkpoint identities,
and memory measurements. Mixed-batch per-rank memory and independent
posterior-reference qualification remain open.

### Mixed 16-case update and production cost

On 2026-11-21, both sampled layouts executed the same 334 of 659
conditionals from the original mixed batch. The critical update times
were 310.52 seconds on 16 A100s and 189.31 seconds on 32 A100s. These
correspond to 1.380 and 1.683 GPU-hours per update before fixed launch
and checkpoint overhead. The maximum per-rank PyTorch reservations
were 38.43 and 38.25 GiB, respectively. The loss and global gradient
norm agreed at float64 reporting precision.

Use the lower observed rate, 1.380 GPU-hours per update, as a bounded
rank-4 scenario in the cost forecast. Two rank-4 replicas alone would
cost 3,450 GPU-hours at 1,000 updates each with the required 25% margin.
The three planned common allowances do not fit the current training
envelope at the measured rate. Production cost for rank-16 and full
tuning on comparable geometry remains to be measured. [Mixed-batch
receipt and cost forecast](../evidence/hierarchical-ft-v1-conditional-sampling/mixed-batch-capacity-and-production-forecast.json)

## Standing assumptions

| ID | Status | Scope and assumption | Source |
|---|---|---|---|
| A02 | Adopted | Use the pinned TabPFN-v2 regression checkpoint and Hugging Face PEFT for LoRA. Budgeted pilot arms use the existing rank-4 targets and eight-member probability ensemble. Variants and numerical policies are recorded in request identity. | [Study contract](hierarchical-ft-v1.md) |
| A03 | Adopted for bounded CUDA pilots | Use FP32 neural execution, FP64 probability and decoding arithmetic, and disabled fusion. Keep tiled math attention and layer and member recomputation. Use `canonical_half_normal_full_support.v1` for consistent density scoring and posterior generation. | [CUDA qualification](cuda-qualification-v1.md) and [study contract](hierarchical-ft-v1.md) |
| A04 | Adopted for bounded distributed benchmarks | Use synchronous NCCL gradient summation and compute-aware case assignment with selective conditional splitting. Match initialization and case streams between comparison arms within each seed. Retain the actual GPU-hours, timing, memory, and checkpoint receipts. | [Study epic](https://github.com/BrieucB/FineTuneForge/issues/9) and A01 source evidence |
| A05 | Adopted | Use fresh context banks and independent simulated query cases on training visits. The training context ladder is 16, 64, 256, and 1,024. Reference and posterior tests keep the sealed test split and the protocol's declared evaluation settings. | [Study contract](hierarchical-ft-v1.md) |
| A06 | Adopted | Every posterior request uses exactly 50,000 complete vectors. Record invalid decoding as a failed request. Held-out density scoring at known generating parameters makes zero posterior requests. | [Study contract](hierarchical-ft-v1.md) |
| A07 | Adopted | Reconcile allocated Slurm time, including failed jobs and idle allocated GPUs, in the study ledger. Prefer concurrent independent jobs, with GPU counts chosen from measured scaling and live placement. The EuroHPC balance and the study budget are separate accounting scopes. | [Budget](budget.md) |
| A08 | Open | The one-wide-case restart fixtures and per-rank memory check on the original mixed batch passed. Qualify larger-context quality and posterior/reference accuracy under I07. Use representative timings to forecast the full validation and campaign matrix before fixing the production allowance. | [Qualification issue I07](https://github.com/BrieucB/FineTuneForge/issues/16) |
