# Rank-4 learning pilot readiness

The next candidate is LoRA rank 4, replica 0, learning rate `1e-4`, with the
declared 200-update allowance and a fresh pretrained initialization. It uses the
canonical 10,000 training and 20 validation setup definitions, matched data
streams, the full-support density law, eager per-layer parametrization caching
and disabled fusion. The first proposed allocation ends after update 1 and is
subject to separate authorization. Production campaigns remain gated by I07.

## Work and measurement plan

| Work | Training episodes | Training conditional stages | Quick validation |
| --- | ---: | ---: | --- |
| Update 1 | 16 | 659 | First check is at update 20 |
| Updates 2–20 | 304 | 15,411 | 320 scores, 15,264 forward conditionals |
| Updates 21–200 | 2,880 | 138,893 | Nine further checks of 320 scores each |
| Full allowance | 3,200 | 154,963 | Ten checks, 3,200 scores total |

Each training conditional uses eight ensemble members and backward with
activation checkpointing. Update 1 spans 10–82 unknowns, 3–372 readings,
1–6 experiment families, and 3–42 datasets. The 16 episodes use context sizes
16, 64, 256 and 1024 with counts 3, 6, 3 and 4, respectively. Their final
teacher-forced stages have up to 412 input features. The batch generates 5,312
worlds across its fresh context and query banks. It has up to 86,317 local
dataset-world pairs before legitimate forward reuse. Preserve actual unique
local forward counts by episode. The first validation check uses the full
fixed panel of 40 cases, two bank seeds and four contexts. Banks are reused
across contexts and milestones, and scoring repeats at each checkpoint.

Two measured cached rank-4 A100 updates took 183.54 and 176.57 seconds on
matched batches of 16 cases with four unknowns and 16 context worlds. The
matched stock updates took 284.10 and 277.46 seconds. Repeating the tiny
cached workload 200 times would take about 9.81–10.20 GPU-hours before
validation. Representative allocation hours remain undetermined. The retained
training-engineering allocations used 4,519 GPU-seconds, including failed
compiler preflights. The study ledger in that receipt recorded 21,499
cumulative allocated GPU-seconds and zero held reservations.
[Training-engineering evidence](../evidence/training-engineering-v1/report.md)

Use a single Karolina A100 worker and the actual first matched batch to measure
cold bank generation or authenticated cache reads, preprocessing and transfer,
training conditionals including recomputation, peak GPU memory, model load,
initialization, and checkpoint/export. Reconcile terminal allocated GPU-time
and authenticate the update-1 checkpoint. Use the timing and memory receipt
to decide whether the update-20 segment can be allocated. Check the larger
shapes covered by that segment and measure the complete first 320-score panel,
including validation cache behavior and checkpoint cost. For later segments,
apply measured rates by context size and feature width to the fixed workload
counts and include load, restore, idle time and failures. Leave unmeasured
shapes and components pending. The full study forecast also includes reference
fitting and refinement and all complete-vector requests. Neither this plan nor
the CPU bundle changes the study ledger.

## Scientific checks

Read the quick-validation score trajectory on the fixed panel. One replica at
one learning rate leaves learning-rate selection and transfer unqualified.
If a later authorized diagnostic examines noise, use a paired frozen and
adapted evaluation on the retained zero-sensor D4/C256 control. Use the same
case, bank, seed, density law and fixed reference. Request 50,000 complete
vectors per model. Record
noise CDF error, mean bias and SD ratio against the reference. Retain failures
when a decoded vector is invalid. The existing rank-4 zero-sensor control had
noise CDF error about 0.732 against a 0.10 gate. The rank-16 strict final-weight
comparison against its historical stock control failed 12 of 422 tensor checks
despite passing the bounded posterior-preservation comparison. Rank 16 needs
its own engineering resolution before use. Final-test data stay sealed until
the existing selection lock.
