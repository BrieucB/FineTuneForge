# CUDA qualification environment

Built with PriorLabs-TabPFN.

The study GPU environment is separate from both the historical inference environment
and the CPU qualification environments. It is a qualification candidate until its
actual CUDA execution, driver compatibility and numerical checks pass on a compute
node. Preparing or importing this environment does not qualify the scientific model.

`environments/study-training-gpu.in` pins every non-Torch package to the matching
CPU lock and records that lock's SHA256. The generated GPU lock adds Torch 2.6.0's
CUDA 12.4 runtime libraries and Triton. The installed Torch build identifies itself
as `2.6.0+cu124`; TabPFN 2.0.9, PEFT 0.21.3, Transformers 4.57.6, NumPy 1.26.4
and SciPy 1.12.0 remain fixed. Package CUDA libraries are provided by the wheel
dependencies; a working compatible NVIDIA driver is supplied by the cluster and
must be verified during compute-node qualification.

Prepare on a network-enabled login/preparation host using Python 3.11 and the
pinned bootstrap uv executable. For example:

```bash
export PF_SCRATCH=/path/to/private/posteriorforge
bash scripts/prepare-study-gpu-env.sh \
  --environment "$PF_SCRATCH/envs/study-training-gpu" \
  --uv "$PF_SCRATCH/envs/bootstrap/bin/uv" \
  --cache "$PF_SCRATCH/cache/uv" \
  --checkpoint "$PF_SCRATCH/cache/model/tabpfn-v2-regressor.ckpt" \
  --receipt "$PF_SCRATCH/results/gpu-environment-preparation.json"
```

The script refuses to modify an existing environment without its own GPU marker,
verifies all package hashes during installation, checks dependency consistency,
imports the model libraries with CUDA hidden, and authenticates the already cached
checkpoint. It does not fetch model weights or initialize a GPU. The receipt retains
the full installed package inventory and source/lock hashes.

After populating the dependency cache, `--offline` prepares another separately
owned environment without package-index access. Compute jobs use the prepared
interpreter and cached weights directly; set `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1` and do not run dependency installation inside the job.
The preparation-time `CUDA_VISIBLE_DEVICES=''` mask applies only to this script;
the qualification launcher must use Slurm's actual device visibility.

The lock was generated with uv 0.13.7:

```bash
/path/to/bootstrap/bin/uv pip compile environments/study-training-gpu.in \
  --python-version 3.11 --generate-hashes \
  --output-file environments/study-training-gpu.lock
```

No CPU/historical lock, environment, sampler default or scientific gate changes as
a consequence of preparing this candidate.
