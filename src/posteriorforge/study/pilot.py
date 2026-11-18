"""Authorization for the separately accounted stock CUDA learning pilot."""
from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys

from .budget import StudyBudgetLedger


CUDA_PILOT_EXECUTION = {
    "backend": "stock_tabpfn_cuda_qualification",
    "parametrization_cache_policy": "per_layer_forward_and_checkpoint_recompute.v1",
    "fusion_enabled": False,
}
CUDA_TILED_ATTENTION_EXECUTION = {
    **CUDA_PILOT_EXECUTION,
    "attention_policy": "tabpfn_fp32_math_sdpa_training_tiled_recompute.v1",
    "attention_max_score_bytes": 512 * 1024 * 1024,
}


def _git(directory, *args):
    try:
        return subprocess.run(
            ("git", "--no-optional-locks", "-C", str(directory), *args),
            capture_output=True,
            check=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError("CUDA pilot requires a readable Git source checkout") from error


def _source_identity():
    module_path = Path(__file__).resolve()
    checkout = Path(os.fsdecode(_git(module_path.parent, "rev-parse", "--show-toplevel").rstrip(b"\n"))).resolve()
    source = Path("src/posteriorforge")
    if module_path != checkout / source / "study/pilot.py":
        raise RuntimeError("CUDA pilot must import PosteriorForge from its source checkout")
    status = _git(checkout, "status", "--porcelain=v1", "--untracked-files=all", "--ignore-submodules=none")
    if status:
        raise RuntimeError("CUDA pilot requires a clean Git source checkout")

    tracked = {Path(os.fsdecode(path)) for path in _git(
        checkout, "ls-files", "--cached", "-z", "--", str(source)
    ).split(b"\0") if path}
    for name, module in tuple(sys.modules.items()):
        if name != "posteriorforge" and not name.startswith("posteriorforge."):
            continue
        filename = getattr(module, "__file__", None)
        if filename is None:
            continue
        path = Path(filename).resolve()
        if not path.is_relative_to(checkout) or path.relative_to(checkout) not in tracked:
            raise RuntimeError(f"CUDA pilot imported {name} outside the tracked source checkout")

    # Without exclude flags Git lists ignored files too. Bytecode in the ordinary
    # Python cache is permitted, while ignored source overlays must be rejected.
    for entry in _git(checkout, "ls-files", "--others", "-z", "--", str(source)).split(b"\0"):
        if entry:
            path = Path(os.fsdecode(entry))
            if path.parent.name != "__pycache__" or path.suffix != ".pyc":
                raise RuntimeError(f"CUDA pilot found an untracked file within the source package: {path}")
    commit = _git(checkout, "rev-parse", "--verify", "HEAD^{commit}").decode("ascii").strip()
    return checkout, commit


def authorize_cuda_pilot(*, request_sha256, ledger, output, resume_from=None, stop_after_update=None):
    """Check allocation and request identities using only CPU-side metadata."""
    job_id = os.getenv("SLURM_JOB_ID", "")
    if (re.fullmatch(r"[0-9]+", job_id) is None
            or os.getenv("SLURM_ARRAY_JOB_ID")
            or os.getenv("SLURM_ARRAY_TASK_ID")
            or os.getenv("SLURM_HET_SIZE")):
        raise RuntimeError("CUDA pilot requires a single decimal Slurm job ID")
    account = os.getenv("SLURM_JOB_ACCOUNT")
    partition = os.getenv("SLURM_JOB_PARTITION")
    if (account, partition) != ("eu-26-17", "qgpu"):
        raise RuntimeError("CUDA pilot requires Slurm account eu-26-17 and partition qgpu")
    device = os.getenv("CUDA_VISIBLE_DEVICES", "")
    if re.fullmatch(r"(?:[0-9]+|(?:GPU|MIG)-[0-9a-fA-F]+(?:-[0-9a-fA-F]+)*)", device) is None:
        raise RuntimeError("CUDA pilot requires exactly one device in CUDA_VISIBLE_DEVICES")
    cublas = os.getenv("CUBLAS_WORKSPACE_CONFIG")
    if cublas not in (":4096:8", ":16:8"):
        raise RuntimeError("CUDA pilot requires a deterministic CUBLAS_WORKSPACE_CONFIG")
    if not isinstance(request_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", request_sha256) is None:
        raise ValueError("CUDA pilot requires an explicit training request SHA256")
    if stop_after_update is not None and (type(stop_after_update) is not int or stop_after_update <= 0):
        raise ValueError("CUDA pilot stop_after_update must be a positive integer")

    ledger = Path(ledger).resolve()
    if not ledger.is_file():
        raise RuntimeError("CUDA pilot requires an existing shared study ledger")
    checkout, commit = _source_identity()
    binding = {
        "training_request_sha256": request_sha256,
        "source_commit": commit,
        "output": str(Path(output).resolve()),
        "resume_from": None if resume_from is None else str(Path(resume_from).resolve()),
        "stop_after_update": stop_after_update,
    }
    reservations = StudyBudgetLedger(ledger).inspect()["reservations"]
    entry = next((row for row in reservations if row["job_id"] == job_id), None)
    if entry is None or entry["state"] != "submitted" or entry["gpus"] != 1:
        raise RuntimeError("CUDA pilot requires a submitted one-GPU reservation for the current Slurm job")
    metadata = entry.get("metadata")
    if not isinstance(metadata, dict) or any(
        key not in metadata or type(metadata[key]) is not type(value) or metadata[key] != value
        for key, value in binding.items()
    ):
        raise RuntimeError("CUDA pilot does not match its ledger reservation metadata")
    return {
        "schema": "posteriorforge.study.cuda_pilot_authorization.v1",
        "job_id": job_id,
        "reservation_id": entry["reservation_id"],
        "account": account,
        "partition": partition,
        "gpus": entry["gpus"],
        "cuda_visible_devices": device,
        "cublas_workspace_config": cublas,
        "ledger": str(ledger),
        "source_checkout": str(checkout),
        **binding,
    }
