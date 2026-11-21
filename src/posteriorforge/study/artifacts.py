"""Authenticated complete-vector artifacts; sample completeness is not accuracy."""
from __future__ import annotations
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
from .banks import array_descriptor, load_bank, CONTEXT_SIZES, SCHEMA as BANK_SCHEMA
from .spec import ProblemSpec

SCHEMA = "finetuneforge.study.samples.v1"


def _write_json(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")


def _validate(spec, samples):
    x = np.asarray(samples)
    if x.dtype != np.float64 or x.shape != (50000, spec.dimension):
        raise ValueError("exactly 50,000 complete float64 physical vectors are required")
    if not spec.support_mask(x).all():
        raise ValueError("nonfinite or unsupported samples; no clipping or incomplete acceptance")
    return np.ascontiguousarray(x)


def _validate_provenance(provenance, kind):
    if not isinstance(provenance, dict):
        raise ValueError("explicit sample provenance is required")
    provenance = dict(provenance)
    if kind == "posterior":
        for name in ("model_sha256", "bank_manifest_sha256", "protocol_sha256"):
            value = provenance.get(name)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("posterior provenance needs explicit model/bank/protocol SHA256 identities")
    elif type(provenance.get("forward_evaluations")) is not int or provenance["forward_evaluations"] != 0:
        raise ValueError("prior controls require an explicit integer zero forward-call count")
    return provenance


def _bank_cost(manifest, spec, context_size):
    if type(context_size) is not int or context_size not in CONTEXT_SIZES:
        raise ValueError("declared context size is required")
    worlds = manifest.get("worlds")
    if (manifest.get("schema") != BANK_SCHEMA or manifest.get("problem_fingerprint") != spec.fingerprint
            or type(worlds) is not int or worlds < context_size):
        raise ValueError("bank problem or context mismatch")
    cost = manifest.get("cost")
    if not isinstance(cost, dict):
        raise ValueError("bank cost trace required")
    trace = cost.get("prefix_forward_evaluations")
    if (not isinstance(trace, list) or len(trace) != worlds + 1
            or any(type(x) is not int or x < 0 for x in trace)
            or trace[0] != 0 or any(a > b for a, b in zip(trace, trace[1:]))):
        raise ValueError("valid bank prefix cost trace required")
    recorded = cost.get("forward_evaluations")
    generation = cost.get("actual_generation_forward_evaluations", recorded)
    if (type(recorded) is not int or recorded != trace[-1] or type(generation) is not int
            or generation < trace[-1]):
        raise ValueError("valid bank generation cost required")
    return {"context_size": context_size, "context_forward_evaluations": trace[context_size],
            "shared_bank_generation_forward_evaluations": generation,
            "posterior_sampling_forward_evaluations": 0}


def save_samples(spec, samples, output, *, observation=None, provenance=None, kind="posterior",
                 bank_directory=None, context_size=None):
    if kind not in ("posterior", "prior_control"):
        raise ValueError("unknown sample artifact kind")
    x = _validate(spec, samples)
    provenance = _validate_provenance(provenance, kind)
    if kind == "posterior":
        y = np.asarray(observation, dtype=np.float64)
        if y.shape != (spec.observation_dimension,) or not np.isfinite(y).all():
            raise ValueError("one finite observation-only query is required")
        if bank_directory is None:
            raise ValueError("posterior requires a bank directory")
        bank = load_bank(bank_directory, expected_sha256=provenance["bank_manifest_sha256"])
        if bank.spec.fingerprint != spec.fingerprint:
            raise ValueError("bank problem mismatch")
        cost = _bank_cost(bank.manifest, spec, context_size)
        prefix = bank.prefix(context_size)
        if prefix.manifest["cost"].get("forward_evaluations") != cost["context_forward_evaluations"]:
            raise ValueError("bank prefix cost mismatch")
        bank_manifest = (Path(bank_directory) / "manifest.json").read_bytes()
        if sha256(bank_manifest).hexdigest() != provenance["bank_manifest_sha256"]:
            raise ValueError("bank manifest checksum mismatch")
    elif observation is not None:
        raise ValueError("prior controls have no conditioning observation and zero forward calls")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    _write_json(root / "spec.json", spec.to_dict())
    with (root / "samples.npz").open("xb") as f:
        np.savez(f, samples=x)
    files = ["spec.json", "samples.npz"]
    if kind == "posterior":
        with (root / "observation.npy").open("xb") as f:
            np.save(f, y, allow_pickle=False)
        (root / "bank_manifest.json").write_bytes(bank_manifest)
        files.extend(("observation.npy", "bank_manifest.json"))
    manifest = {"schema": SCHEMA, "kind": kind, "status": "samples_complete_unassessed",
                "scientifically_qualified": False, "requested_count": 50000,
                "problem_fingerprint": spec.fingerprint, "parameter_order": list(spec.parameter_order),
                "samples": array_descriptor(x), "provenance": provenance,
                "files": {name: sha256((root / name).read_bytes()).hexdigest() for name in files}}
    if kind == "posterior":
        manifest["bank_cost"] = cost
    _write_json(root / "manifest.json", manifest)
    return dict(manifest, manifest_sha256=sha256((root / "manifest.json").read_bytes()).hexdigest())


def load_samples(directory, *, expected_manifest_sha256=None):
    root = Path(directory)
    raw = (root / "manifest.json").read_bytes()
    if expected_manifest_sha256 is not None and sha256(raw).hexdigest() != expected_manifest_sha256:
        raise ValueError("sample manifest checksum mismatch")
    manifest = json.loads(raw)
    if manifest.get("schema") != SCHEMA or manifest.get("kind") not in ("posterior", "prior_control") or manifest.get("status") != "samples_complete_unassessed" or manifest.get("scientifically_qualified") is not False:
        raise ValueError("unsupported sample manifest")
    _validate_provenance(manifest.get("provenance"), manifest["kind"])
    expected = {"spec.json", "samples.npz"} | ({"observation.npy", "bank_manifest.json"} if manifest["kind"] == "posterior" else set())
    if set(manifest.get("files", {})) != expected:
        raise ValueError("unexpected sample files")
    for name, digest in manifest["files"].items():
        if sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError("sample payload checksum mismatch")
    spec = ProblemSpec.from_dict(json.loads((root / "spec.json").read_text()))
    with np.load(root / "samples.npz", allow_pickle=False) as f:
        if set(f.files) != {"samples"}:
            raise ValueError("unexpected sample arrays")
        x = _validate(spec, f["samples"])
    if manifest.get("requested_count") != 50000 or manifest.get("problem_fingerprint") != spec.fingerprint or manifest.get("parameter_order") != list(spec.parameter_order) or manifest.get("samples") != array_descriptor(x):
        raise ValueError("sample/order/problem binding mismatch")
    y = None
    if manifest["kind"] == "posterior":
        bank_raw = (root / "bank_manifest.json").read_bytes()
        if sha256(bank_raw).hexdigest() != manifest["provenance"]["bank_manifest_sha256"]:
            raise ValueError("retained bank manifest checksum mismatch")
        retained_cost = manifest.get("bank_cost")
        if (not isinstance(retained_cost, dict)
                or any(type(value) is not int or value < 0 for value in retained_cost.values())
                or retained_cost != _bank_cost(json.loads(bank_raw), spec, retained_cost.get("context_size"))):
            raise ValueError("retained bank cost mismatch")
        y = np.load(root / "observation.npy", allow_pickle=False)
        if y.shape != (spec.observation_dimension,) or not np.isfinite(y).all():
            raise ValueError("invalid retained query")
    elif manifest.get("provenance", {}).get("forward_evaluations") != 0:
        raise ValueError("invalid zero-forward-call prior-control record")
    return spec, x, y, manifest
