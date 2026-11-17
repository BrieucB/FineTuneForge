"""CPU-first commands; dry-run planning never imports an accelerator or submits."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .config import StudyConfig
from .spec import ProblemSpec


def configure_parser(commands):
    parser = commands.add_parser("study", help="FineTuneForge variable-size study (CPU-only software stage)")
    sub = parser.add_subparsers(dest="study_operation", required=True)
    plan = sub.add_parser("plan", help="expand the agreed design without executing work")
    plan.add_argument("--config")
    plan.add_argument("--output")
    inspect = sub.add_parser("inspect-spec", help="validate a problem definition and show its identity")
    inspect.add_argument("--spec", required=True)
    catalogue = sub.add_parser("catalogue", help="write setup definitions, not observations or inference results")
    catalogue.add_argument("--split", choices=("train", "validation", "test"), required=True)
    catalogue.add_argument("--count", type=int)
    catalogue.add_argument("--seed", type=int, default=20261117)
    catalogue.add_argument("--output", required=True)
    bank = sub.add_parser("prepare-bank", help="generate a paired synthetic training/validation bank on CPU")
    bank.add_argument("--spec", required=True)
    bank.add_argument("--count", type=int, choices=(16,64,256,1024,4096), required=True)
    bank.add_argument("--seed", type=int, required=True)
    bank.add_argument("--output", required=True)
    prior = sub.add_parser("prior-check", help="generate exactly 50,000 prior vectors with zero forward calls")
    prior.add_argument("--spec", required=True)
    prior.add_argument("--seed", type=int, required=True)
    prior.add_argument("--output", required=True)
    budget = sub.add_parser("budget", help="inspect the separate 10,000-hour study ledger; never submit")
    budget.add_argument("--ledger", required=True)
    sub.add_parser("submit", help="disabled during CPU-only implementation")
    return parser


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _spec(path):
    return ProblemSpec.from_dict(json.loads(Path(path).read_text()))


def run(args):
    args = dict(args)
    operation = args.pop("study_operation")
    if operation == "plan":
        result = (StudyConfig.load(args["config"]) if args.get("config") else StudyConfig()).plan()
        if args.get("output"):
            _write_json(args["output"], result)
        return result
    if operation == "inspect-spec":
        spec = _spec(args["spec"])
        return {"problem_fingerprint": spec.fingerprint, "id": spec.id, "split": spec.split,
                "dimension": spec.dimension, "observation_dimension": spec.observation_dimension,
                "parameter_order": list(spec.parameter_order), "roles": list(spec.roles)}
    if operation == "catalogue":
        from .catalogue import generate_catalogue
        specs = generate_catalogue(split=args["split"], count=args["count"], seed=args["seed"])
        result = {"schema": "posteriorforge.study.catalogue.v1", "split": args["split"],
                  "seed": args["seed"], "setups": [s.to_dict() for s in specs],
                  "problem_fingerprints": [s.fingerprint for s in specs],
                  "generated_observation_cases": 0, "status": "definitions_only_unqualified"}
        _write_json(args["output"], result)
        return {"setups": len(specs), "split": args["split"], "output": args["output"],
                "generated_observation_cases": 0, "gpu_jobs_submitted": 0}
    if operation in ("prepare-bank", "prior-check"):
        spec = _spec(args["spec"])
        if spec.split == "test":
            raise RuntimeError("final-test data remains sealed until model/backend selection is locked")
        if operation == "prepare-bank":
            from .banks import prepare_bank, save_bank
            return save_bank(prepare_bank(spec, args["count"], args["seed"]), args["output"])
        from .science import draw_prior
        from .artifacts import save_samples
        samples = draw_prior(spec, 50000, args["seed"])
        return save_samples(spec, samples, args["output"], kind="prior_control",
                            provenance={"forward_evaluations": 0, "seed": args["seed"]})
    if operation == "budget":
        from .budget import StudyBudgetLedger
        return StudyBudgetLedger(args["ledger"]).summary()
    if operation == "submit":
        raise RuntimeError("GPU/Slurm submission is disabled: only code and CPU tests are authorized")
    raise ValueError("unsupported study operation")
