"""CPU-first commands; dry-run planning never imports an accelerator or submits."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .config import StudyConfig
from .spec import ProblemSpec


def configure_parser(commands):
    parser = commands.add_parser("study", help="FineTuneForge variable-size study")
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
    train_plan = sub.add_parser("train-plan", help="show matched pilot/main plans or validate a training request")
    train_plan.add_argument("--config")
    train = sub.add_parser("train", help="execute a bounded, explicit CPU training diagnostic")
    train.add_argument("--config", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--cpu-diagnostic", action="store_true")
    train.add_argument("--resume-from")
    train.add_argument("--stop-after-update", type=int)
    pilot = sub.add_parser("train-cuda-pilot", help="run a ledger-bound stock CUDA learning pilot")
    pilot.add_argument("--config", required=True)
    pilot.add_argument("--output", required=True)
    pilot.add_argument("--ledger", required=True)
    pilot.add_argument("--resume-from")
    pilot.add_argument("--stop-after-update", type=int)
    posterior_plan = sub.add_parser("posterior-plan", help="validate a fixed50k posterior request without loading a model")
    posterior_plan.add_argument("--config", required=True)
    posterior = sub.add_parser("posterior", help="sample exactly50k complete vectors on CPU with recovery evidence")
    posterior.add_argument("--config", required=True)
    posterior.add_argument("--output", required=True)
    posterior.add_argument("--cpu-diagnostic", action="store_true")
    posterior.add_argument("--resume", action="store_true")
    eval_plan = sub.add_parser("evaluation-plan", help="build a bound validation matrix or frozen test matrix")
    eval_plan.add_argument("--config", required=True)
    eval_plan.add_argument("--output", required=True)
    evaluate = sub.add_parser("evaluate", help="consume a matrix using qualified references and CPU posterior diagnostics")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--cpu-diagnostic", action="store_true")
    evaluate.add_argument("--resume", action="store_true")
    seal = sub.add_parser("seal-selection", help="freeze validation-selected checkpoint identities before test access")
    seal.add_argument("--records", required=True)
    seal.add_argument("--manifest-sha256", required=True)
    seal.add_argument("--plan-sha256", required=True)
    seal.add_argument("--test-catalogue-sha256", required=True)
    seal.add_argument("--output", required=True)
    inspect_model = sub.add_parser("inspect-model", help="inspect cached base or trained weights on CPU; never download")
    inspect_model.add_argument("--base-checkpoint", required=True)
    inspect_model.add_argument("--campaign-checkpoint")
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
    if operation in ("train-plan", "train", "train-cuda-pilot", "posterior-plan", "posterior",
                     "evaluation-plan", "evaluate"):
        from . import workflows as w
        config = args.get("config")
        path = Path(config).resolve() if config else None
        value = json.loads(path.read_text()) if path else None
        base = path.parent if path else Path.cwd()
        if operation == "train-plan":
            return w.plan_campaigns() if value is None else w.TrainingRequest.from_dict(value).plan()
        if operation == "train":
            return w.execute_training_request(value, output=args["output"], base_directory=base,
                allow_unqualified_cpu_diagnostic=args["cpu_diagnostic"],
                resume_from=args["resume_from"], stop_after_update=args["stop_after_update"])
        if operation == "train-cuda-pilot":
            return w.execute_cuda_pilot_request(value, output=args["output"], ledger=args["ledger"],
                base_directory=base, resume_from=args["resume_from"],
                stop_after_update=args["stop_after_update"])
        if operation == "posterior-plan":
            return w.PosteriorRequest.from_dict(value).plan()
        if operation == "posterior":
            return w.execute_posterior_request(value, output=args["output"], base_directory=base,
                allow_unqualified_cpu_diagnostic=args["cpu_diagnostic"], resume=args["resume"])
        if operation == "evaluation-plan":
            result = w.plan_evaluation_request(value, base_directory=base)
            _write_json(args["output"], result)
            return {"status": "planned_unqualified", "plan_sha256": result["plan_sha256"],
                    "requests": len(result["requests"]), "output": args["output"], "gpu_jobs_submitted": 0}
        return w.execute_evaluation_request(value, output=args["output"], base_directory=base,
            allow_unqualified_cpu_diagnostic=args["cpu_diagnostic"], resume=args["resume"])
    if operation == "seal-selection":
        from .evaluation_campaign import load_evaluation_records, seal_selection
        plan, records = load_evaluation_records(args["records"],
            expected_manifest_sha256=args["manifest_sha256"], expected_plan_sha256=args["plan_sha256"])
        lock = seal_selection(args["output"], validation_plan=plan, records=records,
                              test_catalogue_sha256=args["test_catalogue_sha256"])
        return {"status": "selection_inputs_frozen", "selection_lock_sha256": lock.sha256,
                "output": args["output"], "gpu_jobs_submitted": 0, "scientifically_qualified": False}
    if operation == "inspect-model":
        from .workflows import inspect_model
        return inspect_model(args["base_checkpoint"], campaign_checkpoint=args["campaign_checkpoint"])
    if operation == "budget":
        from .budget import StudyBudgetLedger
        return StudyBudgetLedger(args["ledger"]).summary()
    if operation == "submit":
        raise RuntimeError("GPU/Slurm submission is disabled: only code and CPU tests are authorized")
    raise ValueError("unsupported study operation")
