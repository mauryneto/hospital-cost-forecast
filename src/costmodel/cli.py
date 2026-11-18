"""Command line: ``uv run costmodel <command> --config configs/<dataset>.toml``."""

from __future__ import annotations

import argparse
import sys

DEFAULT_CONFIG = "configs/sihsus_rj.toml"
RUN_STEPS = ("data", "train", "evaluate", "monitor")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="costmodel")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in [
        ("data", "ingest the configured source, validate it against the contract, save the admissions table"),
        ("eda", "exploratory analysis written to reports/<dataset>/"),
        ("baselines", "fit the baselines and measure them on the validation window"),
        ("train", "search the candidate's settings on validation, refit, register it in MLflow as 'staging'"),
        ("evaluate", "score 'staging' once on the test window, run the gate, promote it to 'production' if it passes"),
        ("monitor", "score the monitoring window month by month: drift (PSI), error, and the retrain rule"),
        ("replay", "what-if: replay the monitoring window under each retraining policy (slow: it retrains)"),
        ("families", "compare other model families (random forest, CatBoost) with the candidate on validation"),
        ("blend", "average the candidate's best per-admission and best month's-total settings, on validation"),
        ("segments", "one model for every admission, or one per admission type / procedure group? on validation"),
        ("recent-price", "does the model do better when given each procedure's recent billed price? on validation"),
        ("base-window", "how much billed history should each retrain use? retrained every 3 months, on validation"),
        ("foundation", "a tabular foundation model (TabPFN) against the candidate, on a sample of validation"),
        ("skill", "is the production model worth it over a plain average? totals by hospital, specialty..., on test"),
        ("explore", "the production model group by group (hospital, specialty...), month by month: data for the explorer"),
        ("site", "one static page with the results (docs/index.html), built from the reports"),
        ("run", "the whole pipeline, in order: data, train, evaluate, monitor"),
    ]:
        cmd = sub.add_parser(name, help=help_text)
        cmd.add_argument("--config", default=DEFAULT_CONFIG)
    check = sub.add_parser("validate", help="check a CSV/Parquet file against the data contract (DATA_CONTRACT.md)")
    check.add_argument("path")
    check.add_argument("--max-paid-amount", type=float, default=None)
    score = sub.add_parser("score", help="batch: price every admission of a month (or a file) with the production model")
    score.add_argument("--config", default=DEFAULT_CONFIG)
    score.add_argument("--month", help="YYYY-MM; without --input, taken from the dataset's admissions table")
    score.add_argument("--input", help="CSV/Parquet with the contract's admission-time columns (the cost may be missing)")
    serve = sub.add_parser("serve", help="HTTP API with the production model: POST /predict, GET /health, docs at /docs")
    serve.add_argument("--config", default=DEFAULT_CONFIG)
    serve.add_argument("--host", default="127.0.0.1", help="127.0.0.1 = this computer only")
    serve.add_argument("--port", type=int, default=8000)
    syn = sub.add_parser("synthetic", help="write made-up admissions in the canonical format (to try the pipeline)")
    syn.add_argument("--out", default="data/synthetic_admissions.csv", help="relative to the work folder")
    syn.add_argument("--n", type=int, default=200_000)
    syn.add_argument("--start", default="2021-01-01")
    syn.add_argument("--end", default="2024-12-31")
    syn.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    if args.command == "synthetic":
        from costmodel.paths import HOME
        from costmodel.synthetic import write

        write(HOME / args.out, args.n, args.start, args.end, args.seed)
        return

    if args.command == "validate":
        from costmodel import contract

        try:
            df = contract.validate(contract.read_table(args.path), args.max_paid_amount or contract.DEFAULT_MAX_PAID_AMOUNT)
        except contract.ContractError as exc:
            print(exc)
            sys.exit(1)
        print(f"OK: {len(df):,} admissions follow the contract.")
        return

    from costmodel.config import load_config

    cfg = load_config(args.config)
    if args.command == "data":
        from costmodel.sources import build_table

        build_table(cfg)
    elif args.command == "eda":
        from costmodel.eda import run

        run(cfg)
    elif args.command == "baselines":
        from costmodel.run_baselines import run

        run(cfg)
    elif args.command == "train":
        from costmodel.train import run

        run(cfg)
    elif args.command == "evaluate":
        from costmodel.holdout import run

        run(cfg)
    elif args.command == "monitor":
        from costmodel.monitor import run

        run(cfg)
    elif args.command == "replay":
        from costmodel.replay import run

        run(cfg)
    elif args.command == "families":
        from costmodel.families import run

        run(cfg)
    elif args.command == "blend":
        from costmodel.blend import run

        run(cfg)
    elif args.command == "segments":
        from costmodel.segments import run

        run(cfg)
    elif args.command == "recent-price":
        from costmodel.recent_price import run

        run(cfg)
    elif args.command == "base-window":
        from costmodel.base_window import run

        run(cfg)
    elif args.command == "foundation":
        from costmodel.foundation import run

        run(cfg)
    elif args.command == "skill":
        from costmodel.skill import run

        run(cfg)
    elif args.command == "explore":
        from costmodel.explore import run

        run(cfg)
    elif args.command == "site":
        from costmodel.site import run

        run(cfg)
    elif args.command == "score":
        from costmodel.contract import ContractError
        from costmodel.score import run_batch

        try:
            run_batch(cfg, args.month, args.input)
        except (ContractError, ValueError) as exc:
            print(exc)
            sys.exit(1)
    elif args.command == "serve":
        import uvicorn

        from costmodel.api import create_app

        uvicorn.run(create_app(cfg), host=args.host, port=args.port)
    elif args.command == "run":
        # Each step in its own process, as if typed by hand: memory is returned between steps, so the peak is that of
        # the largest step rather than of all of them together; and a failed step stops the run.
        import subprocess

        for step in RUN_STEPS:
            print(f"\n=== {step} ===", flush=True)
            code = subprocess.call([sys.executable, "-m", "costmodel.cli", step, "--config", str(cfg.path)])
            if code != 0:
                print(f"step '{step}' failed (exit code {code}); stopping")
                sys.exit(code)


if __name__ == "__main__":
    main()
