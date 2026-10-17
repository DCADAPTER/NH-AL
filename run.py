"""Command-line entry point; components and the AL loop live in pipeline.py."""
import argparse
import json
import sys
from pathlib import Path

from metamodels import list_metamodels
from pipeline import DEFAULT_CONFIG, ROOT, execute, prepare_runs, read_config


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run a configured adaptive-learning method")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--mode", choices=["reference", "nh", "paired"], default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--resume", action="store_true", help="Resume a repeated experiment")
    parser.add_argument("--list-metamodels", action="store_true")
    parser.add_argument("--check-config", action="store_true",
                        help="Validate requested modes without fitting or calling true g")
    parser.add_argument("--list-papers", action="store_true")
    parser.add_argument("--paper", default=None, help="Print a specification, e.g. P16")
    args = parser.parse_args(argv)

    if args.list_metamodels:
        print(json.dumps(list_metamodels(), indent=2))
        return
    if args.list_papers or args.paper:
        papers = read_config(ROOT / "papers.yaml")["profiles"]
        if args.list_papers:
            for row in papers:
                print(row["id"], row["name"], row["implementation"]["status"])
        else:
            found = next((row for row in papers if row["id"] == args.paper), None)
            if found is None:
                parser.error("Unknown paper id")
            print(json.dumps(found, ensure_ascii=False, indent=2))
        return

    cfg = read_config(args.config)
    mode = args.mode or cfg["mode"]
    from experiments import repeat, repetition_config
    if args.check_config:
        pipelines = prepare_runs(repetition_config(cfg, 0), mode)
        print(json.dumps({
            "valid": True,
            "runs": {name: {"model": type(pipeline.model).__name__,
                            "capabilities": sorted(pipeline.model.capabilities)}
                     for name, pipeline in pipelines.items()}
        }))
        return
    if 'repetitions' in cfg or args.resume:
        repeat(cfg, mode, Path(args.output or cfg['output']), resume=args.resume)
    else:
        execute(cfg, mode, Path(args.output or cfg["output"]))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()

