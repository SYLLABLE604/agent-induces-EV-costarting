#!/usr/bin/env python3
"""Build tasks from a user's local source files; no source data are distributed."""
import argparse
import json
from pathlib import Path

from dingle_replay.tasks import build_dataset, validate_dataset, write_inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["inventory", "build", "validate"])
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.example.json"))
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "validate":
        print(json.dumps(validate_dataset(args.output.resolve()), indent=2))
    else:
        function = build_dataset if args.command == "build" else write_inventory
        print(function(args.config.resolve(), args.repository_root.resolve(), args.output.resolve()))


if __name__ == "__main__":
    main()
