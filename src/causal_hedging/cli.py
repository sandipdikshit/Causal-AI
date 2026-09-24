"""Command-line entry point."""

import argparse
from pathlib import Path

from .data import fetch
from .experiment import run
from .visuals import render


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch", help="Refresh free daily WTI market context")
    demo = sub.add_parser("demo", help="Train, evaluate and render the full demo")
    demo.add_argument("--steps", type=int, default=4096)
    demo.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    demo.add_argument("--test-paths", type=int, default=8)
    sub.add_parser("render", help="Regenerate visuals from saved evaluation")
    args = parser.parse_args()
    if args.command == "fetch":
        print(fetch(Path("data")))
    elif args.command == "demo":
        print(run(Path("results"), args.seeds, args.steps, args.test_paths))
        render(Path("."))
    else:
        render(Path("."))


if __name__ == "__main__":
    main()
