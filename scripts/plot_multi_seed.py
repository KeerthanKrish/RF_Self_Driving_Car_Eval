"""Aggregate several single-seed runs into one variance-across-seeds plot.

This is the "second command" in Phase 0's done-when bar
(docs/03_roadmap.md): "a single command launches a run, and a second
command produces a multi-seed evaluation plot with variance bands from the
artifacts it wrote." The first command is any existing script run several
times with different --seed values; this is the second command.

Usage:
    python scripts/plot_multi_seed.py "runs/*_random_metadrive_baseline-seed*" \\
        --tag random-baseline

The glob pattern must be quoted so the shell doesn't expand it -- this
script needs to see the literal pattern to match against run directory names.
"""

import argparse
import datetime
from pathlib import Path

from rlsdc.artifacts import RUNS_DIR
from rlsdc.plotting import plot_seed_variance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pattern", help="glob pattern (relative to runs/), quoted")
    parser.add_argument("--tag", default="multiseed", help="label for the output directory/plot title")
    args = parser.parse_args()

    run_dirs = sorted(RUNS_DIR.glob(args.pattern.removeprefix("runs/")))
    if not run_dirs:
        raise SystemExit(f"no run directories matched pattern: {args.pattern!r} under {RUNS_DIR}")

    print(f"found {len(run_dirs)} matching run(s):")
    for d in run_dirs:
        print(f"  {d}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = RUNS_DIR / f"{timestamp}_plots_{args.tag}"
    png_path = plot_seed_variance(run_dirs, out_dir, title=args.tag)

    print()
    print(f"wrote {png_path}")
    print(f"wrote {out_dir / 'summary.csv'}")
    print(f"wrote {out_dir / 'sources.txt'}")


if __name__ == "__main__":
    main()
