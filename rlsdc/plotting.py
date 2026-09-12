"""Multi-seed aggregation and variance plotting.

RL results have enormous seed variance -- see docs/02_technical_design.md
Section 8. A single seed's result is noise, not a finding. This module
reads several runs' `metrics.csv` (each a different algorithm seed, same
environment/config otherwise) and reports the variance across seeds
explicitly, rather than a single number that hides how much luck was
involved.

Scope note: this is the Phase 0 version -- it compares *final* evaluation
results across seeds (each run's already-computed 50-episode held-out
summary). It deliberately does not yet plot a *learning curve over training
steps* with a shaded variance band the whole way through -- that needs
periodic evaluation checkpoints during training, which is real added
complexity and added compute cost. That's deferred to Phase 1, when there
are actually two algorithms to compare that way (see D-025).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

METRICS = [
    ("success_rate", "arrive_dest", "rate"),
    ("collision_rate", "crash", "rate"),
    ("off_road_rate", "out_of_road", "rate"),
    ("mean_return", "return", "mean"),
    ("mean_cost", "cost", "mean"),
    ("mean_route_completion", "route_completion", "rate_as_mean"),
]
# ("output name", "metrics.csv column", "how to aggregate")
# "rate"          -> fraction of rows where the column is truthy (1/0 flags)
# "mean"          -> plain mean of the column's values
# "rate_as_mean"  -> plain mean, but the column is already a 0..1 fraction
#                    (route_completion), not a 0/1 flag


@dataclass
class RunSummary:
    run_dir: Path
    seed_label: str  # taken from the run directory name, for the x-axis
    values: dict  # metric name -> scalar


def summarize_run(run_dir: Path) -> RunSummary:
    """Read one run's metrics.csv and compute the same summary metrics as
    rlsdc.evaluate.summarize() -- but from disk, for a run whose process has
    already exited, rather than from in-memory EpisodeResult objects."""
    metrics_path = run_dir / "metrics.csv"
    with open(metrics_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{metrics_path} has no rows")

    values = {}
    for name, column, mode in METRICS:
        col_values = [float(r[column]) for r in rows]
        if mode == "rate":
            values[name] = sum(1.0 for v in col_values if v) / len(col_values)
        else:  # "mean" or "rate_as_mean" -- both are a plain average
            values[name] = sum(col_values) / len(col_values)

    # Run directory names are <timestamp>_<algo>_<env>_<tag>, and the tag
    # carries the seed (e.g. "smoke-seed3") per the --seed CLI convention
    # added in scripts/baseline_random.py and scripts/sb3_smoke.py.
    seed_label = run_dir.name.split("_")[-1]
    return RunSummary(run_dir=run_dir, seed_label=seed_label, values=values)


def plot_seed_variance(run_dirs: List[Path], out_dir: Path, title: str = "") -> Path:
    """Aggregate N runs (different seeds, same config) and plot each metric's
    per-seed values plus mean +/- 1 std, side by side.

    Writes out_dir/seed_variance.png, out_dir/summary.csv (one row per run,
    every metric), and out_dir/sources.txt (which run directories went in --
    the plotting equivalent of git_sha.txt: a result should always be
    traceable to exactly which runs produced it).
    """
    import matplotlib

    matplotlib.use("Agg")  # headless -- no display on this workstation
    import matplotlib.pyplot as plt

    if len(run_dirs) < 2:
        raise ValueError(
            f"got {len(run_dirs)} run(s) -- variance across seeds needs at least 2, "
            f"ideally >=3 (docs/02_technical_design.md Section 8)"
        )

    summaries = [summarize_run(d) for d in run_dirs]

    out_dir.mkdir(parents=True, exist_ok=False)

    metric_names = [name for name, _, _ in METRICS]
    fig, axes = plt.subplots(1, len(metric_names), figsize=(3.2 * len(metric_names), 4))
    if title:
        fig.suptitle(title)

    for ax, metric_name in zip(axes, metric_names):
        per_seed = np.array([s.values[metric_name] for s in summaries])
        x = np.arange(len(summaries))

        ax.scatter(x, per_seed, color="tab:blue", zorder=3, label="per seed")
        mean = per_seed.mean()
        std = per_seed.std()
        ax.errorbar(
            [len(summaries) / 2 - 0.5], [mean], yerr=[std],
            fmt="D", color="black", capsize=6, zorder=4, label="mean +/- 1 std",
        )
        ax.set_xticks(x)
        ax.set_xticklabels([s.seed_label for s in summaries], rotation=45, ha="right")
        ax.set_title(f"{metric_name}\n{mean:.3f} +/- {std:.3f}", fontsize=9)
        ax.grid(alpha=0.3)

    axes[0].legend(fontsize=7, loc="best")
    fig.tight_layout()
    png_path = out_dir / "seed_variance.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    # Full per-seed numbers, not just the picture -- so a claim can be
    # checked against the actual data, not just eyeballed off a plot.
    summary_csv_path = out_dir / "summary.csv"
    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run_dir", "seed_label", *metric_names])
        writer.writeheader()
        for s in summaries:
            writer.writerow({"run_dir": str(s.run_dir), "seed_label": s.seed_label, **s.values})

    (out_dir / "sources.txt").write_text(
        "\n".join(str(d.resolve()) for d in run_dirs) + "\n"
    )

    return png_path
