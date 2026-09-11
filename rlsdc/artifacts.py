"""Run-artifact plumbing.

Every run gets its own directory, its exact resolved config, and the exact
git commit it launched from -- so any later result can always be traced back
to precisely the code and config that produced it, per docs/01_infrastructure_and_workflow.md
and the "Config and reproducibility" rule in docs/02_technical_design.md Section 8.

Layout:

    runs/
      <YYYYMMDD-HHMMSS>_<algo>_<env>_<tag>/
        config.yaml          # exact resolved config used
        metrics.csv          # per-episode / per-step logged scalars
        checkpoints/
        videos/
        stdout.log
        git_sha.txt          # commit the run was launched from (+ "-dirty")

The directory name is generated once, never reused, never hand-edited --
mkdir uses exist_ok=False deliberately, so two runs colliding (e.g. launched
in the same second) fails loudly instead of one silently overwriting the other.
"""

from __future__ import annotations

import csv
import datetime
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs"


def git_sha() -> str:
    """Current commit, with a "-dirty" suffix if there are uncommitted changes.

    Uses `git status --porcelain` rather than `git diff --quiet`: the latter
    only catches modifications to already-tracked files and misses new
    *untracked* files -- which is exactly the case for a brand new script
    that hasn't been committed yet. A dirty-check that misses untracked
    files would silently mislabel a run as reproducible-from-a-clean-commit
    when it actually depended on code that isn't in git history at all.
    (Caught this the first time it mattered: see D-023.)

    Never raises: if git isn't available or this isn't a repo, returns a
    string that says so, rather than crashing a run over a non-essential
    provenance detail.
    """
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
        )
        return f"{sha}-dirty" if status.strip() else sha
    except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
        return f"UNKNOWN (git_sha() failed: {exc})"


class _Tee:
    """Writes to multiple streams at once. Used to mirror stdout into a log file
    without giving up seeing output live in the terminal."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        for s in self._streams:
            s.write(data)
        return len(data)

    def flush(self) -> None:
        for s in self._streams:
            s.flush()


class RunDir:
    """One collision-proof directory per run.

    Usage:
        config = dict(algo="random", env="metadrive", seed=0, ...)
        with RunDir(algo="random", env="metadrive", tag="baseline", config=config) as run:
            run.log_metrics({"episode": 0, "return": 12.3, "cost": 0, "length": 88})
            ...
        print(run.path)  # still valid after the `with` block
    """

    def __init__(
        self,
        algo: str,
        env: str,
        tag: str,
        config: Mapping[str, Any],
        runs_dir: Optional[Path] = None,
    ):
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        name = f"{timestamp}_{algo}_{env}_{tag}"
        base = runs_dir if runs_dir is not None else RUNS_DIR
        self.path = base / name
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path / "checkpoints").mkdir()
        (self.path / "videos").mkdir()

        (self.path / "git_sha.txt").write_text(git_sha() + "\n")
        with open(self.path / "config.yaml", "w") as f:
            yaml.safe_dump(dict(config), f, sort_keys=False)

        self._metrics_path = self.path / "metrics.csv"
        self._metrics_file = None
        self._metrics_writer = None
        self._metrics_fieldnames: Optional[list] = None

        self._log_file = open(self.path / "stdout.log", "w")
        self._real_stdout = None

    def log_metrics(self, row: Mapping[str, Any]) -> None:
        """Append one row to metrics.csv. The header is fixed by the first row's
        keys -- every subsequent row must use exactly the same keys, so a
        schema drift mid-run raises immediately instead of silently producing
        a metrics.csv with ragged columns."""
        if self._metrics_writer is None:
            self._metrics_fieldnames = list(row.keys())
            self._metrics_file = open(self._metrics_path, "w", newline="")
            self._metrics_writer = csv.DictWriter(
                self._metrics_file, fieldnames=self._metrics_fieldnames
            )
            self._metrics_writer.writeheader()
        elif list(row.keys()) != self._metrics_fieldnames:
            raise ValueError(
                f"metrics schema changed mid-run: expected {self._metrics_fieldnames}, "
                f"got {list(row.keys())}"
            )
        self._metrics_writer.writerow(row)
        self._metrics_file.flush()

    def close(self) -> None:
        if self._metrics_file is not None:
            self._metrics_file.close()
        if self._real_stdout is not None:
            sys.stdout = self._real_stdout
            self._real_stdout = None
        if not self._log_file.closed:
            self._log_file.close()

    def __enter__(self) -> "RunDir":
        self._real_stdout = sys.stdout
        sys.stdout = _Tee(self._real_stdout, self._log_file)
        print(f"[run] {self.path}")
        print(f"[run] git_sha = {(self.path / 'git_sha.txt').read_text().strip()}")
        return self

    def __exit__(self, *exc) -> None:
        self.close()
