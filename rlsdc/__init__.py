"""Harness code shared across scripts: run artifacts, seeding, evaluation, plotting.

This is project-internal plumbing, not something published or reused elsewhere.
Installed editable (`pip install -e .`) into the `rlsdc` conda env by
scripts/setup_env.sh, so any script can just `import rlsdc` regardless of cwd.
"""
