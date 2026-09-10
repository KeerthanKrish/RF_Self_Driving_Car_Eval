# Infrastructure and Workflow

**Status**: partially verified — see "Pending verification" at the bottom
**Last updated**: 2026-09-10

## The two machines

| | Mac (local) | Ubuntu workstation (remote) |
|---|---|---|
| Role | Interface only — reading, reviewing, editing | **Everything that executes**: training, evaluation, tests, installs |
| Address | — | `keerthan@100.71.12.16` (Tailscale) |
| Hardware | Apple Silicon | NVIDIA GPU |
| Shared with other projects? | No | **Yes — isolation is mandatory** |

The rule, from `working_instructions.md`: real implementation, testing, and execution happen on
the Ubuntu machine. The Mac is not where things run and not where artifacts accumulate. If a file
has to be created locally first because local tooling only reaches the local filesystem, it gets
synced to the Ubuntu machine, verified there, and the local copy is not left as a divergent second
source of truth.

Changes to the Mac itself (installing tools, generating keys, editing dotfiles or config) require
asking first. The Ubuntu machine allows more autonomous action.

## Network access

The workstation is reachable over Tailscale. `100.71.12.16` is in the Tailscale CGNAT range, so
**Tailscale must be running on the Mac** or the host is simply unreachable — `ssh` will time out
rather than give a useful error.

Check and start:

```bash
tailscale status        # "Tailscale is stopped." means this is the problem
tailscale up
```

This has already been the cause of one connection failure. Check it first whenever the workstation
appears to be down.

## Environment isolation on the shared workstation

Other projects run on this machine. Nothing about this project may touch a shared or base Python
environment.

**Rules:**
- One dedicated environment for this project. Never install into `base`, `system`, or another
  project's environment.
- Never `sudo pip install` or `pip install --user`.
- Every dependency is recorded in the repo, so the environment is reconstructible rather than
  being a hand-built artifact that exists only on that machine.
- GPU work should be considerate: other projects may be using the card. Check `nvidia-smi` before
  launching anything long, and prefer to leave headroom rather than claim all memory.

The exact environment manager is pending verification (see below). The intended convention is a
single environment named `rlsdc`, with dependencies pinned in the repo.

## An honest note on the GPU

For the early phases the GPU is close to useless, and this is expected rather than a
misconfiguration. Phase 1 works on ~25-dimensional observation vectors through small MLPs, where
per-kernel launch overhead dominates and **CPU is frequently faster than GPU**. Environment
stepping in highway-env is pure-Python and CPU-bound regardless.

The GPU starts mattering when: observations become images (CNN encoders), models get meaningfully
larger, or many environments run in parallel. Until then, defaulting to CPU is the right call and
is also the polite one on a shared machine.

Do not read "training is slow" in Phase 1 as "the GPU isn't being used."

## File sync

**Code, documentation, configs, and small results sync via git.**

Repository: <https://github.com/KeerthanKrish/RF_Self_Driving_Car_Eval>

The flow is directional:
- **Push from the Ubuntu machine** — that is where work is produced.
- **Pull on the Mac** — to read and review.

Push regularly rather than letting work accumulate uncommitted. Treat "committed and pushed" as
part of finishing a change, not as a separate later step.

**Large files do not go in git.** Videos, model checkpoints, raw logs, and datasets move by `scp`:

```bash
# Pull a run's videos from the workstation to the Mac for review
scp -r keerthan@100.71.12.16:~/RF_Self_Driving_Car_Eval/runs/<run_id>/videos ./
```

`.gitignore` enforces this — `runs/`, `*.mp4`, `*.pt`, `*.ckpt` and similar are excluded by
policy, not by accident. If something large needs to be shared, it goes over `scp` and gets
*referenced* in the docs by run ID, not committed.

## Run artifact organization

From `working_instructions.md`: generated output goes into clearly separated, collision-proof
locations from the start, so a later run cannot silently overwrite an earlier one.

```
runs/
  <YYYYMMDD-HHMMSS>_<algo>_<env>_<short-tag>/
    config.yaml          # exact resolved config used
    metrics.csv          # per-episode / per-step logged scalars
    checkpoints/
    videos/
    stdout.log
    git_sha.txt          # commit the run was launched from
```

The run directory name is generated, never reused, and never hand-edited. `git_sha.txt` exists so
a result can always be traced back to the code that produced it.

## Working agreements that affect execution

Carried from `working_instructions.md`, restated here because they shape how runs are handled:

- **Autonomy is granted per class of action.** A go-ahead to launch runs authorizes launching,
  health-checking, and moving on — not pausing again before each one.
- **Launch, verify healthy, then stop reporting.** After starting something long-running, confirm
  the process is alive and making real progress, then leave it alone until asked to check in.
  No proactive polling.
- **Verify claims against logged data**, not against a plausible-looking curve or a few frames of
  video. A metric asserting success is a claim, not a proof.
- **Prefer reversible changes.** Experimental changes should be revertible cleanly, preserving the
  attempt in history rather than deleting work in place.

## Pending verification

These are assumptions, not confirmed facts. They will be checked on first successful connection to
the workstation and this document updated in place.

- [ ] Which environment manager exists on the workstation (conda / mamba / uv / venv)
- [ ] Installed CUDA version and driver, to pick the matching PyTorch build
- [ ] Available disk space and a sensible location for `runs/`
- [ ] Whether the workstation already has working GitHub credentials for push
- [ ] What other projects are running, so their GPU/CPU usage can be respected
- [ ] Python version available and whether it satisfies Gymnasium + highway-env

## Known open issue: GitHub authentication

Neither auth path works from the Mac as of 2026-09-10:

- `gh` CLI: token in keyring is invalid (`gh auth refresh -h github.com` would fix it)
- SSH: `git@github.com` returns `Permission denied (publickey)` — `~/.ssh/id_ed25519` exists but
  is not registered on the GitHub account

This does not block work, because pushes are meant to originate from the Ubuntu machine anyway.
It does block *pulling on the Mac* for review, so it needs resolving. Fixing it requires a change
to the Mac, so it is left for explicit approval rather than done unilaterally.
