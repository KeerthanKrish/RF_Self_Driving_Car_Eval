# Infrastructure and Workflow

**Status**: verified against the workstation on 2026-09-10
**Last updated**: 2026-09-10

## The two machines

| | Mac (local) | Ubuntu workstation (remote) |
|---|---|---|
| Role | Interface only — reading, reviewing, editing | **Everything that executes**: training, evaluation, tests, installs |
| Hostname | `keerthans-macbook-pro` | `dtgpu` |
| Address | — | `keerthan@100.71.12.16` (Tailscale) |
| Hardware | Apple Silicon | See below |
| Shared with other projects? | No | **Yes — isolation is mandatory** |

### Workstation specification (verified 2026-09-10)

| | |
|---|---|
| OS | Ubuntu 24.04.4 LTS, kernel 7.0.0-30-generic |
| GPU | NVIDIA GeForce RTX 5060 Ti, 16 GB VRAM, driver 580.173.02 |
| **Compute capability** | **12.0 (Blackwell, `sm_120`)** — see the PyTorch warning below |
| CPU | 8 cores |
| RAM | 31 GB total |
| Disk | 915 GB, ~736 GB free |
| Python (system) | 3.12.3 |
| Environment manager | **miniforge3**, conda 26.5.3 |
| Existing conda envs | `base`, `env_isaaclab`, `lerobot` |

Other projects present in `~`: `SO-101-WM`, `UROP_Proj`, `trader`, `csci4551_final_project`,
`newcsciproj`, plus Isaac Sim / Omniverse and a Minecraft server. At survey time an IsaacLab
process was actively holding ~5.3 GB of VRAM at 20% GPU utilization. **The GPU is genuinely in
use by other work** — this is not a hypothetical concern.

### PyTorch must be ≥2.7 with CUDA 12.8+ wheels

The RTX 5060 Ti is Blackwell, compute capability `sm_120`. PyTorch builds against CUDA 12.1 or
12.4 do **not** include `sm_120` kernels. They install cleanly and then fail at runtime with
`no kernel image is available for execution on the device`, which reads like a broken driver
rather than a wheel mismatch.

Install from the CUDA 12.8 (or newer) index and verify before trusting anything:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); \
           print(torch.cuda.get_arch_list()); \
           print(torch.zeros(8, device='cuda').sum())"
```

`get_arch_list()` must contain `sm_120`, and the final line must execute without error. A
successful `torch.cuda.is_available()` alone is **not** sufficient — it returns `True` even when
no compatible kernels are present.

### Practical limits worth knowing

- **8 CPU cores** caps vectorized environments at roughly 6 parallel workers before contention.
  Relevant from Phase 2 onward, when PPO wants throughput.
- **The GPU is shared.** Check `nvidia-smi` before launching anything long, and leave headroom
  rather than claiming all 16 GB.

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

**The convention**: one conda environment named `rlsdc`, created under miniforge3. Verified that
no environment by that name exists, so there is no collision with `env_isaaclab` or `lerobot`.

```bash
source ~/miniforge3/etc/profile.d/conda.sh   # conda is not on PATH for non-interactive ssh
conda activate rlsdc
```

That first line matters: a non-interactive `ssh host 'command'` does not source `.bashrc`, so
`conda` will appear missing unless the profile script is sourced explicitly. Any script or
automation that runs over ssh needs it.

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

## Verification status

Resolved on 2026-09-10:

- [x] Environment manager — miniforge3 / conda 26.5.3
- [x] GPU, driver, and compute capability — RTX 5060 Ti, 580.173.02, `sm_120`
- [x] Disk space — ~736 GB free, ample for `runs/`
- [x] Other projects using the machine — Isaac Sim/IsaacLab actively on the GPU, plus several
      others in `~`
- [x] Python available — 3.12.3 system; the `rlsdc` env will pin its own
- [ ] Whether the workstation can push to GitHub — **not yet tested**, deliberately. Verifying
      this means attempting a push, which is a real write; it will be done as the first action
      of the next work session rather than inspecting stored credentials.

## Known open issue: GitHub authentication from the Mac

Neither auth path works from the Mac as of 2026-09-10:

- `gh` CLI: token in keyring is invalid (`gh auth refresh -h github.com` would fix it)
- SSH: `git@github.com` returns `Permission denied (publickey)`

This does not block the main workflow, since pushes originate from the workstation. It does block
*pulling on the Mac* for review, which is half the sync loop, so it needs resolving. Fixing it
means changing the Mac, so it waits for explicit approval rather than being done unilaterally.

The repository at `RF_Self_Driving_Car_Eval` was empty as of this writing, so the first push
establishes `main` and nothing can be clobbered by it.
