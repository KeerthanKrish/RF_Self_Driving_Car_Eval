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

### The isolation guarantee, and how to check it

Other projects on this machine have their own, different torch builds. As of 2026-09-10:

| Env | torch | Python |
|---|---|---|
| `base` | none | — |
| `env_isaaclab` | 2.7.0+cu128 | 3.11 |
| `lerobot` | 2.11.0+cu130 | 3.12 |
| **`rlsdc`** (this project) | **2.11.0+cu128** | **3.11** |

Each has a physically separate `site-packages`, so these coexist without interacting. **Nothing
this project installs may change any row but its own.**

`scripts/setup_env.sh` enforces this rather than relying on care: before any `pip install` it
asserts that `CONDA_DEFAULT_ENV` is `rlsdc` *and* that `which python` resolves to
`~/miniforge3/envs/rlsdc/bin/python`, aborting otherwise. It also sets `PIP_USER=0` so a stray
`--user` cannot escape into `~/.local`.

To audit isolation at any time — read-only, touches nothing:

```bash
for e in base env_isaaclab lerobot rlsdc; do
  p=~/miniforge3/envs/$e/bin/python; [ "$e" = base ] && p=~/miniforge3/bin/python
  echo "$e: $("$p" -c 'import torch;print(torch.__version__, torch.__file__)' 2>/dev/null || echo 'no torch')"
done
ls ~/.local/lib/python*/site-packages 2>/dev/null | grep -i '^torch' || echo "~/.local clean"
```

The useful property: because the other environments hold *different* versions than this one, any
leak would be immediately visible as a version match rather than having to be inferred.

**The convention**: one conda environment named `rlsdc`, created under miniforge3. Verified that
no environment by that name exists, so there is no collision with `env_isaaclab` or `lerobot`.

```bash
source ~/miniforge3/etc/profile.d/conda.sh   # conda is not on PATH for non-interactive ssh
conda activate rlsdc
```

That first line matters: a non-interactive `ssh host 'command'` does not source `.bashrc`, so
`conda` will appear missing unless the profile script is sourced explicitly. Any script or
automation that runs over ssh needs it.

**Installed and verified 2026-09-10** (`bash scripts/verify_env.sh` → `ALL CHECKS PASSED`):

| Package | Version |
|---|---|
| python | 3.11.16 |
| torch | 2.11.0+cu128 (CUDA 12.8) |
| **metadrive-simulator** | **0.4.3** |
| panda3d | 1.10.13 |
| gymnasium | 1.3.0 |
| stable-baselines3 | 2.9.0 |
| numpy | 2.4.6 |
| ~~highway-env~~ | 1.12.1 — installed but **retired** (D-019) |

Also verified: `sm_120` present in `torch.cuda.get_arch_list()` and a real GPU matmul executes.

MetaDrive verified across all three rendering modes — see "Viewing the environment" below.

**Note**: installing MetaDrive replaced `pygame-ce` with `pygame`. highway-env's renderer may no
longer work in this environment. Irrelevant now that MetaDrive is the project simulator, but it
is why highway-env should not be assumed functional here.

## An honest note on the GPU

For the early phases the GPU is close to useless **for training**, and this is expected rather
than a misconfiguration. Phase 1 works on a 259-dimensional observation vector through small
MLPs, where per-kernel launch overhead dominates and **CPU is frequently faster than GPU**.
MetaDrive's physics runs in Bullet on the CPU, so environment stepping is CPU-bound regardless.

The GPU does get used for **3D rendering** (~1.6 GB measured), but that is for producing review
videos, not routine training. It starts mattering for training only when observations become
images (CNN encoders), models grow, or many environments run in parallel.

Do not read "training is slow" in Phase 1 as "the GPU isn't being used." And do not leave 3D
rendering enabled during training runs on a shared card.

## Viewing the environment

The project uses **MetaDrive** — Panda3D for 3D rendering, Bullet for physics (D-019). The
workstation is headless (`DISPLAY` unset), so the workflow is **render offscreen, write PNG/MP4
into the run directory, and `scp` to the Mac to look at**. `scripts/metadrive_smoke.py` is the
working reference for all three modes below.

### Three rendering modes, at very different costs

| Mode | How | GPU | Use for |
|---|---|---|---|
| **None** | `use_render=False` | none | **Training default.** Physics only. |
| **Top-down** | `env.render(mode="topdown", window=False)` | none — CPU raster | Debugging: schematic view of road, ego, traffic |
| **3D dashcam** | `image_observation=True` with an `RGBCamera` sensor, `vehicle_config={"image_source": "rgb_camera"}` | **~1.6 GB** | Forward-facing image-observation experiments (what a camera-based agent would actually see) |
| **3D chase** | `image_observation=True` with the built-in `main_camera` sensor, `vehicle_config={"image_source": "main_camera"}` | **~1.6 GB** | Review videos — third-person view from behind/above the car, for watching episodes as a human |

The chase camera is not something built for this project — it is MetaDrive's own `MainCamera`
class (`metadrive/engine/core/main_camera.py`), documented in its source as "a third-person
perspective camera for chasing the vehicle." Its default framing is 7.5 m behind and 2.2 m above
the car (`camera_dist`, `camera_height` in the global config), which is exactly a chase-cam
composition out of the box. It carries a HUD overlay (steering/throttle/brake/speed bars, a nav
arrow, a help hint) by default; pass `interface_panel: []` in the config to get clean footage
without it. Both `3d` and `chase` modes are exercised in `scripts/metadrive_smoke.py`.

Measured on `dtgpu` 2026-09-10, with IsaacLab already holding 5252 MiB:

- headless and top-down: GPU compute-app list **byte-identical** before and after — untouched
- 3D: total GPU use peaked at 6990 MiB of 16311, free never below 8856 MiB, and IsaacLab's
  process was unchanged at 5252 MiB throughout

Re-checked 2026-09-11 for the chase-camera mode specifically, with IsaacLab now *actively
computing* (65% utilization, a different PID than the earlier check) rather than idle: IsaacLab's
own GPU memory (3326 MiB) was byte-identical before and after the chase-camera run, and 12+ GB
was free before starting. Same rule either way: check `nvidia-smi` first, don't assume idle means
safe headroom forever.

**Rule for the shared card**: 3D rendering is for producing review videos, not for routine
training. Check `nvidia-smi` before enabling it. `image_on_cuda=True` (MetaDrive's option to keep
frames in GPU memory) is a significant further allocation and must not be enabled without
checking headroom first.

Note that 3D frames do **not** come from `env.render()`. MetaDrive returns them through a camera
sensor attached to the vehicle, inside the observation dict as `obs["image"]`, shaped
`(H, W, C, T)` where `T` is a stack of the most recent frames — newest last.

### A blank render does not raise an exception

Learned the hard way while this project still used highway-env (D-018): the conventional headless
fix `SDL_VIDEODRIVER=dummy` produced **all-black frames** with correct shape and dtype and no
error. The resulting PNGs were 342 bytes — a perfectly plausible size for a real image.

The lesson generalizes beyond that one flag and still applies to MetaDrive:

> **Validate any rendering change with a pixel statistic** — `frame.mean()` and unique colour
> count — never with file existence or file size. A training run will happily record hours of
> black video without complaining.

For reference, healthy values measured here: MetaDrive top-down `mean=253.95`, 25 colours;
MetaDrive 3D `mean=145.25`, 26,322 colours. Blank is `mean=0.00`, 1 colour.

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
- [x] Whether the workstation can push to GitHub — **it cannot, yet.** See below.

## Blocking issue: GitHub authentication is not configured on either machine

The git half of the sync loop does not work yet. Both ends fail, for different reasons.

**Workstation** — `git push` over HTTPS fails with
`could not read Username for 'https://github.com'`. No credential helper or cached token is
configured for this remote.

**Mac** — `gh` CLI reports its keyring token is invalid, and `git@github.com` over SSH returns
`Permission denied (publickey)`.

Resolving this requires a credential decision, so it is left for explicit choice rather than
configured unilaterally. Reasonable options, roughly in order of convenience:

1. `gh auth login` on the workstation, which also configures git's credential helper for HTTPS
2. An SSH deploy key or account key on the workstation, and switch `origin` to the SSH URL
3. A personal access token stored in a credential helper

The Mac needs its own fix regardless, since pulling for review is the other half of the loop.

**Current state in the meantime**: the repository exists with full history on *both* machines and
`origin` is configured on both. The two are in sync via a direct `rsync` over Tailscale, which
works today and needs no GitHub credentials:

```bash
# Mac -> workstation
rsync -az ~/Keerthan/Projects/RF_Self_Driving_Car/ keerthan@100.71.12.16:~/RF_Self_Driving_Car_Eval/

# workstation -> Mac
rsync -az keerthan@100.71.12.16:~/RF_Self_Driving_Car_Eval/ ~/Keerthan/Projects/RF_Self_Driving_Car/
```

This is a stopgap, not the intended workflow — it has no conflict detection and will silently
overwrite divergent edits, so only one machine should be edited at a time until git push works.
Note also that macOS ships an old `rsync` that rejects `--info=stats1`; use `--stats`.

The GitHub repository was empty as of this writing, so the first push will establish `main`
without any risk of clobbering existing history.

**Directory names differ by design**: `~/Keerthan/Projects/RF_Self_Driving_Car` on the Mac,
`~/RF_Self_Driving_Car_Eval` on the workstation (matching the repo name).
