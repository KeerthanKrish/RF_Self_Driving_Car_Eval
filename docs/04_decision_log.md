# Decision Log

Append-only record of decisions, why they were made, and what evidence backed them. Written as
decisions happen, not reconstructed afterward.

Format: newest last. When a decision is reversed, add a new entry rather than editing the old one,
so the reasoning trail stays intact.

---

### D-001 — Project is learning-first, not performance-first
**Date**: 2026-09-10
**Decision**: The objective is learning RL fundamentals, methodology, and implementation. The
driving agent is the vehicle for that, not the goal.
**Rationale**: Stated directly. This inverts several of the research dossier's recommendations,
which were compiled under the assumption that a good driving agent was the objective.
**Consequence**: Algorithms get written by hand; Stable-Baselines3 is demoted from implementation
to correctness oracle. Documented in `docs/00_project_charter.md`.

---

### D-002 — Simulation only, permanently
**Date**: 2026-09-10
**Decision**: No real hardware at any point. No Duckiebot, Donkey Car, or F1TENTH.
**Rationale**: Stated directly and unambiguously.
**Consequence**: The entire sim-to-real chapter of `research/05` is inert. Safety-during-exploration
stops being a first-class design constraint. Platform choice is freed to optimize purely for
learning value, which is what makes highway-env the right pick over Duckietown.

---

### D-003 — Incremental capability progression rather than one target task
**Date**: 2026-09-10
**Decision**: Build capability by capability — obstacles, then moving obstacles, then road signs,
and onward — rather than aiming at a full driving task from the start.
**Rationale**: Stated directly. Also happens to be well supported: `research/10` documents large
quantified gains from curriculum learning in driving RL specifically.
**Consequence**: The progression is chosen so each capability forces the next RL concept.
`research/10`'s documented failure mode applies directly — a stage built on a poorly-trained
predecessor is worse than no curriculum — so every phase gets a competence gate.

---

### D-004 — Simulator is highway-env — ⚠️ SUPERSEDED BY D-019
**Date**: 2026-09-10
**Status**: **Superseded the same day by D-019**, which moves the project to MetaDrive after a
physics-based, visually real environment became a stated requirement. Retained for the reasoning
trail; do not act on it.
**Decision**: highway-env, with MetaDrive held as fallback.
**Rationale**: Gymnasium-native, pure Python and readable end to end, CPU-fast, and per
`research/09` the only surveyed platform exposing literal IDM/MOBIL constants rather than an
abstraction over them. Readability is weighted heavily because the plan involves extending the
environment, not just consuming it.
**Known limitation accepted**: highway-env has no traffic lights or road signs. This is a
scheduled transition at Phase 4c, not a surprise. Intended response is to extend highway-env by
hand rather than switch platforms, keeping kinematics and rendering trusted.

---

### D-005 — Operating principle: never debug two unknowns at once
**Date**: 2026-09-10
**Decision**: Phases are ordered so at most one of {environment, algorithm} is unproven at a time.
Trusted env + trusted algo, then trusted env + my algo, then my env + my trusted algo.
**Rationale**: A hand-written algorithm failing in a hand-written environment gives no signal
about which is broken.
**Consequence**: Determines the entire phase structure in `docs/03_roadmap.md`. Also the reason
SB3 stays in the dependency list despite algorithms being hand-written.

---

### D-006 — Reward correctness enforced by tests, not by observation
**Date**: 2026-09-10
**Decision**: Sparse outcome-only core reward, densified only via potential-based shaping.
Preference ordering asserted in a `pytest` suite over hand-constructed trajectory pairs, run
before training.
**Rationale**: `research/04`. Knox et al. found 7 of 9 published driving reward functions rank a
mid-drive crash above safe idling, and all 8 papers that disclosed their process used
trial-and-error tuning against observed behavior. Prior SO-101 experience with reward hacking and
PBRS transfers here nearly unchanged.
**Consequence**: Timeout is scored `0`, never negative. Safety is tracked as a separate cost
signal rather than summed into reward.

---

### D-007 — Research dossier moved to `research/`, new project docs in `docs/`
**Date**: 2026-09-10
**Decision**: The 13 pre-existing research files moved into `research/`. New project-scoped
documentation created under `docs/`.
**Rationale**: The dossier is reference material compiled under a different assumption about the
project's goal; the new docs are the actual plan. Mixing them at the root made it unclear which
was which, and `research/00_overview.md` indexed only files `01`–`06` while `07`–`12` also
existed, so a top-down reader would never find half of it.
**Consequence**: Baseline was committed before the move, so the original flat layout is
recoverable from history. `git mv` used to preserve file history.

---

### D-008 — Large artifacts stay out of git
**Date**: 2026-09-10
**Decision**: Videos, checkpoints, and raw logs move by `scp`; git carries code, docs, configs,
and small results only. Enforced in `.gitignore`.
**Rationale**: Stated directly. Also avoids bloating a repo whose purpose is fast sync between
two machines.
**Consequence**: Run artifacts are referenced in docs by run ID rather than committed. Run
directories are timestamped and never reused, per the collision-proofing requirement in
`working_instructions.md`.

---

### D-009 — GPU is not expected to help in early phases
**Date**: 2026-09-10
**Decision**: Default to CPU through Phase 1–2; revisit when observations become images or many
parallel environments are needed.
**Rationale**: small observation vectors through small MLPs are dominated by kernel-launch
overhead, and environment stepping is CPU-bound. GPU would likely be slower.
**Still holds under D-019, with one addition**: MetaDrive's physics runs in Bullet on CPU and its
vector observation is 259-dimensional — still far too small for the GPU to help training. What
changed is that MetaDrive's **3D rendering** does use the GPU (~1.6 GB measured), so the card is
now needed for review videos and any future image-observation work, just not for routine
training.
**Consequence**: Also reduces contention on a workstation shared with other projects. Recorded so
that slow Phase 1 training is not misdiagnosed as a GPU configuration problem.

---

### D-010 — Workstation verified; PyTorch pinned to ≥2.7 / CUDA 12.8+
**Date**: 2026-09-10
**Decision**: Environment is a conda env named `rlsdc` under the existing miniforge3 install.
PyTorch must come from the CUDA 12.8-or-newer index, and `sm_120` must be confirmed present in
`torch.cuda.get_arch_list()` before the environment is considered working.
**Rationale**: Survey of `dtgpu` found an RTX 5060 Ti with **compute capability 12.0** —
Blackwell. PyTorch wheels built against CUDA 12.1 or 12.4 contain no `sm_120` kernels. They
install without complaint and fail only at first GPU op, with an error that reads like a driver
fault rather than a wheel mismatch. `torch.cuda.is_available()` returns `True` regardless, so it
is not a valid check.
**Also confirmed**: miniforge3 / conda 26.5.3; existing envs `base`, `env_isaaclab`, `lerobot`,
so `rlsdc` does not collide; 8 CPU cores (caps vectorized envs around 6); 31 GB RAM; ~736 GB free
disk; an IsaacLab process actively holding ~5.3 GB VRAM, confirming GPU sharing is real.
**Consequence**: Environment setup gets an explicit verification step rather than being assumed
to have worked. `conda` must be sourced explicitly in non-interactive ssh commands.

---

### D-011 — Credentials are not inspected; push access is tested by pushing
**Date**: 2026-09-10
**Decision**: Do not read `~/.gitconfig`, `~/.ssh/`, stored credential files, or `gh auth status`
on the workstation to determine whether push works. Attempt the push instead and handle failure.
**Rationale**: Inspecting credential-bearing state is broader access than the question requires,
and the question is answerable directly by the operation itself.
**Consequence**: GitHub push capability from the workstation is listed as unverified rather than
assumed, and gets settled by the first real push.
**Outcome**: The push was attempted and failed — HTTPS has no cached credentials on the
workstation. Recorded in `docs/01_infrastructure_and_workflow.md` as a blocking issue awaiting a
credential decision. No further auth probing was done.

---

### D-012 — `rsync` over Tailscale as an interim sync, git as the intended one
**Date**: 2026-09-10
**Decision**: Until GitHub auth is configured, keep the two machines in sync with direct `rsync`
over Tailscale. Git remains the intended mechanism and `origin` is already configured on both.
**Rationale**: Neither machine can currently authenticate to GitHub, but the work needs to move
between them now. Tailscale already provides an authenticated direct path.
**Consequence and risk**: `rsync` has no conflict detection and will silently overwrite divergent
edits, so **only one machine may be edited at a time** while this stopgap is in use. This is a
real footgun and the reason it is explicitly temporary rather than adopted as the workflow.
**Partially superseded by D-013.**

---

### D-013 — `origin` uses the SSH URL, not HTTPS
**Date**: 2026-09-10
**Decision**: `origin` is `git@github.com:KeerthanKrish/RF_Self_Driving_Car_Eval.git` on both
machines.
**Rationale**: The earlier push failure (D-011) was self-inflicted — `origin` had been configured
with the HTTPS URL, for which no credentials are cached, while the workstation already had a
registered SSH key. Switching the URL fixed it immediately.
**Outcome**: **Workstation → GitHub push works.** `main` is published.
**Still outstanding**: the Mac's SSH key is not registered on the GitHub account, so
`git fetch` from the Mac still fails with `Permission denied (publickey)`. Fix is to add
`~/.ssh/id_ed25519.pub` to the GitHub account. Until then the Mac syncs by `rsync`, and D-012's
one-machine-at-a-time warning still applies to the Mac side.
**Lesson worth keeping**: an auth failure is not automatically a credentials problem. Check the
remote URL scheme first.

---

### D-014 — Tabular gridworld detour is kept (Phase 1a)
**Date**: 2026-09-10
**Decision**: Keep the short tabular detour — policy iteration, Q-learning vs SARSA on a toy
gridworld — before DQN, rather than starting directly on highway-env.
**Rationale**: Chosen deliberately. Tabular methods need a small discrete state space, and seeing
Bellman backups converge directly is worth an afternoon before they are buried inside a neural
network. The stated preference to "learn while doing" is satisfied by keeping it short and
concrete rather than by skipping it.
**Consequence**: Phase 1a stays in the roadmap as written.

---

### D-015 — Roadmap stays gated on completion, never on calendar time
**Date**: 2026-09-10
**Decision**: No wall-clock estimates or target dates anywhere in the project docs. Phases
advance on their definition-of-done only.
**Rationale**: Availability is bursty and unpredictable. An estimate built on a guessed time
budget would be treated as a commitment and would create pressure to advance a phase before its
competence gate is met — which is precisely the curriculum failure mode `research/10` documents,
where a stage built on a poorly-trained predecessor performs worse than no curriculum at all.
**Consequence**: Scheduling pressure and the main technical risk point the same direction here,
which is convenient. "On track" means the current phase's gate is not yet met, nothing more.

---

### D-016 — Environment setup is a versioned script, not ad-hoc commands
**Date**: 2026-09-10
**Decision**: `scripts/setup_env.sh` creates the environment; `scripts/verify_env.sh` proves it
works. Both live in the repo.
**Rationale**: The charter requires the environment be reconstructible rather than a hand-built
artifact existing only on one machine. It also makes the cu128 pin self-documenting at the point
of use.
**Consequence**: `verify_env.sh` asserts `sm_120` is in `torch.cuda.get_arch_list()` **and**
runs a real GPU matmul, because `torch.cuda.is_available()` returns `True` even when no
compatible kernels exist and is therefore not a valid check.

---

### D-017 — Environment isolation is enforced by an assertion, not by care
**Date**: 2026-09-10
**Decision**: `scripts/setup_env.sh` aborts before any `pip install` unless
`CONDA_DEFAULT_ENV == rlsdc` **and** `which python` resolves to
`~/miniforge3/envs/rlsdc/bin/python`. It also exports `PIP_USER=0`.
**Rationale**: Raised directly as a concern — other projects on this machine have their own torch
builds and must not be disturbed. A convention that depends on remembering to activate the right
environment is not a safeguard; an assertion that fails loudly is.
**Verified at the time of raising**: `base` had no torch, `env_isaaclab` 2.7.0+cu128, `lerobot`
2.11.0+cu130, `rlsdc` 2.11.0+cu128, `~/.local` and system Python both free of torch. Because the
other environments hold *different* versions than this project's, a leak would show up as a
version match rather than needing to be inferred — none was present.
**Standing rule**: this applies to every install, not just torch. Nothing is installed outside
`rlsdc`. No `sudo pip`, no `pip --user`, no changes to `base` or another project's environment.
An audit command is in `docs/01_infrastructure_and_workflow.md`.

---

### D-018 — Headless rendering uses `OFFSCREEN_RENDERING=1`, never `SDL_VIDEODRIVER=dummy`
**Date**: 2026-09-10
**Decision**: Render via `rgb_array` with highway-env's `OFFSCREEN_RENDERING=1` set before pygame
is imported. Write PNG/MP4 into the run directory and `scp` to the Mac for review.
**Rationale**: The workstation has no display. The conventional headless-pygame fix
(`SDL_VIDEODRIVER=dummy`) was tried first and **silently produced all-black frames** — correct
shape and dtype, every pixel zero, no error raised. Measured across four configurations: both
variants with the dummy driver gave mean 0.00 and one unique colour; both without it gave mean
103.95 and seven colours.
**Consequence**: A rendering regression here would not throw — it would quietly record hours of
black video. Any change to the rendering path must be validated with a pixel statistic
(`frame.mean()`, unique colour count), never by confirming a file exists or has plausible size.
The first attempt's PNGs were 342 bytes, which is a perfectly reasonable size for a blank image;
file size is not evidence of content.
**Also settled**: highway-env is pygame 2D with no physics engine — not Isaac Sim, MuJoCo, or
Gazebo. Comparison table added to `docs/02_technical_design.md`.
**Superseded by D-019**, which replaces highway-env entirely. The rendering discipline in this
entry still stands and carried over.

---

### D-019 — Simulator changed from highway-env to MetaDrive
**Date**: 2026-09-10
**Supersedes**: D-004 (highway-env)

**Decision**: MetaDrive 0.4.3 is the project environment. highway-env is retired. CARLA, Isaac
Sim, MuJoCo, PyBullet and Gazebo were considered and rejected.

**Trigger**: a requirement that had not been stated before — a physics-based engine producing a
visually real result, so that progress can actually be watched. highway-env fails this
categorically: no physics engine, 2D coloured rectangles. The follow-up clarification mattered
too — realistic *vehicle* physics is **not** required, which lowers the bar considerably and is
what rules CARLA out rather than in.

**Why MetaDrive**:
- Real Bullet rigid-body vehicle dynamics — chassis, wheels, joints under constraints. Its own
  paper contrasts this against highway-env, which "only simulates vehicle with simple kinematic
  model."
- 3D rendering via Panda3D. **Verified**: a 640×360 frame with mean 145.25 and 26,322 unique
  colours — textured road, lane markings, terrain, sky.
- ~300 FPS versus CARLA's ~25 FPS. On a bursty schedule with multi-seed comparisons of
  hand-written algorithms, a 12× throughput difference dominates.
- Ships traffic lights, pedestrians and cyclists, deleting most of Phase 4's simulator work.
- Procedural map generation, which `research/05` identifies as the one empirically validated
  generalization lever.
- A `pip install`, not a separate server process.

**Why not the others**:
- **CARLA**: photorealism is a non-goal. ~25 FPS, 130–170 GB, and 16 GB VRAM recommended against
  a card already holding IsaacLab's 5.3 GB — leaving ~10.7 GB, under the 12 GB at which CARLA
  warns the default map may not load.
- **Isaac Sim**: already installed, so seriously evaluated. Ruled out on NVIDIA's own guidance —
  built for "indoor robots operating in structured, controlled environments," no autonomous-
  vehicle assets, no OpenDRIVE support, and explicitly not recommended for outdoor AV simulation.
- **MuJoCo / PyBullet**: better physics, but no road, car, traffic, or driving environment at
  all. Would mean weeks building a simulator instead of learning RL. Relaxing the physics
  requirement argues *against* these, not for them — their advantage was the thing no longer
  needed.

**Verified on `dtgpu`** before committing to the change:

| Check | Result |
|---|---|
| Headless physics | 88 steps, reward 49.25, GPU untouched |
| Observation (vector) | `(259,)` float32 — ego, navigation, 240 LiDAR beams |
| Observation (image) | `(360, 640, 3, 3)` uint8 + `(19,)` state — 3-frame stack |
| Action space | `Box(-1, 1, (2,))` — **continuous only** |
| CPU top-down render | works, GPU untouched |
| 3D render | works, peak ~1.6 GB GPU |
| IsaacLab disturbed? | **No** — pid 448412 at 5252 MiB, identical before and after |

**Consequences**:
1. **DQN needs a discretizing action wrapper.** MetaDrive is continuous-only; highway-env's
   `DiscreteMetaAction` is gone. New Phase 1b builds it and validates it with SB3 *before* any
   hand-written DQN uses it, so a failure never has two possible causes.
2. **The fast fallback environment is now classic control**, not a second driving sim. CartPole
   for discrete, Pendulum for continuous.
3. **Phase 4 gets substantially easier and better.** Traffic lights and pedestrians were "must
   build from scratch" and are now native. Remaining work shifts from simulator plumbing to
   scenario composition, observation design, and reward validation — the parts that actually
   teach RL.
4. **Train/eval scenario ranges must be disjoint**, since `num_scenarios`/`start_seed` govern
   procedural map generation. Otherwise generalization is measured on memorized maps.
5. Installing MetaDrive replaced `pygame-ce` with `pygame`, which may have broken highway-env's
   renderer. Not a problem, but highway-env should not be assumed functional in `rlsdc`.

**Documentation caveat recorded**: the 2021 MetaDrive paper lists "no pedestrians or cyclists" as
a limitation. That is stale — both landed in 2023, and traffic-light detection was updated in
January 2025. Read the repository, not the paper, for current capability. `research/01` and
`research/03` cite the paper and are therefore out of date on this point.
