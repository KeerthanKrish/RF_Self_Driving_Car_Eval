# Technical Design

**Status**: MetaDrive selected and verified working on `dtgpu`. Values marked **verified** were
measured on the installed version, not taken from documentation.
**Last updated**: 2026-09-10

Covers the concrete decisions: which simulator, what the car actually is, what the agent sees and
does, how reward is built, which algorithms get implemented in what order, and what the
supporting machinery has to be.

> **Changed 2026-09-10.** This document previously specified highway-env. The project moved to
> MetaDrive because a physics-backed 3D environment was required — see D-019 in the decision log
> for the full rationale and the alternatives rejected.

---

## 1. Simulator: MetaDrive

[MetaDrive](https://github.com/metadriverse/metadrive) — an open-source driving simulator built
specifically for reinforcement learning research. **Version 0.4.3 installed and verified.**

### What it actually is

A **Python package**, not a separate application you launch and connect to over a socket. That
distinction matters: it imports directly, so there is no server process to manage, no port
handshake, and no version-matched client library.

- **Rendering**: Panda3D, a real 3D game engine, with physically-based rendering
- **Physics**: Bullet — the car is a rigid-body chassis with four wheels and joints under
  constraints, not a point whose position is computed
- **API**: Gymnasium-native
- **Size**: ~100 MB plus downloaded assets
- **Roads**: procedurally composed from building blocks (straight, curve, intersection,
  roundabout, ramp, merge), so the supply of distinct maps is effectively unlimited
- **Already populated**: scripted IDM traffic, **traffic lights**, pedestrians, cyclists

### Why it fits this project

- **It has real physics**, which was the requirement that triggered the change. MetaDrive's own
  paper draws the contrast explicitly, describing highway-env and similar tools as ones that
  "only simulate vehicle with simple kinematic model."
- **~300 FPS** single-agent with traffic — roughly **12× CARLA's** ~25 FPS. With hand-written
  algorithms needing multi-seed comparisons, this is the difference between an hour and most of a
  day per experiment.
- **Procedural map generation** is the one empirically validated generalization lever in
  `research/05`, and it is this simulator's core mechanism rather than an add-on.
- **Traffic lights and pedestrians ship with it**, which deletes a substantial chunk of
  simulator engineering from Phase 4 — engineering that would have taught nothing about RL.
- **It coexists with IsaacLab** on the shared GPU. Measured peak: ~1.6 GB, leaving 8.8 GB free.

### Rejected alternatives

| Rejected | Reason |
|---|---|
| **highway-env** (the previous choice) | No physics engine at all — kinematics integrated in Python, collisions by rectangle intersection, 2D rectangles for visuals. Also has no traffic lights or road signs. |
| **CARLA** | Photorealism is not needed. Costs ~25 FPS, a 130–170 GB install, and 16 GB VRAM against a card already hosting IsaacLab — leaving ~10.7 GB, below the 12 GB threshold at which CARLA warns the default map may fail to load. |
| **Isaac Sim / Isaac Lab** | Already installed on the workstation, so seriously considered. Ruled out on NVIDIA's own guidance: Isaac Sim "is primarily developed for simulating indoor robots operating in structured, controlled environments," has "no assets related to autonomous vehicles," does not support OpenDRIVE road formats, and NVIDIA explicitly does not recommend it for outdoor autonomous-vehicle simulation. |
| **MuJoCo / PyBullet** | Excellent physics, but **no road, no car, no traffic, no driving environment**. Choosing these means personally building vehicle models, lane geometry, and traffic before training anything — weeks of MJCF/URDF modelling that teaches simulation engineering, not RL. |
| **Gazebo** | Has a Prius-and-city demo, but drags in ROS and is slow for RL throughput. |

### Known limitation

Visual fidelity is below CARLA's by design — the MetaDrive paper is explicit that it "trades off
the visual appearance quality for its high sample efficiency." Textured roads, lane markings,
terrain, and sky, but not photorealism. This is accepted: photorealism is a project non-goal.

One documentation caveat worth recording: the 2021 MetaDrive paper lists "no pedestrians or
cyclists" under limitations. **That is stale.** Pedestrian and cyclist support landed in 2023,
traffic lights the same year, with traffic-light detection updated January 2025. Read the
repository, not the paper, for current capability.

---

## 2. The car: Bullet rigid body

The ego vehicle is a **rigid-body chassis with four wheels and joints**, simulated by Bullet —
not a kinematic approximation. Wheels have contact with the road surface, so the vehicle has
traction, weight transfer, and can be unsettled by its own maneuvers in ways a bicycle model
cannot express.

**Timing** (MetaDrive defaults): physics substep `dt = 0.02 s`, `decision_repeat = 5`, so one
`env.step()` advances **0.1 s** of simulated time, with Bullet integrating five substeps inside
it. The agent therefore acts at 10 Hz while physics runs at 50 Hz.

**Other traffic** is driven by MetaDrive's `IDMPolicy` — the Intelligent Driver Model from
`research/09`. These are part of the environment's dynamics, not agents to coordinate with, which
is what keeps this a single-agent problem and multi-agent RL out of scope.

### How this differs from what was planned before

| | highway-env (previous) | MetaDrive (now) |
|---|---|---|
| Vehicle model | Kinematic bicycle, integrated in Python | Bullet rigid body: chassis, wheels, joints |
| Collisions | Rectangle intersection test | Physics contacts |
| Renderer | 2D coloured rectangles | Panda3D 3D with PBR |
| Traffic lights | None | Native |
| Pedestrians | None | Native |
| Speed | thousands FPS | ~300 FPS |

Still not comparable to Isaac Sim or MuJoCo in physics fidelity, and that is fine — the
requirement was a physics-backed car on a road that can be watched, not vehicle-dynamics research.

---

## 3. Observation space

MetaDrive offers two families, and the project uses both at different phases.

### Vector observation (default) — **verified `(259,)` float32, `Box(0, 1)`**

Composed of ego state, navigation information, and a **240-beam LiDAR-style distance scan** of
surrounding objects. The LiDAR beam count is configurable and 240 is more than this project needs
early on; reducing it is a cheap way to shrink the input and speed up learning.

### Image observation — **verified `(360, 640, 3, 3)` uint8 + `(19,)` state**

An RGB camera mounted on the vehicle, returned as a dict with `image` and `state` keys. The
trailing dimension is a **stack of the 3 most recent frames** — MetaDrive's built-in answer to
partial observability, since a single frame carries no velocity information. The accompanying
state vector drops to 19 dimensions because the LiDAR scan is replaced by pixels.

`image_on_cuda=True` keeps rendered frames in GPU memory as tensors instead of round-tripping to
CPU, reported as a large speedup. **Not to be enabled without checking GPU headroom first**, given
the shared card.

### Staging

| Phase | Observation | Why |
|---|---|---|
| 0–2 | Vector, with LiDAR beams reduced | Small and fast; the MDP loop and algorithm correctness are what is being learned, not perception |
| 3 | Vector, full | Continuous control with richer surroundings |
| 4 | Vector + explicit capability features (e.g. signal phase) | How state design changes what is learnable |
| 5 (optional) | Image | Representation learning; the VAE-then-RL pattern from `research/12` |

---

## 4. Action space

**Verified: `Box(-1, 1, (2,))` — continuous `[steering, acceleration]`. MetaDrive is
continuous-only.**

This is a real consequence of the simulator change and needs planning for:

> **DQN needs discrete actions, and MetaDrive does not provide them.** Phase 1 requires a
> discretizing wrapper — a small `gymnasium.ActionWrapper` mapping a handful of discrete choices
> (steer left / straight / right × accelerate / coast / brake) onto continuous pairs.

highway-env supplied `DiscreteMetaAction` for free; MetaDrive does not. The wrapper is perhaps
thirty lines, and writing it is itself instructive — it makes the discrete/continuous distinction
concrete rather than a configuration flag. But it is a new component, and per the operating
principle it must be verified **before** being used to validate a hand-written DQN, or there
would be two unknowns at once. It gets tested by running SB3's DQN through it.

| Phase | Action | Notes |
|---|---|---|
| 1–2 | Discretized via wrapper | Needed for DQN. Wrapper validated with SB3 first. |
| 3+ | Native continuous | What SAC and PPO-continuous are for |

---

## 5. Reward design

Unchanged in method by the simulator switch — see `research/04`. The method matters more than the
platform.

### Core reward is sparse and outcome-only

```
+1   on reaching the destination
-1   on collision
 0   otherwise, including timeout
```

Timeout scores `0` rather than negative **on purpose**, so "did nothing" can never rank below
"crashed."

### Density comes only from potential-based shaping

```
F(s, s′) = γ · Φ(s′) − Φ(s)
```

with `Φ(s)` = negative remaining route distance. Ng, Harada & Russell (1999) proved this cannot
change the optimal policy. It cannot rescue a broken core reward, so the core is validated first.

### Correctness is enforced by tests

Preference ordering becomes a `pytest` suite over hand-constructed trajectory pairs, run before
training. Of the papers in the Knox et al. audit that disclosed their process, **all 8 tuned by
trial and error against observed behaviour** — which is what let 7 of 9 published reward functions
rank a mid-drive crash above safe idling.

### The Phase 0 exercise, now against MetaDrive's reward

MetaDrive ships its own reward: a driving-progress term along the lane, a speed term, and
terminal penalties for collision and leaving the road. **Verified**: 88 steps of mild throttle
with no steering produced a total reward of 49.25 while ending `out_of_road=True` — that is a
substantial positive return for an episode that failed.

That single observation is the whole motivation for the exercise. The first real task is to read
the reward function out of the installed source and run the trajectory-pair test on it: **does it
prefer a fast run that crashes over a slow one that survives?** The answer is interesting either
way, and it makes the dossier's central warning concrete on code actually being used.

### Safety as cost, not reward

MetaDrive has a **native safe-RL cost signal** (`cost = 1` on collision or off-road), tracked
separately from reward. This is the convention `research/05` recommends, available without having
to build it.

---

## 6. Algorithms: what gets implemented, in what order

Each is written by hand, then validated against Stable-Baselines3 on identical environment, seeds,
and hyperparameters. "Correct" means **learning curves match within seed noise**.

| # | Algorithm | Action space | Concepts it forces |
|---|---|---|---|
| 0 | Random + scripted baseline | — | Env API, evaluation harness, where the floor is |
| 1 | Tabular Q-learning / SARSA (toy gridworld) | discrete | Bellman equations, TD error, on- vs off-policy, ε-greedy |
| 2 | **DQN** | discrete (via wrapper) | Function approximation, replay buffer, target networks, deadly triad |
| 3 | Double DQN, Dueling DQN | discrete | Overestimation bias; value/advantage decomposition |
| 4 | REINFORCE | discrete | Policy gradient theorem, Monte Carlo returns, variance |
| 5 | Actor-Critic / A2C | discrete | Baselines, advantage, bias–variance tradeoff |
| 6 | **PPO** | both | GAE, importance sampling, clipped surrogate objective |
| 7 | **SAC** | continuous | Max-entropy RL, twin critics, reparameterization |
| 8 (optional) | Model-based / TD-MPC2 | continuous | World models, planning — connects to prior SO-101 work |

**The highest-value exercise in the list**: once DQN works, remove the replay buffer, then restore
it and remove the target network, and watch training destabilize each time.

### Policy representations encountered

ε-greedy over Q-values (value-based), categorical (discrete policy gradient), squashed Gaussian
(SAC), deterministic (evaluation). Networks stay small — 2–3 hidden layers of 64–256 units for
vector observations. Capacity is not the bottleneck at this scale.

---

## 7. Libraries

| Library | Role | Version |
|---|---|---|
| `metadrive-simulator` | Simulator | **0.4.3** |
| `panda3d` | 3D rendering (via MetaDrive) | 1.10.13 |
| `gymnasium` | Environment API contract | 1.3.0 |
| `torch` | Autodiff, NN, optimizers | 2.11.0+cu128 |
| `numpy` | Numerics | 2.4.6 |
| `stable-baselines3` | **Correctness oracle only** | 2.9.0 |
| `pytest` | Reward and component tests | 9.1.1 |
| `tensorboard` | Metric logging | 2.21.0 |
| `imageio` / `imageio-ffmpeg` | Episode video capture | 2.37.4 |

Deliberately **not** used: Ray/RLlib (distributed complexity with no payoff here), CleanRL as a
dependency (a *reading* resource — the point is writing these), CARLA.

**Note**: installing MetaDrive replaced `pygame-ce` with `pygame`, which may break highway-env's
renderer. Irrelevant since highway-env is no longer the project environment, but it is why
highway-env should not be assumed still functional in this environment.

---

## 8. The parts not in the original question, which matter as much

### Evaluation harness, kept strictly separate from reward

Training reward is the optimization signal, **not** the measure of success — conflating them is
how reward hacking goes unnoticed. Evaluation is separate, deterministic, and run on **held-out
seeds never used for training**: success rate (route completed without collision), collision rate,
off-road rate, mean episode length and distance, and cost reported separately.

MetaDrive's `info` dict supplies `arrive_dest`, `crash`, `out_of_road`, and `velocity` directly —
**verified present** — so these metrics come from the simulator rather than being inferred.

### The fast fallback environment

The operating principle needs an environment where a hand-written algorithm can be proven correct
with instant feedback. MetaDrive at ~300 FPS is good but not instant, and a driving task has many
ways to fail that have nothing to do with algorithm correctness.

**Classic control fills this role**: CartPole for DQN and discrete PPO, Pendulum for SAC. They run
in seconds, SB3's reference results on them are exhaustively validated, and if a hand-written PPO
fails on CartPole the bug is definitely in the algorithm. This replaces the role highway-env would
have played, without maintaining a second driving simulator.

### Seeding and multiple runs

RL results have enormous seed variance. **A single-seed result is noise, not a finding.** Every
reported claim runs ≥3 seeds, ideally 5, with variance shown. This is also the criterion for "my
DQN matches SB3's" — overlapping bands, not one lucky curve.

MetaDrive adds a second axis: `num_scenarios` and `start_seed` control which procedurally
generated maps appear. **Training and evaluation must use disjoint scenario ranges**, or
"generalization" is being measured on maps already trained on.

### Rendering discipline

Three modes, at different costs — details and the measured GPU figures are in
`docs/01_infrastructure_and_workflow.md`:

- **no rendering** — training default, no GPU
- **top-down** — CPU schematic, for debugging
- **3D camera** — ~1.6 GB GPU, for review videos only, never during routine training

A blank render does not raise an exception. Any change to the rendering path is checked with a
pixel statistic, never file existence (D-018).

### Config and reproducibility

Every run writes its fully-resolved config and the git SHA it launched from. A result that cannot
be traced to exact code and exact config is not a result.

### Hyperparameters

Start from `rl-baselines3-zoo` values for the same algorithm family, so that when a hand-written
implementation fails to learn, hyperparameters are not a competing explanation. Tune later,
deliberately, one thing at a time.
