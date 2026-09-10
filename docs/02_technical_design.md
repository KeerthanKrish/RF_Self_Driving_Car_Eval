# Technical Design

**Status**: proposed — API specifics marked "confirm" are to be verified against the pinned
version at install time rather than trusted from memory
**Last updated**: 2026-09-10

Covers the concrete decisions: which simulator, what the car actually is, what the agent sees and
does, how reward is built, which algorithms get implemented in what order, and what the
supporting machinery has to be.

---

## 1. Simulator: highway-env

[highway-env](https://github.com/Farama-Foundation/HighwayEnv), maintained by the Farama
Foundation (the same organization that maintains Gymnasium itself).

**Why it wins for this project specifically**, beyond the dossier's Tier 1 recommendation in
`research/06`:

- **Readable end to end.** Pure Python, small codebase. This matters more than usual here because
  the plan is to *modify and extend* the environment in later phases, not just consume it.
- **Textbook-faithful traffic models.** `research/09` establishes that highway-env is the only
  surveyed platform where you edit the actual named constants from the traffic literature
  (`TIME_WANTED`, `POLITENESS`, `DISTANCE_WANTED`) rather than a simulator-specific abstraction.
  When traffic behavior gets tuned by hand, that transparency is the whole point.
- **Gymnasium-native**, so both hand-written algorithms and SB3 attach to it without a wrapper.
- **Seconds-to-minutes training loops** on CPU. Iteration speed is the dominant factor in how
  much gets learned per week.
- **Multiple built-in scenarios** to graduate through: `highway-v0`, `merge-v0`, `roundabout-v0`,
  `intersection-v0`, `racetrack-v0`, `two-way-v0`, `exit-v0`, `u-turn-v0`, `parking-v0`.

**The limitation that will eventually force a decision, stated now rather than discovered later:**

> highway-env has **no traffic lights and no road signs**. It models vehicles, lanes, and
> collisions — not traffic-control infrastructure.

The stated capability progression (obstacles → moving obstacles → road signs → …) therefore runs
out of built-in support exactly at "road signs." That is not a problem, it is a scheduled
transition, and it lands naturally at the point where the roadmap wants to move from *trusted
environment* to *my environment*. When it arrives, the intended answer is to **extend highway-env
by hand** rather than switch platforms — a new road-object type, a signal-phase state machine, an
observation feature, and a termination condition — because that keeps kinematics, rendering, and
traffic models as trusted components while only the new capability is unproven. Switching to
MetaDrive is the fallback if extension turns out to fight the codebase.

**Rejected alternatives** (full survey in `research/01`):

| Rejected | Reason |
|---|---|
| MetaDrive | Better for generalization research, heavier, less readable. Held as the fallback if extending highway-env fails. |
| CARLA | Photorealism is a non-goal. Would add GPU cost, install weight, and porting churn (`research/07`: third-party CARLA wrappers age out within 1–2 years) for zero learning benefit here. |
| Waymax / GPUDrive / nuPlan | Require Waymo dataset licenses and JAX/CUDA fluency; solve a large-scale realism problem this project does not have. |
| Custom environment from zero | Would mean debugging my kinematics *and* my algorithm simultaneously — the exact failure mode the charter's operating principle exists to prevent. |

---

## 2. The car: kinematic bicycle model

The ego vehicle is a **kinematic bicycle model** — the standard abstraction across driving RL.
No tire slip, no suspension, no soft-body dynamics. Position, heading, and speed only.

State: position `(x, y)`, heading `ψ`, speed `v`.
Control inputs: acceleration `a`, steering angle `δ`.

```
ẋ = v · cos(ψ + β)
ẏ = v · sin(ψ + β)
ψ̇ = (v / l) · sin(β)
v̇ = a
β = arctan( ½ · tan(δ) )        # slip angle at the center of gravity
```

`β` is the correction for measuring from the center of gravity rather than the rear axle. Vehicle
footprint is roughly 5.0 m × 2.0 m (confirm against pinned version).

**Two distinct vehicle classes matter**, and conflating them causes confusion later:

- **The ego vehicle** — driven by the learned policy. This is the agent.
- **`IDMVehicle`** — scripted traffic, following IDM for car-following and MOBIL for lane changes.
  These are *part of the environment's dynamics*, not agents to coordinate with. Treating them as
  environment is what keeps this a single-agent problem and keeps multi-agent RL out of scope.

IDM and MOBIL parameters, with highway-env's defaults, are tabulated in `research/09`. The one
worth knowing up front: **MOBIL's politeness factor defaults to `0.0`**, meaning scripted traffic
is purely selfish and will not yield courteously. If traffic feels unrealistically aggressive,
that constant is why.

**Two clocks**, which trip people up:
- `simulation_frequency` (~15 Hz) — physics integration
- `policy_frequency` (~1 Hz by default) — how often the agent acts

The agent acts far less often than the world updates. Raising `policy_frequency` gives finer
control and longer episodes in agent-steps; it also makes credit assignment harder. This is a
tuning lever with real pedagogical content, not a detail.

---

## 3. Observation space

Deliberately staged, because each representation teaches something different.

| Phase | Observation | Shape | What it teaches |
|---|---|---|---|
| 1 | `Kinematics` | V×F, default 5 vehicles × 5 features, flattened to ~25 | The MDP loop with a small, fully-visible state. No perception problem. |
| 2 | `Kinematics` + explicit capability features | ~25 + n | How state design changes what is learnable — e.g. adding signal phase |
| 3 | `OccupancyGrid` | grid × channels | Spatial representation, CNN encoders, why structure in the input matters |
| 4 (optional) | `GrayscaleObservation` | stacked frames | Representation learning; the VAE-then-RL pattern from `research/12` |

Phase 1 features are `[presence, x, y, vx, vy]` per vehicle, normalized, ego-relative (confirm
defaults against pinned version). This is **privileged state** — the agent gets exact positions
and velocities of nearby cars with no perception pipeline. `research/03` correctly calls this the
largest sim-to-real gap of any modality. That penalty is irrelevant here and the speed is worth
everything.

A concept that arrives for free with this representation: it is a **fixed-size window over a
variable number of vehicles**, so `presence` flags padding. That is a real design pattern with
real consequences (what happens when a sixth car matters?) and worth confronting rather than
hiding.

---

## 4. Action space

| Phase | Action type | Space | Why |
|---|---|---|---|
| 1–2 | `DiscreteMetaAction` | 5 discrete: `LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER` | Enables value-based methods (DQN family). Simple credit assignment. |
| 3+ | `ContinuousAction` | `[acceleration, steering]`, each in `[-1, 1]` | Enables policy-gradient and actor-critic methods (PPO, SAC). |

**The distinction that matters pedagogically**: under `DiscreteMetaAction`, highway-env runs its
own low-level controller to execute the chosen maneuver. The agent is learning *decisions*, and
the control problem is being solved for it. Under `ContinuousAction`, the agent learns actual
control. Moving between them is not a formatting change — it changes what problem is being solved,
and it is the reason SAC and PPO become necessary rather than optional.

---

## 5. Reward design

The area with the highest risk of silent failure, and the one where prior experience with reward
hacking transfers directly. Method comes from `research/04`.

### The core reward is sparse and outcome-only

```
+1   on reaching the destination / completing the route
-1   on collision
 0   otherwise, including timeout
```

Timeout scores `0` rather than negative **on purpose**, so that "did nothing" can never rank below
"crashed."

### Density comes only from potential-based shaping

```
F(s, s′) = γ · Φ(s′) − Φ(s)
```

with `Φ(s)` = negative remaining route distance. Ng, Harada & Russell (1999) proved this cannot
change the optimal policy. The caveat that matters: **PBRS preserves whatever the core reward
already prefers**, so it cannot rescue a broken core reward. Validate the core first, then densify.

### Reward correctness is enforced by tests, not by watching training

Of the papers in the Knox et al. audit that disclosed their process, **all 8 used trial-and-error
tuning against observed behavior**, and that is precisely what let 7 of 9 published reward
functions rank a mid-drive crash above safe idling.

So preference ordering becomes a **`pytest` suite**: construct trajectory pairs by hand, assert
the reward function ranks them correctly, run it in CI before any training. This is cheap, needs
no training, and would have caught every failure in that audit.

### A concrete Phase 1 exercise, and a prediction to test rather than assume

highway-env ships its own default reward for `highway-v0`, roughly a weighted sum of a collision
term, a right-lane term, and a high-speed term, normalized to `[0, 1]` (exact coefficients to be
read from the pinned source, not trusted from memory).

Note the structure: with normalization to `[0, 1]`, a crash scores `0` and **every non-crash step
scores positive**. A crash is punished only by the future reward it forfeits. That is exactly the
shape Knox et al. identify as dangerous.

**Prediction, to be tested and not assumed**: there exists a speed setting at which a fast policy
that crashes partway through outscores a slow policy that survives the full episode. Running the
trajectory-pair test against highway-env's *own default reward* is the first real exercise of the
project — it makes the dossier's central warning concrete on the actual code, and the answer is
interesting whichever way it comes out.

### Safety tracked as cost, not folded into reward

Following MetaDrive's safe-RL convention (`research/05`): collisions and off-road events are logged
as a **separate cost signal**, never silently summed into the reward scalar. Reward-only reporting
can hide a safety regression behind a better score.

---

## 6. Algorithms: what gets implemented, in what order, and why

Each is written by hand, then validated against Stable-Baselines3 on identical environment, seeds,
and hyperparameters. "Correct" means **the learning curves match within seed noise** — not "it
learns something."

| # | Algorithm | Action space | Concepts it forces |
|---|---|---|---|
| 0 | Random + scripted heuristic baseline | discrete | Env API, evaluation harness, what the floor actually is |
| 1 | Tabular Q-learning / SARSA (toy env) | discrete | Bellman equations, TD error, bootstrapping, on- vs off-policy, ε-greedy |
| 2 | **DQN** | discrete | Function approximation, replay buffer, target networks, the deadly triad |
| 3 | Double DQN, Dueling DQN | discrete | Q-value overestimation bias; value/advantage decomposition |
| 4 | REINFORCE | discrete | Policy gradient theorem, Monte Carlo returns, variance |
| 5 | Actor-Critic / A2C | discrete | Baselines, advantage, bias–variance tradeoff |
| 6 | **PPO** | both | GAE, importance sampling, trust regions, the clipped surrogate objective |
| 7 | **SAC** | continuous | Max-entropy RL, twin critics, reparameterization, off-policy continuous control |
| 8 (optional) | Model-based / TD-MPC2 tie-in | continuous | World models, planning — connects to prior SO-101 work (`research/08`) |

Step 1 is a short, deliberate detour into a toy environment rather than highway-env. Tabular
methods need a small discrete state space, and forcing highway-env into one would teach less than
a gridworld does in an afternoon. It exists so that Bellman backups are something seen working
directly, before they are buried inside a neural network.

**The highest-value learning exercise in the whole list**: once DQN works, remove the replay
buffer, then restore it and remove the target network, and watch training destabilize each time.
Reading why they are necessary conveys much less than seeing the failure.

### Policy representations encountered

- **ε-greedy over Q-values** — implicit policy, value-based methods
- **Categorical** — discrete stochastic policy, policy-gradient methods
- **Squashed Gaussian** (`tanh`-transformed) — continuous stochastic policy, SAC
- **Deterministic** — evaluation-time behavior, and DDPG-family if reached

Networks stay small on purpose: 2–3 hidden layers of 64–256 units for vector observations. Model
capacity is not the bottleneck at this scale, and small networks train fast enough to iterate.

---

## 7. Libraries

| Library | Role | Notes |
|---|---|---|
| `gymnasium` | Environment API contract | The interface everything speaks |
| `highway-env` | Simulator | Pinned; source read directly, not just used |
| `torch` | Autodiff, NN, optimizers | CUDA build matched to the workstation's driver |
| `numpy` | Numerics | |
| `stable-baselines3` | **Correctness oracle only** | Never the implementation. Reference curves to validate against. |
| `sb3-contrib` / `rl-baselines3-zoo` | Known-good hyperparameters | `research/07`: fastest way to a credible baseline |
| `pytest` | Reward and component tests | Where the trajectory-pair preference tests live |
| `tensorboard` | Metric logging | Local, no account, sufficient at this scale |
| `imageio` / `moviepy` | Episode video capture | Output goes to `runs/<id>/videos`, moved by `scp` |
| `pyyaml` / `omegaconf` | Config | Deliberately lightweight; Hydra is overkill here |

Deliberately **not** used: Ray/RLlib (distributed complexity with no payoff at this scale, and
`research/08` notes its PPO underperformed reference implementations in a controlled comparison),
CleanRL as a dependency (it is a *reading* resource — the point is writing these, not importing
someone else's single-file version).

---

## 8. The parts not in the original question, which matter as much

### Evaluation harness, kept strictly separate from reward

Training reward is the optimization signal. It is **not** the measure of success — that
conflation is how reward hacking goes unnoticed. Evaluation is separate, deterministic, and run
on **held-out seeds never used for training**:

- success rate (route completed without collision)
- collision rate
- off-road rate
- mean episode length and distance travelled
- cost (cumulative constraint violations), reported separately

Metric definitions are borrowed rather than invented, per `research/05`.

### Seeding and multiple runs

RL results have enormous seed variance. **A single-seed result is noise, not a finding.** Every
reported claim runs ≥3 seeds, ideally 5, with variance shown rather than only the mean. This is
also the criterion for "my DQN matches SB3's DQN" — overlapping bands across seeds, not one lucky
curve.

Environment seeds, action-sampling seeds, and network-init seeds are all set and recorded in
`runs/<id>/config.yaml`.

### Config and reproducibility

Every run writes its fully-resolved config and the git SHA it launched from. A result that cannot
be traced to exact code and exact config is not a result.

### Vectorized environments

Not needed in Phase 1, but worth knowing early: running N environments in parallel is the main
throughput lever for on-policy methods like PPO, and it changes how batches are shaped. Introduce
it when PPO arrives, not before — it complicates the data path and Phase 1 does not need it.

### Hyperparameters

Not a place to be creative early. Start from `rl-baselines3-zoo`'s tuned values for the same
algorithm/environment family so that when a hand-written implementation fails to learn, the
hyperparameters are not a competing explanation. Tune later, deliberately, one thing at a time.
