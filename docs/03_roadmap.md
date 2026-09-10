# Roadmap: Capability Curriculum and Learning Syllabus

**Status**: proposed
**Last updated**: 2026-09-10

Phases are gated by **definition of done**, not by elapsed time. Wall-clock estimates are
deliberately omitted — see "On scheduling" at the end.

## The organizing idea

Each capability added to the environment is chosen because it *forces* the next RL concept into
existence. The concepts are not introduced abstractly and then applied; the environment creates
a problem that the concept is the answer to.

This is a curriculum in the sense of `research/10`, which carries a specific and well-documented
warning: **a curriculum stage built on a poorly-trained previous stage performs worse than no
curriculum at all.** The mitigation is the definition-of-done gate on every phase. Advance on
demonstrated competence, never on a step count or a feeling that it is time to move on.

The phase structure also enforces the charter's operating principle — never more than one unknown
at a time.

---

## Phase 0 — Harness

**Unknowns**: none. Trusted environment, trusted algorithm.
**Purpose**: prove the plumbing before anything interesting depends on it.

- Environment set up and isolated on the workstation; dependencies pinned in the repo
- `highway-v0` runs; random-policy baseline measured
- SB3 PPO or DQN trains on it successfully, purely as a smoke test
- Run-artifact layout working: per-run directory, config, git SHA, metrics CSV, video capture
- Evaluation harness working: held-out seeds, deterministic rollouts, metrics separate from reward
- Multi-seed runs and variance plotting working

**Done when**: a single command launches a run, and a second command produces a multi-seed
evaluation plot with variance bands from the artifacts it wrote.

**RL concepts**: the Gymnasium contract (`reset`, `step`, `terminated` vs `truncated`), episodes,
returns, discounting, why evaluation must be separate from training.

---

## Phase 1 — Fundamentals and the value-based family

**Unknowns**: my algorithm. Environment stays trusted.

### 1a. Tabular detour (toy gridworld, not highway-env)

A short, deliberate diversion. Tabular methods need a small discrete state space, and forcing
highway-env into one would teach less than a gridworld teaches in an afternoon.

- Policy evaluation and policy iteration, so Bellman backups are seen converging directly
- Q-learning and SARSA on the same problem — the on-policy vs off-policy difference made visible
  rather than described
- ε-greedy, and what changing the schedule does

**Done when**: Q-learning and SARSA produce visibly different policies on a cliff-walk-style
problem, and the reason is explainable without notes.

### 1b. DQN from scratch on `highway-v0`

Observation: `Kinematics`. Action: `DiscreteMetaAction`.

- Replay buffer, target network, ε-schedule, Huber loss, all hand-written
- Validated against SB3's DQN: same env, same seeds, same hyperparameters (taken from
  `rl-baselines3-zoo` so hyperparameters are not a competing explanation for failure)
- **Ablation, the highest-value exercise in the project**: remove the replay buffer and watch it
  break; restore it and remove the target network and watch it break differently

### 1c. Double DQN and Dueling DQN

- Demonstrate Q-value overestimation, then show Double DQN reducing it
- Value/advantage decomposition

**Done when**: hand-written DQN matches SB3's learning curve within seed noise across ≥3 seeds,
and the ablations have been run and written up.

**RL concepts**: function approximation, bootstrapping, the deadly triad, off-policy learning,
experience replay, target networks, overestimation bias.

---

## Phase 2 — Policy gradients

**Unknowns**: my algorithm. Environment still trusted.

- REINFORCE — the policy gradient theorem, Monte Carlo returns, and directly observing how
  high-variance it is
- Add a baseline, then a learned critic (A2C) — variance reduction, bias–variance tradeoff
- **PPO** — GAE, importance sampling, the clipped surrogate objective
- Vectorized environments arrive here, because on-policy methods need throughput

**Done when**: hand-written PPO matches SB3's PPO within seed noise, and the variance reduction
from REINFORCE → baseline → critic is measured rather than asserted.

**RL concepts**: policy gradient theorem, score function estimator, advantage estimation, GAE,
trust regions, on-policy vs off-policy data efficiency.

---

## Phase 3 — Continuous control

**Unknowns**: my algorithm. Switch to `ContinuousAction`.

The action space change is the point: the agent now learns actual steering and throttle rather
than picking maneuvers the environment executes for it.

- PPO-continuous — Gaussian policy over actions
- **SAC** — maximum-entropy RL, twin critics, the reparameterization trick, automatic temperature
  tuning
- Direct comparison of PPO vs SAC sample efficiency on the same task, which is the concrete
  version of the claim `research/02` makes in the abstract

**Done when**: hand-written SAC matches SB3's SAC, and the PPO-vs-SAC sample-efficiency comparison
is run over multiple seeds with variance shown.

**RL concepts**: continuous action spaces, stochastic vs deterministic policies, entropy
regularization, squashed Gaussians, twin critics, sample efficiency as a measurable property.

---

## Phase 4 — My environment: incremental capabilities

**Unknowns**: my environment. The algorithms are now trusted, having been validated in Phases 1–3.

This is where the stated progression lives. Each capability is added on top of a working base,
with a competence gate before the next one.

| Step | Capability | Built-in? | RL concept it forces |
|---|---|---|---|
| 4a | Static obstacles | Yes — highway-env has obstacle objects | Exploration under termination risk; reward shaping around hazards |
| 4b | Moving obstacles / dense traffic | Yes — `IDMVehicle` traffic | Partial observability; why a single frame is not a sufficient state; frame stacking or velocity features |
| 4c | Traffic lights | **No — must be built** | Long-horizon credit assignment; conditional behavior; how a state feature changes what is learnable |
| 4d | Stop signs / road signs | **No — must be built** | Rule compliance as constraint vs reward (`research/04`) |
| 4e | Intersections and turns | Partly — `intersection-v0` exists | Multi-objective reward aggregation; the Knox et al. trap in full |
| 4f | Scenario randomization | Partly — needs work | Generalization; the diversity lever from `research/05`, tested on held-out layouts |

**4c is the real transition point.** highway-env has no traffic-control infrastructure at all, so
this is where custom environment work genuinely begins: a signal-phase state machine, a new road
object, an observation feature, a termination/violation condition, and reward or cost handling.
Extending highway-env is preferred over switching platforms, so that kinematics, rendering, and
traffic models remain trusted while only the new capability is unproven.

**Done, per step**: the agent reaches a pre-agreed success-rate threshold on held-out seeds
*before* the next capability is added. This gate is the specific mitigation for the curriculum
failure mode in `research/10`.

**RL concepts**: curriculum learning and its failure mode, state representation design,
constrained MDPs and cost signals, generalization to held-out configurations.

---

## Phase 5 — Capstone integration

Bring the capabilities together into the single artifact the project is meant to produce.

- One environment combining the capabilities from Phase 4
- One or more trained agents, evaluated with the standard harness over multiple seeds
- Written analysis: what worked, what failed, what each ablation showed
- Reproducible from the repo by someone else

**Optional extension**: the model-based tie-in from `research/08`. TD-MPC2 has no driving-simulator
integration, and the dossier estimates 1–3 weeks for a highway-env integration by someone who
already knows the codebase. This connects directly to prior SO-101 work and would produce
genuinely new public information, but it is explicitly optional and should only start once
Phases 0–4 are complete.

---

## Concept coverage check

Tracking that the phases actually cover the fundamentals, rather than assuming they do.

| Concept | Where |
|---|---|
| MDPs, states, actions, rewards, returns, discounting | Phase 0 |
| Bellman equations, policy/value iteration | Phase 1a |
| TD learning, on-policy vs off-policy | Phase 1a |
| Exploration vs exploitation | Phase 1a, 4a |
| Function approximation, the deadly triad | Phase 1b |
| Experience replay, target networks | Phase 1b |
| Overestimation bias | Phase 1c |
| Policy gradient theorem | Phase 2 |
| Advantage estimation, GAE, baselines | Phase 2 |
| Trust regions, clipped objectives | Phase 2 |
| Continuous control, entropy regularization | Phase 3 |
| Sample efficiency as a measured quantity | Phase 3 |
| Reward design, shaping, PBRS, reward hacking | Phase 0 onward — continuous, not a phase |
| Credit assignment over long horizons | Phase 4c |
| Constrained MDPs, cost signals | Phase 4d |
| Curriculum learning | Phase 4, structurally |
| Generalization | Phase 4f |
| Model-based RL, world models | Phase 5, optional |

Reward design deliberately does not sit in one phase. It starts in Phase 0 with the trajectory-pair
test against highway-env's own default reward, and recurs every time the environment changes.

---

## On scheduling

No wall-clock estimates until weekly availability is known — an estimate built on a guessed
time budget is worse than no estimate, because it will be treated as a commitment.

What can be said from `research/06` and `research/12`: Phase 0 and Phase 1b are days-to-weeks of
part-time work, dominated by debugging rather than compute. Phase 4 is where time genuinely
expands, because each capability needs its own reward iteration. Every documented solo success in
`research/12` stayed narrow in scope, and none reached the equivalent of a full city.
