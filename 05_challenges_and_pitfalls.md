# Known Pitfalls and Open Challenges

## Reward hacking (cross-reference)

Covered in depth in `04_reward_design.md` — including the specific published finding that 7 of 9 real driving reward functions preferred a mid-drive crash over safe idling. Framed more generally, reward hacking is a special case of **Goodhart's Law**: an agent optimizing an imperfect proxy for what you actually want can achieve high proxy reward while producing poor true outcomes. Lilian Weng's widely-cited survey distinguishes **specification gaming** (exploiting ambiguities in the reward design, e.g. maximizing average velocity as a proxy for "minimize commute time" in a traffic-control setting, producing clearly undesirable but highly-rewarded behavior) from **reward tampering/wireheading** (the agent directly interferes with the reward-generating process itself — a more exotic failure mode, more relevant to advanced/agentic systems than to a car-driving RL agent, but conceptually part of the same family). Controlled studies of mitigation strategies (reward engineering, adversarial training, hybrid model checks) report reducing hacking incidence by up to ~54.6% — meaningful, but nowhere near a complete fix, reinforcing that reward-hacking risk should be actively designed against (per the checklist in `04`) rather than assumed away.

Sources: [Reward Hacking in RL (Lilian Weng)](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/), [Robust Optimization for Mitigating Reward Hacking with Correlated Proxies](https://arxiv.org/html/2604.12086).

## The sim-to-real gap

Multiple distinct gaps compound rather than being one single problem:

- **Appearance/domain gap**: simulated camera imagery looks different from real camera imagery (lighting, texture, noise characteristics) even when the underlying geometry matches. Dedicated tools exist specifically to shrink this — e.g. [CARLA2Real](https://arxiv.org/pdf/2410.18238), which post-processes CARLA renders to look more photorealistic, and hybrid approaches combining 3D Gaussian Splatting with LiDAR-derived meshes to get both visual and structural realism simultaneously.
- **Physics/dynamics gap**: most driving simulators use simplified vehicle dynamics (kinematic or simple bicycle models); BeamNG's soft-body physics is a notable exception, at the cost of being a less RL-training-oriented platform (see `01_simulation_platforms.md`).
- **Sensor gap**: real cameras/LiDAR are noisier and sparser than their simulated counterparts by default.
- **Control-loop/actuation gap**: real actuation has delay and imprecision a simulator may not model.
- **Data-driven alternative**: rather than closing these gaps from a game-engine simulator, MIT's **Vista/Vista 2.0** takes the opposite approach — building the simulator directly from real recorded sensor data (video, LiDAR, event-camera) via view synthesis, so the "sim" is photorealistic by construction because it *is* reconstructed real data, at the cost of needing real sensor data up front rather than authoring an environment by hand.
- **Multi-agent sim-to-real is strictly harder**: a dedicated study on transferring multi-agent RL policies to real hardware identified three additional, distinct gaps beyond the single-agent case — a **control-architecture gap** (synchronization of multiple agents' actions differs between sim and real), an **observation gap** (limited/different perception fidelity in real scaled environments), and a **communication gap** (real inter-agent communication is more limited and less reliable than simulated).

Sources: [Sim2Real-AD framework](https://arxiv.org/html/2604.03497), [Transferring MARL policies via Sim-to-Real (2203.11653)](https://arxiv.org/pdf/2203.11653), [MIT Vista release](https://news.mit.edu/2022/researchers-release-open-source-photorealistic-simulator-autonomous-driving-0621).

**A scoping note specific to your situation**: this entire category of problem is almost entirely avoidable simply by staying in simulation — nothing here matters until you decide to deploy on real hardware (Duckiebot, F1TENTH car, Donkeycar). Given this is explicitly a from-scratch, early-stage project, it's worth treating "do we ever deploy to real hardware" as a genuinely open, deferrable question rather than a default assumption — see `06_recommendations.md`.

## The safety/exploration tension

RL fundamentally learns through trial and error, which means an agent must, at some point during training, try actions it doesn't yet know are unsafe — a structurally uncomfortable property for a domain where "unsafe" can mean a simulated (or real) collision. The standard formalization is the **Constrained Markov Decision Process (CMDP)**: extend the ordinary MDP with an explicit cost function and a threshold, so the agent is trained to maximize reward *subject to* keeping expected cumulative cost below a bound, rather than folding safety into the same scalar as everything else (directly connecting to the reward-aggregation problem in `04_reward_design.md`). Concretely, MetaDrive's built-in `cost = 1` on collision-or-off-road, tracked as a separate logged signal from reward, is exactly this pattern implemented in a real platform.

Mitigation approaches actually used in the literature:
- **Action masking / safe action sets**: identify unsafe actions during exploration and project the policy's choice onto a safe subset, reducing (sometimes to zero) constraint violations during training itself.
- **Control-barrier-function-style state-wise constraints**: encode a safety margin directly into the learning process at every state, not just in expectation over an episode.
- **Dual short-term + long-term constraints**: separately bound immediate-state risk and cumulative/long-horizon risk, since a policy can satisfy one while violating the other.
- **Safe DAgger's learned "safety policy"** (see `02_rl_algorithms.md`): predicts whether the primary policy is likely to deviate dangerously *before* acting, deferring to a reference/expert policy in exactly those situations, minimizing the number of genuinely risky actions ever executed during learning.

Sources: [Survey of Constraint Formulations in Safe RL](https://arxiv.org/pdf/2402.02025), [Long and Short-Term Constraints Driven Safe RL for AD](https://arxiv.org/abs/2403.18209).

**Scoping note**: as above, this tension is almost fully neutralized by staying in simulation for training — it becomes a first-class, unavoidable design problem only the moment real-hardware deployment enters the picture.

## Long-horizon credit assignment

A full drive across even a modest town easily spans hundreds to thousands of low-level control steps between an action and its eventual consequence — e.g., an overly optimistic gap-acceptance decision at an intersection forty steps ago causing a near-miss now. This is a specific instance of the general **sparse-reward, long-horizon credit assignment problem**: determining which of many past actions in a trajectory actually caused a later outcome becomes fundamentally harder as the trajectory lengthens, and early decisions can have consequences that only manifest much later, making the causal chain increasingly indirect for a model-free learner relying on a single (or sparse) terminal reward signal.

Mitigations directly relevant here, cross-referencing earlier files:
- **Potential-based reward shaping** (`04_reward_design.md`) — densifies a sparse core reward without corrupting its optimum, directly easing the credit-assignment burden.
- **Hierarchical RL / temporal abstraction** (`02_rl_algorithms.md`) — options/maneuvers collapse the effective decision horizon a top-level policy has to reason over, since each high-level choice spans many low-level steps.
- **Model-based/world-model planning** (`02_rl_algorithms.md`) — imagined rollouts inside a learned world model let the agent effectively evaluate consequences further ahead per real environment step than raw step-count would suggest, directly attacking the same problem from a different angle.
- Plain model-free mitigations like n-step or eligibility-trace returns remain available as a baseline, if a less powerful, response.

Sources: [Hierarchical SAC for Sparse-Reward Long-Horizon RL](https://arxiv.org/html/2607.23726v1).

## Generalization across intersections, scenarios, and weather

This is one of the field's most persistently unsolved problems, and the empirical evidence is fairly consistent across studies:

- **MetaDrive's own generalization experiments** found that increasing the **diversity and size of the procedurally-generated training set** is the most reliable lever for improving generalization to unseen scenes — not any particular algorithmic trick.
- **Fail2Drive**, a benchmark specifically built to evaluate generalization in CARLA under novel scenarios/unseen assets, found an average **success-rate drop of 22.8%** across multiple state-of-the-art models when moved out-of-distribution — a large, consistent degradation, not a rare edge case.
- **Driver Dojo**, a dedicated generalization benchmark, uses a configurable catalog of randomized scenario generators (road layout, traffic patterns, observation types, vehicle dynamics models) specifically because no single fixed benchmark reliably stresses generalization.
- **Weather/lighting** is its own axis, studied directly in CARLA: agents trained exclusively under clear-noon conditions show measurable performance degradation when evaluated under rain, night, or slippery-surface conditions — a reminder that "the same town, different weather" is already an out-of-distribution generalization test, not a trivial variation.
- A pointed finding from large-scale self-play research: **self-play RL agents trained at scale can fail to generalize to unseen traffic agents** even when they generalize well to unseen maps — generalization along the "other agents' behavior" axis is not automatically implied by generalization along the "map/topology" axis.

The honest state of the field: there is no single dominant trick that "solves" generalization — training-distribution diversity remains the most empirically validated lever, which is a real argument for procedural generation (MetaDrive-style) over a small number of hand-built towns if generalization matters to your goals.

Sources: [Fail2Drive benchmark](https://arxiv.org/html/2604.08535v1), [Driver Dojo benchmark](https://arxiv.org/pdf/2207.11432), [Beyond Self-Play and Scale (2605.10034)](https://arxiv.org/html/2605.10034v1).

## Long-tail / corner-case coverage

Real-world safety-critical scenarios follow a **long-tail distribution** — individually rare but collectively important, and by definition underrepresented in any naturally-sampled training distribution. This motivates a distinct sub-literature on **automatic/adversarial scenario generation**: game-theoretic adversarial scene generation, diffusion-based corner-case traffic/image generation, and RL-based scenario editing, all aimed at manufacturing rare-but-critical scenarios deliberately rather than waiting to encounter them by chance. This isn't purely academic — the **CARLA Leaderboard 2.0** itself added 39 curated real-world corner cases specifically because Leaderboard 1.0's 10 base scenarios weren't stressing this enough.

Sources: [CC-SGG: Corner Case Scenario Generation](https://arxiv.org/pdf/2309.09844), [Safety-Critical Scenario Generation via RL-Based Editing](https://arxiv.org/pdf/2306.14131).

## Standard evaluation metrics used in the field

Rather than inventing bespoke metrics, the field has converged on a small number of standardized suites worth adopting directly:

- **CARLA Leaderboard "Driving Score"**: `route completion % × infraction penalty` — multiplicative, meaning a single severe infraction can crater an otherwise fully-completed route's score. Infraction coefficients are explicit and severity-graded: collision with a pedestrian (0.70× — reducing score by that fraction), collision with another vehicle (0.70×), collision with a static object (0.60×), running a red light (0.40×), failing to yield to an emergency vehicle (0.40×), running a stop sign (0.25×). Agents start at a perfect base score of 1.0, and infractions multiplicatively reduce it toward a floor of 0.
- **nuPlan's Closed-Loop Score (CLS)**: a weighted aggregate of progress (fraction of route completed), safety violations (collision, off-road departure, wrong-direction driving — often a **hard zero-out gate** on the whole scenario's score rather than a soft penalty), comfort (jerk/acceleration/yaw-rate bounds, also can hard-zero the score if violated), speed-limit adherence, and minimum time-to-collision. The extension **nuPlan-R** adds **Success Rate** (% of scenarios with a nonzero CLS — i.e., no catastrophic failure) and **All-Core Pass Rate** (% of scenarios where safety, comfort, and efficiency sub-scores all clear 0.5) as additional robustness-oriented metrics.
- **WOSAC (Waymo Open Sim Agents Challenge)**: uniquely **realism-focused rather than task-success-focused** — it doesn't ask "did the ego car succeed," it asks "do the *other* simulated agents' behavior distributions match real logged human distributions," via negative log-likelihood across kinematic (speed, acceleration, angular velocity/acceleration), interactive (collision indication, distance-to-nearest-object, time-to-collision), and map-based (off-road indication, distance-to-road-edge) metric groups. Directly relevant if your "other cars" need to look like real traffic, not merely avoid collisions.
- **MetaDrive/safe-RL convention**: track raw success rate, collision rate, and off-road rate, plus a first-class separate **cost** metric (cumulative constraint violations) rather than folding safety into the reward number — precisely because reward-only optimization can mask safety regressions, tying directly back to the reward-hacking discussion in `04_reward_design.md`.
- **Comfort specifically**: ISO 2631 (see `04_reward_design.md`) as the standardized alternative to an invented jerk threshold.

The practical recommendation: **adopt an existing standardized metric suite rather than inventing a bespoke one**, both for comparability with published work and because these suites already encode the community's accumulated lessons about reward/metric-hacking resistance (multiplicative infraction penalties, hard safety/comfort gates rather than soft averaged penalties).

Sources: [CARLA Leaderboard evaluation criteria v2.0](https://leaderboard.carla.org/evaluation_v2_0/), [nuPlan-R (2511.10403)](https://arxiv.org/pdf/2511.10403), [Waymo Open Sim Agents Challenge](https://ml4ad.github.io/files/papers2023/The%20Waymo%20Open%20Sim%20Agents%20Challenge.pdf).
