# Reward Function Design for Driving

Reward design is explicitly called out across multiple surveys as **"still very much an open problem"** for RL-based driving. This file surveys what's actually been tried, documents a genuinely alarming published finding about how bad this can get in practice, and — because you have direct hard-won experience with reward hacking and potential-based reward shaping from the SO-101 arm project — draws out exactly where that experience does and doesn't transfer.

## The standard reward components

A dedicated review (["A Review of Reward Functions for RL in the context of Autonomous Driving"](https://arxiv.org/pdf/2405.01440)) categorizes almost all published driving reward functions into four recurring components:

1. **Safety** — collision avoidance/crash prevention (almost always the largest-magnitude single term, usually a large negative terminal penalty).
2. **Comfort** — passenger/vehicle smoothness, operationalized via acceleration and especially **jerk** (rate of change of acceleration) penalties in both longitudinal and lateral directions.
3. **Progress** — speed maintenance and forward movement toward a destination.
4. **Traffic-rule compliance** — lane-keeping, speed-limit adherence, stop-sign/red-light compliance, minimum-headway maintenance.

The same review identifies four **structural** problems with how the field currently combines these:
- **The objectives-aggregation problem**: there's no principled method for combining these four (often competing) objectives — nearly everyone just hand-tunes a linear weighted sum.
- **Context indifference**: fixed weights don't adapt to context (a school zone and a highway arguably warrant different weightings, but almost no published reward function adapts).
- **A standardization gap**: because every paper invents its own formulation, results aren't meaningfully comparable across the literature.
- **Inadequate individual-objective formulation**: even single terms (e.g., "comfort") often lack a rigorous, agreed mathematical definition.

Their proposed fixes — a reward *validation* framework, context-aware structured rewards, and explicit conflict-resolution mechanisms — are aspirational rather than standard practice yet, which is itself informative: this genuinely is an unsolved design problem, not something you're missing an established playbook for.

Sources: [Review of Reward Functions for RL in AD (2405.01440)](https://arxiv.org/pdf/2405.01440), [Comprehensive Review of RL for AD in CARLA (2509.08221)](https://arxiv.org/abs/2509.08221).

## Comfort/jerk specifically — there's an actual standard to borrow

Unlike the other three components, comfort has a pre-existing, principled external standard you can borrow instead of inventing your own threshold: **ISO 2631**, a vibration/motion-comfort standard (originally from structural/industrial engineering) that classifies human comfort/discomfort/motion-sickness response to lateral and longitudinal motion in the 0.5–10 Hz frequency band relevant to car ride quality. Multiple recent AV ride-comfort papers adopt ISO 2631 wholesale (combined with jerk-threshold detection) rather than inventing an ad hoc "penalize jerk above some made-up number" term — worth doing the same rather than reinventing this component from scratch.

One documented failure mode specific to comfort terms: a study that focused purely on minimizing acceleration/jerk to optimize ride comfort produced **overly cautious driving and poor navigation efficiency in urban environments** — a direct illustration that any single-objective optimization in this domain tends to produce a degenerate, if locally "correct," policy, reinforcing why the field insists on multi-objective combination despite lacking a principled way to do it.

Sources: [Standards for passenger comfort in AVs (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S0003687022002046), [Ride comfort assessment via ISO 2631 + Monte Carlo](https://onlinelibrary.wiley.com/doi/full/10.1111/mice.12787).

## The documented failure modes — and why this should worry you specifically

["Reward (Mis)design for Autonomous Driving"](https://arxiv.org/abs/2104.13906) (Knox, Allievi, Banzhaf, Schmid, Stone) is the single most important paper for this file. It audits real published autonomous-driving reward functions and finds concrete, specific problems:

- **The circular-motion/looping exploit**: a reward function that gives credit for incremental "progress" without ensuring eventual task completion lets an agent **move in circles to repeatedly collect the progress reward, never actually reaching the goal.** This is the exact "absolute/incremental reward exploited by dwelling" pattern — in the arm project, an analogous exploit would be an agent that hovers near a rewarded configuration without completing the task; here it's a car that loops rather than arrives.
- **Lane-centering-penalty side effect**: penalizing deviation from the lane center taught a policy to "always try to remain in the track center" — even in situations where swerving toward the lane edge would have been the objectively safer maneuver (e.g., to avoid a hazard). A locally-sensible shaping term produced a globally wrong preference ordering.
- **Lane-change-penalty side effect**: a term meant to discourage erratic lane-changing also discouraged moving away from unpredictable nearby traffic or pedestrians — exactly when moving away would have been the safe choice.
- **The headline result**: the authors constructed a simple pair of trajectories — one where the car crashes partway through, one where the car simply stays idle forever (never making progress, but never crashing) — and checked which trajectory each of 9 real, published reward functions preferred. **7 of the 9 reward functions ranked the crashing trajectory higher than the safely-idle one.** This is not a subtle numerical quirk; it's most of the field's published reward functions, if actually optimized to convergence, preferring to crash over doing nothing — the exact opposite of the safety priority the reward was nominally designed to encode. The proximate cause in most cases: a small positive per-timestep progress/speed term, summed over many timesteps, can outweigh a one-time terminal collision penalty unless the relative magnitudes and discounting are handled very carefully.
- **The process problem behind all of this**: of the papers that disclosed their reward-design process, **all 8 used manual trial-and-error weight tuning based on observed learned behavior** — which overfits the reward function to one specific algorithm/seed/run and hides exactly these kinds of preference-ordering inversions, since "the agent looked reasonable in my test runs" doesn't imply "the reward function has no exploitable degenerate optimum."

The authors' recommended lessons, worth treating as close to hard rules:
1. **Don't encode *how* to drive (behavioral heuristics) directly as reward terms** — specify the *outcome* you want (safely reach the destination) and let shaping be added in a way that's provably guaranteed not to change what the optimal policy actually is (see potential-based shaping, next section), rather than adding ad hoc heuristic terms whose interaction effects you haven't checked.
2. **Explicitly test candidate reward functions against hand-constructed trajectory pairs before ever training on them** — e.g., literally check "does my reward function rank a safe-but-idle trajectory above a trajectory that crashes?" This is cheap (no training required) and would have caught the failure above immediately.
3. **Don't rely on trial-and-error tuning against observed training behavior as your only validation method** — it hides exactly this class of bug.

Sources: [Reward (Mis)design for Autonomous Driving (2104.13906)](https://arxiv.org/abs/2104.13906).

## Where your potential-based shaping experience transfers directly

**Potential-based reward shaping (PBRS)**, from Ng, Harada & Russell's 1999 "Policy Invariance Under Reward Transformations," is exactly the tool that would have prevented the crash-preferred-over-idle inversion above, and it's a tool you already have hands-on validated experience applying (in the SO-101 gripper-closing shaping work).

The theorem: given any potential function Φ over states, adding a shaping term
```
F(s, s') = γ·Φ(s') − Φ(s)
```
to the reward is **guaranteed not to change the set of optimal policies** of the underlying MDP — you get to inject dense, per-step guidance without risking the kind of exploitable distortion documented above, *provided* the core (unshaped) reward itself is already correctly specified. This is the crucial caveat: PBRS guarantees policy invariance **relative to whatever the true/core reward already says is optimal** — it doesn't fix a core reward that's already broken (e.g., a core reward that itself prefers crashing over idling). So the two lessons compose: first get the sparse, terminal, "outcome-only" core reward right and validated against hand-built trajectory pairs (Knox et al.'s lesson), *then* use PBRS to densify it for faster learning, rather than reaching for ad hoc dense per-step terms as your first move.

A concrete driving-specific instantiation: define the core reward as sparse and terminal-only — e.g. `+1` on safely reaching the destination, `-1` on collision, `0` otherwise (with a timeout also scored as `0`, deliberately *not* negative, so that "did nothing" never scores worse than "crashed") — and define a potential function `Φ(s)` as, say, negative remaining route distance to the goal. The resulting shaped reward gives dense per-step feedback ("you got closer to the goal") while being mathematically guaranteed to still prefer the safe-idle trajectory over the crashing one, because that preference is baked into the untouched core reward and PBRS cannot alter it. This is the dense-shaping-without-corrupting-the-optimum pattern you already validated works in the gripper-closing shaping — it transfers to driving essentially without modification, just with a different potential function (route progress instead of gripper-closing progress).

Sources: [Policy Invariance Under Reward Transformations (Ng, Harada, Russell)](https://www.andrewng.org/publications/policy-invariance-under-reward-transformations-theory-and-application-to-reward-shaping/), [Potential-Based Shaping and Q-Value Initialization](https://jair.org/index.php/jair/article/download/10338/24713/19085).

## Sparse milestone bonuses vs. dense shaping — the tradeoff, and how model-based RL changes it

- **Dense per-timestep shaping** (speed reward, lane-centering, jerk penalty, all summed every step) is what the large majority of published driving reward functions actually use, because it trains faster. It is also, per Knox et al. above, **exactly the pattern responsible for the exploitable inversions documented**, because dense terms compound over long episodes in ways that are easy to get wrong and hard to notice without deliberate testing.
- **Sparse milestone/terminal-only rewards** (survive to destination = `+1`, collision = `-1`, everything else = `0`) are far more robust to reward hacking — there's no dense per-step term to game — but they reintroduce the **long-horizon credit-assignment problem** in a serious way: a 60-second drive with a single terminal signal is a genuinely hard temporal-credit-assignment problem for model-free RL (see `05_challenges_and_pitfalls.md`).
- Two techniques independently soften this tradeoff, and are worth considering *together*: (1) **potential-based shaping**, which densifies a sparse core reward without corrupting it, as above; and (2) **model-based/world-model RL** (TD-MPC2, Dreamer-family — see `02_rl_algorithms.md`), where the effective "many more updates per real environment step" from imagined rollouts substantially eases the practical burden of sparse rewards, since the model gets to internally simulate many more attempts at reaching the sparse milestone than the raw environment-step count would suggest. Combining both — a sparse, validated, outcome-only core reward, densified via PBRS, learned inside a world model — is close to what state-of-the-art systems like Think2Drive are actually doing, even if not always framed in exactly these terms.

## Traffic-rule compliance as a reward term vs. as a constraint

Simple traffic-rule terms (stay in lane, respect speed limit, maintain minimum headway) are usually implemented as straightforward reward penalties. But more nuanced rules ("maintain sufficient speed to avoid impeding traffic flow") require real contextual information that a scalar penalty struggles to encode, and real driving involves continuous *degrees* of rule adherence that a simple boolean check doesn't capture well. Recent work addresses this by formalizing traffic rules in **Linear Temporal Logic (LTL)**, giving an automatically and unambiguously evaluable specification of rule violation, and by treating the resulting compliance signal as a **constraint** (in a constrained-MDP/safe-RL sense — see `05_challenges_and_pitfalls.md`) rather than folding it into the scalar reward at all. This is a genuinely cleaner design than adding "traffic rule violation" as just another weighted penalty term subject to the same aggregation problems described above, and worth considering directly for any traffic-rule component you build in.

Sources: [Predictive Traffic Rule Compliance using RL](https://arxiv.org/pdf/2503.22925), [Provable Traffic Rule Compliance in Safe RL](https://arxiv.org/pdf/2402.08502).

## Practical checklist distilled from this section

1. Write the core reward as sparse and outcome-only first (goal reached / collision / timeout), and explicitly verify it prefers "safe and slow/idle" over "fast but crashes," by hand, on constructed trajectory pairs, before training anything.
2. Add density via potential-based shaping (`F(s,s') = γΦ(s') − Φ(s)`) rather than ad hoc per-step heuristic terms, so densification is provably policy-invariant relative to the validated core reward.
3. Treat comfort via the existing ISO 2631 jerk/acceleration framework rather than an invented threshold.
4. Treat hard traffic-rule compliance as a constraint (safe-RL/CMDP cost) rather than folding it into the same scalar reward as everything else, especially for rules with real contextual nuance.
5. Don't validate reward design solely by eyeballing training-run behavior — that's precisely the process every one of the 8 papers that disclosed it used, and precisely what let the crash-preferred-over-idle inversion ship in 7 of 9 published systems.
