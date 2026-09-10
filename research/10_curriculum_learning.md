# Curriculum Learning for Driving RL

Curriculum learning — training on a deliberately-ordered sequence of tasks of increasing difficulty rather than the full target task from the start — has a specific, quantified track record in driving RL that goes beyond the general intuition that "easier first" should help. This file surveys concrete curriculum designs actually used in the literature, the real quantitative evidence for (and, importantly, a documented case against) it actually working, and how it composes with material already covered in `02` and `05`.

## What curriculum learning means here, precisely

Following the standard framing (Bengio et al. 2009's original formulation; Narvekar et al.'s 2020 survey of curriculum RL specifically), a curriculum is a **sequence of tasks or environment configurations** (a "syllabus") that a learner progresses through, ideally transferring knowledge from earlier, easier tasks to make later, harder tasks more tractable than they'd be from a cold start. Two broad design choices recur across the driving-specific papers below:
- **Fixed/hand-designed curricula**: the practitioner manually specifies the stage order in advance (e.g., "empty road → light traffic → dense traffic → pedestrians").
- **Automatic/adaptive curricula**: the system itself decides what to train on next, based on some measure of the current policy's competence or learning progress — closing the loop so the curriculum adapts to the specific learner rather than following a fixed script.

## Concrete curriculum designs actually used in driving RL papers

### Reinforced Curriculum Learning for Autonomous Driving in CARLA (ICIP 2021)
[`Luca96/carla-driving-rl-agent`](https://github.com/Luca96/carla-driving-rl-agent) (already flagged in `07` for its architecture) organizes PPO training into successive stages of increasing difficulty, layered under a multi-modal encoder (ShuffleNetV2 for images, feed-forward branches for road/vehicle/navigation features, GRU for temporal aggregation). The project frames curriculum staging as core to making the full CARLA task tractable at all for a PPO agent trained from a from-scratch policy, rather than as an optional refinement — consistent with the broader finding (below) that curriculum matters more, not less, as task complexity rises.

### CuRLA: two-fold curriculum (2025)
CuRLA's specific contribution is doing curriculum along **two axes simultaneously**: progressively increasing environment difficulty (the usual axis) *and* progressively introducing a collision penalty into the reward function itself (i.e., the reward function's own complexity is part of the curriculum, not just the environment's). This is a genuinely different lever than the other papers here — most curricula hold the reward fixed and vary only the environment; CuRLA varies both, motivated explicitly by promoting safety once the agent has already learned basic competence rather than penalizing collisions the agent doesn't yet have the skill to avoid.

### Investigating the Value of Curriculum RL Under Diverse Road and Weather Conditions (2021) — the paper with the clearest numbers, including a genuine negative result
This is the single most quantitatively useful paper found for the "does this actually help" question. It systematically tests **5 curriculum sequences**, each combining road-geometry stages (straight → U-turn → race-track) with weather-condition stages (clear → rainy → snowy), against a non-curriculum baseline trained directly on each of the 9 possible (road × weather) combinations from scratch. Concretely:

| Curriculum scenario | Sequence | Final-task reward | Non-curriculum baseline on same final task | Improvement |
|---|---|---|---|---|
| Scenario 4 | Straight(Clear) → Race Track(Rainy) → Race Track(Snowy) | ~1,248,453 | 70,742 | **~17.6× improvement** |
| Scenario 5 | Straight(Clear) → Straight(Snowy) → Race Track(Snowy) | ~597,622 | 70,742 | **~8.4× improvement** |

The paper reports curriculum learning outperforming the non-curriculum baseline in nearly every scenario tested — a real, large, quantified effect, not a marginal one.

**The important caveat, and the reason this paper is worth citing over a purely positive result**: the same paper found that when a *poor-performing intermediate-stage agent* was used as the starting point for the next curriculum stage, the resulting final performance was **worse than the non-curriculum baseline** — i.e., curriculum learning is not free, and a badly-chosen or badly-trained intermediate stage can actively hurt compared to training on the target task directly. This directly parallels a general lesson from the broader curriculum-RL literature (Narvekar et al.): curriculum design quality itself is a hyperparameter, and a bad curriculum is not merely "less good than a good curriculum" — it can be worse than no curriculum at all. Practically, this means: validate each intermediate stage actually reaches competence before advancing, rather than advancing on a fixed schedule regardless of whether the current stage's agent is actually any good.

### Automatic Curriculum Learning for Driving Scenarios (2025)
This is the most sophisticated *automatic* (non-hand-designed) curriculum framework found. It alternates two phases:
- **Exploration**: randomly samples scenario parameters, scores each candidate scenario by a **"positive value loss"** metric (roughly, how much the current policy's value estimates are wrong on that scenario, capturing "this scenario is at the edge of what I can currently handle") and keeps only scenarios exceeding a minimum learning-potential threshold in a scenario buffer.
- **Exploitation**: samples from that buffer weighted by a combination of learning potential and "staleness" (so old scenarios don't get abandoned once they stop looking maximally difficult), then an "editor" mutates the selected scenario via one of three operations — modify the agent's goal, adjust actor attributes, or add/remove actors (i.e., the traffic-density axis from `03`).

Compared against both **fixed-scenario training** and **domain randomization** (fully random scenario generation with no structured progression), the reported numbers are concrete:
- **+9% success rate at low traffic density, +21% at high traffic density** on held-out test scenarios, versus domain randomization.
- **Collision rate reduced from 29% (domain randomization) to 20%** at traffic density 0.5.
- At **1 million training updates and traffic density 0.75**, the automatic-curriculum agent reached **58% success vs. 25% for domain randomization** at the same training budget — i.e., not just a better final policy, but genuinely faster learning per training step, which is the sample-efficiency claim curriculum learning is supposed to deliver.

This result is particularly informative because the comparison isn't "curriculum vs. nothing" but "structured curriculum vs. domain randomization" — domain randomization is itself already a reasonable diversity-injection baseline (and the mechanism MetaDrive's own generalization results, cited in `05`, credit for improving generalization), so beating it specifically is a stronger claim than merely beating a single fixed scenario.

### Related work worth knowing about but not needing full separate treatment
- **Diverse and Adaptive Behavior Curriculum** (2025) applies a student-teacher framework specifically in a multi-agent RL setting, reporting that students trained under an automatic curriculum generalize more effectively — safer navigation in rule-based traffic, higher route progress, higher average velocity — than students trained without one. Relevant if your "other cars" are themselves learning agents (per `02`'s multi-agent RL section) rather than scripted.
- **Scenario-Based Curriculum Generation for Multi-Agent Autonomous Driving** (2024) tackles the same student-teacher automatic-curriculum idea specifically for the multi-agent case, generating scenarios rather than single-agent task sequences.

## What gets curriculum'd, in practice: a taxonomy

Synthesizing across the papers above and `03`'s design-space axes, curriculum designs in this literature vary one or more of:
- **Road complexity**: straight segments → curves/U-turns → intersections → full town (the most common single axis).
- **Traffic density**: zero cars → light traffic → dense traffic (the axis the automatic-curriculum paper's actor-count mutation directly targets).
- **Weather/lighting**: clear → rainy → snowy/night (the axis the "diverse road and weather conditions" paper isolates).
- **Reward complexity**: progress-only reward first, safety/comfort terms introduced later (CuRLA's distinguishing contribution — curriculum-ing the *reward function itself*, not just the environment).
- **Pedestrian presence**: absent → present (implied by, though not the central focus of, the automatic-curriculum actor-mutation mechanism).

## The evidence, synthesized

**For**: multiple independent papers report large, concrete, positive effects — up to 17.6× reward improvement in the best-case curriculum ordering (2021 weather/road paper), and clear sample-efficiency and generalization gains over both fixed-scenario training and domain randomization specifically (2025 automatic-curriculum paper: +9-21% success rate, ~30% relative collision-rate reduction, roughly double the success rate at matched training budget). This is a considerably stronger empirical basis than "curriculum learning seems intuitively reasonable" — there are real numbers behind it in this specific domain.

**Against / caveats**: the same 2021 paper that produced the headline positive numbers also produced the field's clearest documented negative result — a curriculum stage built on a poorly-trained intermediate agent performed *worse* than skipping curriculum entirely. The general curriculum-RL literature's standard caveats apply directly: a curriculum can induce something like catastrophic forgetting of earlier-stage competence if later stages diverge too far, and "increasing difficulty" is not self-evidently well-ordered — the *ordering itself* is a design decision that needs validation, not an assumption. None of the papers surveyed here report a case where a *well-validated* curriculum (each stage confirmed competent before advancing) underperformed a non-curriculum baseline — the negative result specifically traces to skipping that validation step, which is actionable rather than a fundamental limit on the technique.

## Practical recommendation, tying to `06`'s tiers

For **Tier 1** (`06`): curriculum learning is likely unnecessary — the whole point of Tier 1 is a task simple enough to learn directly, and introducing curriculum machinery adds complexity to a stage meant for pipeline validation.

For **Tier 2**: this is where curriculum learning starts earning its complexity budget — a hand-designed fixed curriculum (road complexity, then traffic density, then weather, roughly following the taxonomy above) is a reasonable, low-engineering-cost starting point, with the explicit discipline of **validating each stage's competence before advancing** (a simple held-out success-rate check against a threshold, not just "trained for N steps") to avoid the exact failure mode the 2021 paper documented.

For **Tier 3**: the automatic-curriculum approach (learning-potential-driven scenario selection, as in the 2025 automatic-curriculum paper) becomes worth the extra engineering investment, precisely because at full-city scale the space of possible scenarios is too large to hand-author a good fixed ordering — and this is also the tier where CuRLA's "curriculum the reward function too" idea composes naturally with `04`'s advice to start with a validated sparse core reward and add complexity deliberately rather than all at once.

Sources: [CuRLA (2501.04982)](https://arxiv.org/abs/2501.04982), [Reinforced Curriculum Learning for AD in CARLA / Luca96 repo](https://github.com/Luca96/carla-driving-rl-agent), [Investigating Value of Curriculum RL in AD (2103.07903)](https://arxiv.org/abs/2103.07903), [Automatic Curriculum Learning for Driving Scenarios (2505.08264)](https://arxiv.org/html/2505.08264), [Diverse and Adaptive Behavior Curriculum (2507.19146)](https://arxiv.org/html/2507.19146), [Scenario-Based Curriculum Generation for Multi-Agent AD (2403.17805)](https://arxiv.org/pdf/2403.17805).
