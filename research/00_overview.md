# Self-Driving Car RL: Research Overview

> **Scope note added 2026-09-10.** This dossier was compiled *before* the project's scope was
> set, and assumes the objective is a good driving agent. The actual objective is **learning RL**,
> with the driving task as the teaching device, in simulation only, with algorithms implemented
> by hand. Several recommendations here — most notably "use Stable-Baselines3 rather than
> implementing algorithms yourself" — are therefore wrong for this project.
>
> The facts remain accurate and this dossier is still the reference for them. But read
> [`../docs/00_project_charter.md`](../docs/00_project_charter.md) first; it tabulates exactly
> where the two diverge. The current project plan lives in [`../docs/`](../docs/).

This is a research dossier for a brand-new side project: training a self-driving car agent with reinforcement learning inside a simulated city/urban environment. No design decisions have been made yet — this folder is meant to give you the full lay of the land (simulators, algorithms, environment-design levers, reward design, known pitfalls, and a tiered recommendation) so you can make those decisions deliberately rather than by default.

This is a **research and planning phase only**. Nothing here is code or a final decision — it's the map, not the territory.

## How this folder is organized

| File | Contents |
|---|---|
| `01_simulation_platforms.md` | Survey of ~20 simulation platforms actually used for driving RL — what they simulate, RL-friendliness, maturity, licensing, hardware needs |
| `02_rl_algorithms.md` | PPO, SAC, DQN/Rainbow, model-based (Dreamer/TD-MPC2/MuZero), imitation+RL hybrids, offline RL, multi-agent RL, hierarchical RL — what the driving-RL literature actually favors and why |
| `03_environment_design_space.md` | The actual levers: road topology, traffic-control elements, other agents (scripted vs. learning, pedestrians), sensing modalities, action space design — presented as tradeoffs, not a recommendation |
| `04_reward_design.md` | What reward terms are actually used, documented failure modes (including a genuinely alarming published finding), and how potential-based shaping connects to this |
| `05_challenges_and_pitfalls.md` | Sim-to-real gap, the safety/exploration tension, long-horizon credit assignment, generalization, and the standard evaluation metrics the field actually uses |
| `06_recommendations.md` | Three project-scope tiers (minimal / medium / full-city) with concrete platform + algorithm + design-choice bundles and realistic time/compute estimates for each |
| `07_reference_implementations.md` | Specific GitHub repos and tutorials to study or fork, with maintenance status checked directly and an honest quality verdict on each |
| `08_training_frameworks_and_tdmpc2_gap.md` | SB3 / RLlib / CleanRL / Tianshou compared, plus an itemized engineering breakdown of what a real TD-MPC2 driving integration would require |
| `09_traffic_behavior_modeling.md` | IDM, MOBIL, and the social force model at implementation detail — actual formulas, actual default constants, and which simulators ship each natively |
| `10_curriculum_learning.md` | Concrete curriculum designs used in driving RL, the quantified evidence they work, and the documented case where a bad curriculum was worse than none |
| `11_real_world_datasets.md` | WOMD, nuScenes, nuPlan, Argoverse 2, comma2k19 — access friction, licensing, and the three patterns for combining logged data with RL |
| `12_small_scale_case_studies.md` | What solo developers and small teams have actually shipped, as a counterweight to the lab-scale focus of everything above |

## The single biggest thing this research surfaced

**The field's practice does not match the "textbook RL" mental model.** Across CARLA, MetaDrive, Waymax, and industrial systems (Apollo), the dominant real-world pattern is *not* "define an MDP, run PPO from scratch." It's:

1. Model-free RL (PPO/SAC/DQN and variants) still dominates by volume of publications — one CARLA-focused review found **>80% of studies use DQN, PPO, or SAC** — but
2. almost every *serious* system either warm-starts from imitation learning (behavior cloning on logged/expert driving) before RL, uses offline RL on logged datasets instead of live exploration, or (increasingly, since ~2023) uses a **world model** (Dreamer-family, TD-MPC-style) so that most of the learning happens on cheap imagined rollouts rather than expensive/risky real environment steps.
3. Reward function design is explicitly called out in multiple surveys as "still very much an open problem" — and one detailed audit of published autonomous-driving reward functions found that **7 of 9 real published reward functions would rather rank a mid-drive crash above idling safely forever**, once you actually compute out their preference ordering. This is not a hypothetical concern.

That third point connects directly to something you already have hands-on scars from in the SO-101 arm project: reward hacking via exploitable per-step terms, and the value of potential-based shaping (Ng, Harada & Russell, 1999) as the principled fix. Section `04` draws that connection out explicitly, because it is one of the few places in this whole research area where your existing experience transfers almost without modification.

## A rough mental map of the design space

Every design decision below is really a dial on a small number of independent axes, and most of the platforms in `01` are really just *pre-committed bundles* of choices along these axes:

- **Fidelity**: privileged ground-truth state → BEV/occupancy raster → semantic segmentation → raw RGB/LiDAR. Higher fidelity = more realistic, more sim-to-real relevant, much slower/harder to learn from.
- **World size/topology**: single loop → grid → small hand-built town → real city (OpenStreetMap-derived). Bigger = more generalization pressure, more compute, more traffic-control complexity.
- **Other-agent fidelity**: scripted rule-based (IDM/Traffic-Manager autopilot) → logged/replayed real trajectories → learned reactive agents (self-play or offline-learned). Each step up is more realistic and more expensive/complicated.
- **Action abstraction**: discrete high-level maneuvers → continuous throttle/steer/brake → hierarchical (both, at two timescales).
- **Learning paradigm**: RL-from-scratch → RL warm-started by imitation → offline RL on logged data → model-based/world-model RL.

Read `03_environment_design_space.md` for the full treatment of these tradeoffs, and `06_recommendations.md` for how they bundle together sensibly at three different project-scope tiers.

## Note on sourcing

This research was compiled from ~40 targeted searches across academic surveys (arXiv), simulator documentation/GitHub repos, and platform release notes, current as of September 2026. Where a specific paper or system is named, it's named because it was directly found and read (or its abstract/content fetched), not inferred. Links are given inline in each file so you can go straight to primary sources for anything that looks interesting.
