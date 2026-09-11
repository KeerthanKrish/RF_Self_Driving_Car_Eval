# Project Charter

**Status**: active
**Last updated**: 2026-09-10

## The goal

Learn reinforcement learning — fundamentals, methodology, and implementation — by building a
simulated self-driving agent incrementally. The driving task is the vehicle for the learning,
not the objective in itself.

The end state is a single project that demonstrates the concepts learned along the way, built
up capability by capability, where each capability added to the environment is chosen because
it forces the next RL concept into play.

## Why this framing matters

The research dossier in `research/` was compiled before this scope was set. It is written as
though the objective is a *good driving agent*, and its recommendations optimize for that. They
are still accurate on the facts, but several of its defaults are wrong for this project, and the
divergence is worth stating explicitly so the dossier is read correctly rather than followed
literally.

| Dossier position | Why it changes here |
|---|---|
| "Nobody rewrites an RL algorithm from scratch — use Stable-Baselines3" (`research/12`) | Correct for shipping, wrong for learning. Calling `model.learn()` teaches environment and reward engineering while leaving the algorithm a black box. Here the algorithms get written by hand, and SB3 becomes the **correctness oracle** rather than the implementation. |
| Sim-to-real gap treated as a major concern (`research/05`) | Out of scope. This project is simulation-only, permanently. An entire chapter of the dossier is inert. |
| Privileged ground-truth state is "unrealistic" (`research/03`) | Its realism penalty is irrelevant here. Low-dimensional privileged state is *ideal* for learning RL fundamentals fast, and is the correct Phase 1 choice. |
| Photorealism, multi-agent realism, pedestrian behavioral fidelity | Optional-to-never. None of these teach an RL concept that a cheaper environment cannot. |
| Reward design (`research/04`) | Becomes **more** central, not less. Reward bugs are the expected primary failure mode. |
| Traffic behavior models (`research/09`) | Becomes more central, because IDM/MOBIL get implemented and tuned by hand rather than consumed as a black box. |
| Curriculum learning (`research/10`) | Becomes structurally central — the incremental capability plan *is* a curriculum, and its documented failure mode applies directly. |

## Non-goals

These are deliberately excluded. Reopening any of them is a decision to record in
`docs/04_decision_log.md`, not a drift to fall into.

- **No real hardware, ever.** No Duckiebot, no Donkey Car, no F1TENTH. Simulation only.
- **No flawless agent.** A policy that mostly works and is well understood beats a policy that
  scores higher and is not.
- **No leaderboard chasing.** Not targeting CARLA Leaderboard or nuPlan benchmark numbers.
- **No photorealism.** Not a perception project unless perception is later chosen as a concept
  to learn deliberately. A 3D physics-backed environment *is* required (D-019) — that is about
  being able to watch a real car on a real road, not about photorealistic sensors.
- **No reimplementation of infrastructure.** Autodiff, physics integration, and rendering are
  taken off the shelf. The from-scratch line is drawn at RL algorithm internals.

## Where "from scratch" starts and stops

| Written by hand | Taken off the shelf |
|---|---|
| Replay buffers, target networks, ε-schedules | PyTorch, autodiff, optimizers |
| Policy gradient, advantage estimation, GAE | Gymnasium API contract |
| DQN, Double/Dueling DQN, REINFORCE, A2C, PPO, SAC | The simulator's kinematics and rendering (initially) |
| Reward functions and their unit tests | Stable-Baselines3 (as reference/oracle only) |
| Evaluation harness and metrics | IDM/MOBIL equations (taken from literature, implemented by hand) |
| Environment capabilities added in later phases | NumPy, logging, plotting |

## Success criteria

This project is successful if, at the end:

1. Each RL algorithm listed above was implemented from scratch and **verified to match a trusted
   reference implementation's learning curve** on the same environment and seeds.
2. Each capability added to the environment can be traced to the RL concept it was added to teach.
3. The reward function passes explicit, automated preference-ordering tests (`research/04`) rather
   than having been tuned by watching training runs.
4. Results are reported over multiple seeds, because single-seed RL results are noise.
5. The whole thing is reproducible from the repo by someone who is not me.

Explicitly *not* a success criterion: any particular success rate, collision rate, or driving score.

## Operating principle: never debug two unknowns at once

The single discipline that structures the entire roadmap. If a hand-written algorithm runs in a
hand-written environment and nothing learns, there is no way to tell which is broken, and that
failure mode is expensive.

| Phase kind | Environment | Algorithm | What is being validated |
|---|---|---|---|
| 0 | Trusted (MetaDrive) | Trusted (SB3) | The harness: logging, evaluation, seeding, run layout |
| 1 | Trusted (classic control first, then MetaDrive) | **Mine** | My algorithm implementation, against SB3's curve |
| 2 | **Mine** (custom scenarios and capabilities) | Trusted (mine, now verified) | My environment |
| 3 | Mine | Mine | The actual project |

The fast known-good environment is **classic control** — CartPole for discrete methods, Pendulum
for continuous. If a hand-written PPO fails on CartPole, the bug is unambiguously in the
algorithm, with no driving-specific confound.

See `docs/03_roadmap.md` for how this maps onto concrete phases.
