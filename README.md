# RF_Self_Driving_Car

Learning reinforcement learning by building a simulated self-driving agent, capability by
capability. The driving task is the teaching device; the goal is the RL.

**Simulation only. No real hardware. Not aiming for a flawless agent.**

## Start here

| Document | What it is |
|---|---|
| [`docs/00_project_charter.md`](docs/00_project_charter.md) | Goals, non-goals, where "from scratch" starts and stops, success criteria |
| [`docs/01_infrastructure_and_workflow.md`](docs/01_infrastructure_and_workflow.md) | The two machines, environment isolation, git/scp sync, run layout |
| [`docs/02_technical_design.md`](docs/02_technical_design.md) | Simulator, vehicle model, observation/action spaces, reward, algorithms, libraries |
| [`docs/03_roadmap.md`](docs/03_roadmap.md) | Phased capability curriculum and RL concept syllabus |
| [`docs/04_decision_log.md`](docs/04_decision_log.md) | Every decision made, with rationale |
| [`working_instructions.md`](working_instructions.md) | How to work on this project — autonomy, verification, documentation standards |

## Layout

```
docs/         Project plan — the current, authoritative scope
research/     Background dossier (13 files) compiled before scope was set
runs/         Run artifacts — gitignored, moved between machines by scp
```

## Reading the research dossier correctly

`research/` is a thorough survey of driving-RL platforms, algorithms, reward design, pitfalls,
and prior work. It is accurate on the facts and worth consulting.

It was written under the assumption that the goal was **a good driving agent**. This project's
goal is **learning RL**. Several of its recommendations are therefore wrong here — most notably
its advice to use Stable-Baselines3 rather than implementing algorithms yourself, which is
correct for shipping and counterproductive for learning.

The charter has a table of exactly where the two diverge. Read that before following anything in
`research/` literally.

## Two operating rules that shape everything

**Never debug two unknowns at once.** Phases are ordered so at most one of {environment,
algorithm} is unproven at any time. A hand-written algorithm failing in a hand-written
environment tells you nothing about which is broken.

**Reward correctness is tested, not observed.** Preference ordering is asserted in `pytest`
against hand-built trajectory pairs before training. Watching a training run and deciding it
looks reasonable is the exact process that let 7 of 9 published driving reward functions rank a
crash above safe idling.

## Machines

Everything runs on the Ubuntu workstation (`keerthan@100.71.12.16`, over Tailscale). The Mac is
an interface for reading and reviewing. Push from Ubuntu, pull on the Mac. Large files move by
`scp`, never git. Details in the infrastructure doc.
