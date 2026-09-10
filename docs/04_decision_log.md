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

### D-004 — Simulator is highway-env
**Date**: 2026-09-10
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
**Rationale**: ~25-dimensional observations through small MLPs are dominated by kernel-launch
overhead; environment stepping in highway-env is pure-Python and CPU-bound. GPU would likely be
slower.
**Consequence**: Also reduces contention on a workstation shared with other projects. Recorded so
that slow Phase 1 training is not misdiagnosed as a GPU configuration problem.
