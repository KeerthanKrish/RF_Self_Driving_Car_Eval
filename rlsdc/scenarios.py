"""Train/eval scenario seed ranges for MetaDrive -- decided once, here, so every
script agrees rather than each one inventing its own split.

MetaDrive procedurally generates a road map per scenario, chosen from the
range [start_seed, start_seed + num_scenarios). If training and evaluation
draw from overlapping ranges, an agent can simply memorize eval maps during
training, and "generalization" would actually be measuring memorization.
See docs/02_technical_design.md Section 8, "Seeding and multiple runs", and
D-019's consequence #4 in docs/04_decision_log.md.

Chosen values:
  - Training gets 1000 scenarios (seeds 0..999) -- enough road-layout
    diversity that memorizing all of them is not a realistic shortcut.
  - Evaluation gets a separate block of 50 scenarios, offset by +100,000
    seeds from the training range. The gap is deliberately far larger than
    either range so there is no ambiguity about overlap even if the
    training range grows substantially later; 50 is enough scenarios for a
    stable held-out average without every evaluation being slow.

These are read, not decided, by scripts. Change them here if the split needs
to change, and log why in docs/04_decision_log.md -- not silently per-script.
"""

TRAIN_START_SEED = 0
TRAIN_NUM_SCENARIOS = 1000

EVAL_START_SEED = 100_000
EVAL_NUM_SCENARIOS = 50


def train_scenario_config() -> dict:
    """MetaDrive config keys selecting the training scenario range."""
    return dict(start_seed=TRAIN_START_SEED, num_scenarios=TRAIN_NUM_SCENARIOS)


def eval_scenario_config() -> dict:
    """MetaDrive config keys selecting the held-out evaluation scenario range."""
    return dict(start_seed=EVAL_START_SEED, num_scenarios=EVAL_NUM_SCENARIOS)


def assert_disjoint() -> None:
    """Fails loudly if the two ranges above were ever edited into overlapping.
    Cheap enough to call at the start of any script that uses both."""
    train_range = range(TRAIN_START_SEED, TRAIN_START_SEED + TRAIN_NUM_SCENARIOS)
    eval_range = range(EVAL_START_SEED, EVAL_START_SEED + EVAL_NUM_SCENARIOS)
    overlap = set(train_range) & set(eval_range)
    if overlap:
        raise ValueError(
            f"train/eval scenario seed ranges overlap ({len(overlap)} seeds) -- "
            f"fix rlsdc/scenarios.py before trusting any evaluation result"
        )
