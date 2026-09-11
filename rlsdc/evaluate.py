"""Deterministic evaluation harness, kept strictly separate from training reward.

Per docs/02_technical_design.md Section 8 ("Evaluation harness, kept strictly
separate from reward"): training reward is the optimization signal, not the
measure of success -- conflating them is how reward hacking goes unnoticed.
This module always rolls out on rlsdc.scenarios' held-out evaluation seeds
(never the training range), and reports metrics pulled directly from
MetaDrive's own `info` dict -- verified present on every step: `arrive_dest`,
`crash`, `out_of_road`, `velocity`, `cost` -- rather than inferred from reward.

`cost` is MetaDrive's native safety signal (1.0 on crash/off-road by default,
see D-020) and is reported as its own column, never folded into the return.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from rlsdc import scenarios

# Observation -> action. Deliberately just a plain callable, not an interface
# tied to any particular library: a random baseline, an SB3 model wrapped as
# `lambda obs: model.predict(obs, deterministic=True)[0]`, and a future
# hand-written algorithm can all be passed here with zero adapter code.
PolicyFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class EpisodeResult:
    scenario_seed: int
    episode_return: float
    episode_cost: float
    length: int
    arrive_dest: bool
    crash: bool
    out_of_road: bool
    route_completion: float
    hit_max_steps: bool  # our own safety cap fired, not MetaDrive's own horizon


@dataclass
class EvalSummary:
    num_episodes: int
    success_rate: float
    collision_rate: float
    off_road_rate: float
    mean_return: float
    mean_cost: float
    mean_length: float
    mean_route_completion: float


def evaluate_policy(
    policy: PolicyFn,
    env_config: Optional[dict] = None,
    max_steps: int = 1000,
) -> List[EpisodeResult]:
    """Roll out `policy` deterministically on all held-out evaluation scenarios.

    Always uses the full reserved evaluation range (rlsdc.scenarios.EVAL_*),
    every time -- this project's convention (see D-022) is that a reported
    evaluation result always covers all 50 held-out scenarios, not a subset,
    so results are comparable across runs without an extra "how many
    scenarios" caveat attached to every number.

    Args:
        policy: maps one observation to one action. No assumptions about
            what produced it.
        env_config: everything about the environment *except* the scenario
            seed range -- traffic density, observation type, etc. The eval
            scenario range is force-applied after this, so a caller cannot
            accidentally leak training seeds into an evaluation run by
            passing start_seed/num_scenarios here; they are always
            overwritten with the held-out range.
        max_steps: hard per-episode cap, purely defensive -- MetaDrive
            truncates on its own horizon well before this in normal use.

    Returns:
        One EpisodeResult per evaluation scenario, in seed order.
        Deterministic: re-running the same policy reproduces identical
        results, since each episode resets with an explicit seed.
    """
    from metadrive.envs.metadrive_env import MetaDriveEnv

    config = dict(env_config or {})
    config.setdefault("use_render", False)
    config.setdefault("log_level", 50)
    # Force-applied last: this is what guarantees eval never silently runs on
    # training seeds, regardless of what a caller passed in env_config.
    config.update(scenarios.eval_scenario_config())

    env = MetaDriveEnv(config)
    results: List[EpisodeResult] = []
    try:
        for i in range(scenarios.EVAL_NUM_SCENARIOS):
            seed = scenarios.EVAL_START_SEED + i
            obs, info = env.reset(seed=seed)
            episode_return = 0.0
            episode_cost = 0.0
            length = 0
            terminated = truncated = False
            while not (terminated or truncated) and length < max_steps:
                action = policy(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_return += reward
                episode_cost += info.get("cost", 0.0)
                length += 1
            results.append(
                EpisodeResult(
                    scenario_seed=seed,
                    episode_return=episode_return,
                    episode_cost=episode_cost,
                    length=length,
                    arrive_dest=bool(info.get("arrive_dest", False)),
                    crash=bool(info.get("crash", False)),
                    out_of_road=bool(info.get("out_of_road", False)),
                    route_completion=float(info.get("route_completion", 0.0)),
                    hit_max_steps=(length >= max_steps and not (terminated or truncated)),
                )
            )
    finally:
        env.close()
    return results


def summarize(results: List[EpisodeResult]) -> EvalSummary:
    n = len(results)
    if n == 0:
        raise ValueError("no episodes to summarize")
    return EvalSummary(
        num_episodes=n,
        success_rate=sum(r.arrive_dest for r in results) / n,
        collision_rate=sum(r.crash for r in results) / n,
        off_road_rate=sum(r.out_of_road for r in results) / n,
        mean_return=sum(r.episode_return for r in results) / n,
        mean_cost=sum(r.episode_cost for r in results) / n,
        mean_length=sum(r.length for r in results) / n,
        mean_route_completion=sum(r.route_completion for r in results) / n,
    )
