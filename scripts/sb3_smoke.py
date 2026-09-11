"""SB3 PPO smoke test -- Phase 0 harness correctness oracle.

Purpose: prove a trusted, independently-implemented algorithm can actually
learn through this exact harness (RunDir + train scenario config +
evaluate_policy) before any hand-written algorithm is trusted to use it. If
this doesn't learn, the harness/environment is the suspect, not a
not-yet-written algorithm -- the "never debug two unknowns at once"
operating principle. See docs/03_roadmap.md Phase 0.

This is a smoke test, not a real experiment: it exists to prove the pipe is
not clogged, not to produce a good driving policy. Do not read anything past
"did it learn a clearly positive trend vs. the random baseline" into its
result.

Deliberately conservative on compute -- this project is explicitly
lower-priority than other work already running on this shared workstation:
  - device="cpu": the observation here is a 259-dim vector through a small
    MLP; GPU is documented (D-009) to not help, and sometimes hurt, at this
    scale, so nothing is being sacrificed by skipping it.
  - a single environment, not vectorized -- keeps the CPU footprint to
    roughly one core's worth of work rather than saturating all 8.
  - torch's internal thread pool capped at 2, so PyTorch's own BLAS/MKL
    parallelism doesn't spread across every core on its own initiative.
  - meant to be launched under `nice -n 19` (see the run command below), so
    the OS scheduler always prefers other processes when there's contention.

Usage:  nice -n 19 python scripts/sb3_smoke.py [total_timesteps]
        (default total_timesteps: 100000)
"""

import os

# Must be set before numpy/torch are imported anywhere -- limits internal
# thread pools so this single low-priority process doesn't try to spread
# across every core on its own.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import sys

import torch

torch.set_num_threads(2)

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from rlsdc import scenarios
from rlsdc.artifacts import RunDir
from rlsdc.evaluate import evaluate_policy, summarize

TOTAL_TIMESTEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
ENV_CONFIG = dict(traffic_density=0.1)
SEED = 0


def make_train_env():
    from metadrive.envs.metadrive_env import MetaDriveEnv

    config = dict(ENV_CONFIG)
    config.setdefault("use_render", False)
    config.setdefault("log_level", 50)
    # Training always draws from the training seed range -- never the
    # held-out eval range (rlsdc.scenarios), so results below are a genuine
    # generalization check, not a memorization check.
    config.update(scenarios.train_scenario_config())
    env = MetaDriveEnv(config)
    return Monitor(env)  # tracks per-episode reward/length for SB3's own logging


def main():
    config = dict(
        algo="sb3_ppo",
        env="metadrive",
        seed=SEED,
        total_timesteps=TOTAL_TIMESTEPS,
        env_config=ENV_CONFIG,
        device="cpu",
        n_envs=1,
        purpose="Phase 0 harness correctness oracle -- not a real experiment",
    )

    with RunDir(algo="sb3ppo", env="metadrive", tag="smoke", config=config) as run:
        env = make_train_env()
        try:
            model = PPO(
                "MlpPolicy",
                env,
                device="cpu",
                seed=SEED,
                verbose=1,
            )
            print(f"[sb3_smoke] training for {TOTAL_TIMESTEPS} timesteps, device=cpu, n_envs=1")
            model.learn(total_timesteps=TOTAL_TIMESTEPS)
        finally:
            env.close()

        model.save(str(run.path / "checkpoints" / "ppo_final"))

        def policy(obs):
            action, _ = model.predict(obs, deterministic=True)
            return action

        print()
        print("--- final evaluation on held-out scenarios (never seen in training) ---")
        results = evaluate_policy(policy, env_config=ENV_CONFIG)
        for r in results:
            run.log_metrics(
                {
                    "scenario_seed": r.scenario_seed,
                    "return": r.episode_return,
                    "cost": r.episode_cost,
                    "length": r.length,
                    "arrive_dest": int(r.arrive_dest),
                    "crash": int(r.crash),
                    "out_of_road": int(r.out_of_road),
                    "route_completion": r.route_completion,
                }
            )
        summary = summarize(results)
        print(f"episodes            {summary.num_episodes}")
        print(f"success_rate        {summary.success_rate:.2%}")
        print(f"collision_rate      {summary.collision_rate:.2%}")
        print(f"off_road_rate       {summary.off_road_rate:.2%}")
        print(f"mean_return         {summary.mean_return:.2f}")
        print(f"mean_cost           {summary.mean_cost:.2f}")
        print(f"mean_length         {summary.mean_length:.1f}")
        print(f"mean_route_complet. {summary.mean_route_completion:.2%}")


if __name__ == "__main__":
    main()
