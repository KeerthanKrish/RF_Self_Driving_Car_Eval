"""Random-policy baseline -- Phase 0 harness, first real run.

Purpose:
  1. Establish what "doing nothing intelligent" looks like on MetaDrive's
     held-out evaluation scenarios -- the floor every future algorithm gets
     compared against.
  2. Prove rlsdc.artifacts.RunDir and rlsdc.evaluate work correctly together
     end-to-end, on a policy simple enough that nothing about the *policy*
     can be the explanation for a surprising number.

Usage: python scripts/baseline_random.py [--seed N]
"""

import argparse

import numpy as np

from rlsdc.artifacts import RunDir
from rlsdc.evaluate import evaluate_policy, summarize

ENV_CONFIG = dict(traffic_density=0.1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0,
                         help="policy RNG seed -- vary this across runs for multi-seed comparisons")
    args = parser.parse_args()
    policy_seed = args.seed

    # Seeded once, at the top -- this makes even a "random" policy's
    # performance on this particular run reproducible, rather than a fresh
    # unseeded draw producing a different number every invocation.
    rng = np.random.default_rng(policy_seed)

    def random_policy(obs):
        # MetaDrive's action space is Box(-1, 1, (2,)) -- steering,
        # throttle/brake (verified in D-019). Hardcoded rather than read off
        # an env object, since evaluate_policy owns env construction
        # internally so eval always uses the held-out seed range.
        return rng.uniform(-1.0, 1.0, size=2).astype(np.float32)

    config = dict(
        algo="random",
        env="metadrive",
        policy_seed=policy_seed,
        env_config=ENV_CONFIG,
    )

    with RunDir(algo="random", env="metadrive", tag=f"baseline-seed{policy_seed}", config=config) as run:
        results = evaluate_policy(random_policy, env_config=ENV_CONFIG)

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
        print()
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
