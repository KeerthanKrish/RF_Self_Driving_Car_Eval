#!/usr/bin/env bash
# Verify the `rlsdc` environment is actually usable, not merely installed.
#
# The important check is sm_120 in torch.cuda.get_arch_list(). On this machine
# torch.cuda.is_available() returns True even with wheels that have no
# compatible kernels, so it is NOT a sufficient test -- an actual GPU op is.

set -euo pipefail

source ~/miniforge3/etc/profile.d/conda.sh
conda activate rlsdc

python - <<'PY'
import sys

print(f"python           {sys.version.split()[0]}")

import numpy, gymnasium, torch
print(f"numpy            {numpy.__version__}")
print(f"gymnasium        {gymnasium.__version__}")

import highway_env
print(f"highway-env      {highway_env.__version__}")

import stable_baselines3
print(f"stable-baselines3 {stable_baselines3.__version__}")

print(f"torch            {torch.__version__}  (cuda {torch.version.cuda})")
print(f"cuda available   {torch.cuda.is_available()}")

arches = torch.cuda.get_arch_list()
print(f"arch list        {arches}")

ok = True

if not torch.cuda.is_available():
    print("FAIL: CUDA not available")
    ok = False
else:
    print(f"device           {torch.cuda.get_device_name(0)}")
    cap = torch.cuda.get_device_capability(0)
    print(f"compute cap      {cap[0]}.{cap[1]}")

    if "sm_120" not in arches:
        print("FAIL: sm_120 missing from arch list -- wrong PyTorch build for Blackwell")
        ok = False

    # The real test: is_available() lies, an actual kernel launch does not.
    try:
        x = torch.randn(1024, 1024, device="cuda")
        result = (x @ x).sum().item()
        print(f"gpu matmul       OK ({result:.2f})")
    except Exception as exc:
        print(f"FAIL: GPU op raised: {exc}")
        ok = False

# Environment smoke test: does highway-env actually construct and step?
env = gymnasium.make("highway-v0")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
print(f"highway-v0       OK  obs{obs.shape}  action_space={env.action_space}")
env.close()

print()
print("ALL CHECKS PASSED" if ok else "CHECKS FAILED -- see FAIL lines above")
sys.exit(0 if ok else 1)
PY
