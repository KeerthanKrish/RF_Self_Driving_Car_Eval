"""MetaDrive smoke test, staged by GPU risk.

The workstation's GPU is shared with another project, so this deliberately
separates what needs the GPU from what does not:

  headless : physics stepping only, no rendering  -> no GPU
  topdown  : CPU top-down raster renderer         -> no GPU
  3d       : Panda3D offscreen 3D render          -> needs a GL context

Usage:  python scripts/metadrive_smoke.py {headless|topdown|3d} [out_dir]
"""

import sys
from pathlib import Path

import numpy as np

mode = sys.argv[1] if len(sys.argv) > 1 else "headless"
out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else f"runs/metadrive_smoke_{mode}")

from metadrive.envs.metadrive_env import MetaDriveEnv

# num_scenarios > 1 exercises procedural map generation, which is the feature
# that matters for the generalization work later in the roadmap.
config = dict(
    use_render=False,       # never open a window; this box is headless
    num_scenarios=5,
    start_seed=0,
    traffic_density=0.1,
    log_level=50,           # quiet
)

# 3D frames do not come from env.render() -- MetaDrive produces them through a
# camera sensor attached to the ego vehicle, returned inside the observation.
# This is the only mode that needs a GL context, hence a separate branch.
if mode == "3d":
    from metadrive.component.sensors.rgb_camera import RGBCamera

    config.update(
        image_observation=True,
        sensors={"rgb_camera": (RGBCamera, 640, 360)},
        vehicle_config={"image_source": "rgb_camera"},
        norm_pixel=False,   # keep uint8 so frames are writable as-is
    )

def latest_frame(observation):
    """Pull the camera image out of an image-observation dict.

    MetaDrive returns a stack of recent frames as (H, W, C, T); the newest is
    the last along the trailing axis.
    """
    img = observation["image"]
    img = np.asarray(img)
    if img.ndim == 4:
        img = img[..., -1]
    return img.astype(np.uint8)


env = MetaDriveEnv(config)
try:
    obs, info = env.reset(seed=0)
    if mode == "3d":
        print(f"observation keys  {list(obs.keys())}")
        print(f"image shape       {np.asarray(obs['image']).shape}")
        print(f"state shape       {np.asarray(obs['state']).shape}")
    else:
        print(f"observation      shape={np.asarray(obs).shape} dtype={np.asarray(obs).dtype}")
    print(f"observation_space {env.observation_space}")
    print(f"action_space      {env.action_space}")

    total_reward = 0.0
    frames = []
    steps = 0
    for i in range(100):
        # Mild throttle, no steering: enough to show the physics integrating.
        obs, reward, terminated, truncated, info = env.step([0.0, 0.5])
        total_reward += reward
        steps += 1

        if mode == "topdown":
            frames.append(env.render(mode="topdown", window=False,
                                     screen_size=(600, 600), scaling=3))
        elif mode == "3d":
            frames.append(latest_frame(obs))

        if terminated or truncated:
            break

    print(f"stepped          {steps} steps, total reward {total_reward:.2f}")
    print(f"final speed      {info.get('velocity', float('nan')):.2f} m/s")
    print(f"arrive_dest={info.get('arrive_dest')} crash={info.get('crash')} "
          f"out_of_road={info.get('out_of_road')}")

    if frames:
        import imageio.v2 as imageio
        out_dir.mkdir(parents=True, exist_ok=True)
        arr = np.asarray(frames[len(frames) // 2])
        # A blank render is the failure mode that does not raise -- check pixels,
        # not file existence. See D-018.
        print(f"frame            shape={arr.shape} mean={arr.mean():.2f} "
              f"colors={len(np.unique(arr.reshape(-1, arr.shape[-1]), axis=0))} "
              f"{'BLANK' if arr.max() == 0 else 'HAS CONTENT'}")
        imageio.imwrite(out_dir / "frame_mid.png", arr)
        imageio.mimsave(out_dir / "episode.mp4", frames, fps=20)
        print(f"wrote            {out_dir}/frame_mid.png and episode.mp4")
finally:
    env.close()

print("OK")
