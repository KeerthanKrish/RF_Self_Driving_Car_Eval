"""Render a few frames of highway-v0 headless, to confirm visualization works
over SSH without a display, and to produce something reviewable on the Mac.

highway-env renders with pygame. Under `render_mode="rgb_array"` it draws to a
surface and hands back a numpy array, so no window and no X server are needed.

Use highway-env's own OFFSCREEN_RENDERING flag, which makes it skip
pygame.display.set_mode() and read pixels straight off the drawing surface.

Do NOT set SDL_VIDEODRIVER=dummy. Measured on dtgpu 2026-09-10: the dummy
driver yields an all-black frame (mean 0.0, one unique colour) whether or not
OFFSCREEN_RENDERING is also set. It fails silently -- valid shape, no pixels.

Usage:  python scripts/render_smoke.py [out_dir]
"""

import os

# Must be set before pygame is imported by highway-env.
os.environ.setdefault("OFFSCREEN_RENDERING", "1")

import sys
from pathlib import Path

import gymnasium
import imageio.v2 as imageio
import numpy as np

import highway_env  # noqa: F401  (registers the envs with gymnasium)

out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/render_smoke")
out_dir.mkdir(parents=True, exist_ok=True)

env = gymnasium.make("highway-v0", render_mode="rgb_array")
obs, info = env.reset(seed=0)

frames = []
for step in range(30):
    frame = env.render()
    frames.append(frame)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        break
env.close()

frames = np.asarray(frames)
print(f"captured {len(frames)} frames, each {frames[0].shape} dtype={frames[0].dtype}")

imageio.imwrite(out_dir / "frame_000.png", frames[0])
imageio.imwrite(out_dir / "frame_mid.png", frames[len(frames) // 2])
imageio.mimsave(out_dir / "episode.mp4", frames, fps=10)

print(f"wrote {out_dir}/frame_000.png")
print(f"wrote {out_dir}/frame_mid.png")
print(f"wrote {out_dir}/episode.mp4")
