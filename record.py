import os
import re
import sys

import numpy as np
import pygame
from moviepy import ImageSequenceClip
from stable_baselines3 import PPO

from car_env import CarEnv


def latest_run():
    # The most recent training run in models/checkpoints (by modification time)
    runs = [os.path.join("models/checkpoints", name) for name in os.listdir("models/checkpoints")]
    return os.path.basename(max(runs, key=os.path.getmtime))


RUN_NAME = sys.argv[1] if len(sys.argv) > 1 else latest_run() # Which training run to record: python record.py [run name], default = latest
CHECKPOINT_DIR = f"models/checkpoints/{RUN_NAME}"
OUTPUT_PATH = f"videos/{RUN_NAME}_progress.mp4"
FRAME_SKIP = 3 # Keep 1 frame out of 3: the video plays at 3x speed, so it stays short
VIDEO_FPS = 60
RECORD_SEED = 7 # Every clip drives the same random road (a road never used in training), so the clips can be compared
MAX_CLIP_SECONDS = 6 # Cut episodes that last longer (e.g. an agent standing still until the time limit)
HOLD_SECONDS = 1 # Hold the last frame of each clip, so the final result is readable
TARGET_SECONDS = 30 # The video should last about this long: less important clips are dropped to fit
CLEAR_IMPROVEMENT = 3 # With the same outcome, a clip is shown only if its reward is at least this much better than every clip before...
CLEAR_PROGRESS = 0.05 # ...or if it drives at least 5% more of the road


def checkpoint_steps(filename):
    # "ppo_car_v2_30000_steps.zip" -> 30000
    return int(re.search(r"_(\d+)_steps\.zip$", filename).group(1))


def outcome_of(env, terminated):
    # "finish", "off road" or "time out" (also works with the older straight-road environments)
    if not terminated:
        return "time out"
    if hasattr(env, "road_points"):
        reached_end = env.road_index == len(env.road_points) - 1
    else:
        reached_end = env.car.top < env.road.top # Old straight road: the finish was the top of the road
    return "finish" if reached_end else "off road"


def play(model, deterministic, render=False, label="", font=None):
    # Play one episode on the recording road; return (outcome, total reward, steps, frames, share of the road driven)
    env = CarEnv(render_mode="rgb_array" if render else None)
    obs, info = env.reset(seed=RECORD_SEED)
    frames = []
    total_reward = 0
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if render and (env.step_count % FRAME_SKIP == 0 or terminated or truncated):
            frames.append(add_label(env.render(), f"{label}   reward: {total_reward:.1f}", font))
        if env.step_count >= MAX_CLIP_SECONDS * VIDEO_FPS * FRAME_SKIP:
            break # Long enough for the video
    if render:
        frames.extend([frames[-1]] * HOLD_SECONDS * VIDEO_FPS)
    progress = env.road_index / (len(env.road_points) - 1) if hasattr(env, "road_points") else 0 # Old straight roads: not measured
    return outcome_of(env, terminated), total_reward, env.step_count, frames, progress


def add_label(frame, text, font):
    # Write a label in the top-left corner of a frame (frame is a (height, width, 3) array)
    surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
    label = font.render(text, True, (255, 255, 255))
    background = pygame.Surface((label.get_width() + 20, label.get_height() + 10))
    background.set_alpha(160)
    surface.blit(background, (10, 10))
    surface.blit(label, (20, 15))
    return np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))


def clip_seconds(clip):
    # How long a clip will last in the video
    return min(clip["steps"] / FRAME_SKIP / VIDEO_FPS, MAX_CLIP_SECONDS) + HOLD_SECONDS


def choose_clips(candidates):
    # Keep the clips that tell the story: the untrained agent, every clear improvement, the first step back
    # after learning to finish (learning is not a straight line), and the final result. Then fit TARGET_SECONDS.
    chosen = [candidates[0]]
    best = candidates[0]
    shown_step_back = False
    for clip in candidates[1:-1]:
        finishes, best_finishes = clip["outcome"] == "finish", best["outcome"] == "finish"
        clearly_better = clip["reward"] >= best["reward"] + CLEAR_IMPROVEMENT or clip["progress"] >= best["progress"] + CLEAR_PROGRESS
        if (finishes and not best_finishes) or (finishes == best_finishes and clearly_better):
            clip["why"] = "improvement"
            chosen.append(clip)
            best = clip
        elif best_finishes and not finishes and not shown_step_back:
            clip["why"] = "step back"
            clip["label"] += " (a step back)"
            chosen.append(clip)
            shown_step_back = True
    chosen.append(candidates[-1]) # The final result is always shown

    while sum(clip_seconds(c) for c in chosen) > TARGET_SECONDS and len(chosen) > 2:
        middle = chosen[1:-1]
        step_backs = [c for c in middle if c.get("why") == "step back"]
        drop = step_backs[0] if step_backs else min(middle[:-1] or middle, key=lambda c: c["reward"]) # First the step back, then the weakest improvement
        chosen.remove(drop)
    return chosen


pygame.init()
font = pygame.font.SysFont(None, 32)

env = CarEnv()
candidates = [{"model": PPO("MlpPolicy", env, seed=0), "label": "Untrained (0 steps)", "deterministic": False}] # A random agent: where it all begins
for filename in sorted(os.listdir(CHECKPOINT_DIR), key=checkpoint_steps):
    model = PPO.load(os.path.join(CHECKPOINT_DIR, filename))
    if model.observation_space.shape != env.observation_space.shape: # Old runs were trained on a different world (e.g. 3 observations instead of 5)
        sys.exit(f"{RUN_NAME} was trained on an older version of the environment: record a newer run.")
    candidates.append({"model": model, "label": f"After {checkpoint_steps(filename):,} steps", "deterministic": True})

for clip in candidates: # 1. Play every checkpoint without drawing (fast), to choose what to show
    clip["outcome"], clip["reward"], clip["steps"], _, clip["progress"] = play(clip["model"], clip["deterministic"])

all_frames = []
for clip in choose_clips(candidates): # 2. Draw only the chosen clips
    print(f'{clip["label"]}: {clip["outcome"]}, reward {clip["reward"]:.1f}')
    all_frames.extend(play(clip["model"], clip["deterministic"], render=True, label=clip["label"], font=font)[3])

os.makedirs("videos", exist_ok=True)
ImageSequenceClip(all_frames, fps=VIDEO_FPS).write_videofile(OUTPUT_PATH, codec="libx264", logger=None)
pygame.quit()
print(f"Saved {OUTPUT_PATH} ({os.path.getsize(OUTPUT_PATH) / 1_000_000:.1f} MB, {len(all_frames) / VIDEO_FPS:.0f} s)")
