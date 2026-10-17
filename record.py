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
MAX_CLIP_SECONDS = 6 # Cut episodes that last longer (e.g. an agent standing still until the time limit)


def checkpoint_steps(filename):
    # "ppo_car_v2_30000_steps.zip" -> 30000
    return int(re.search(r"_(\d+)_steps\.zip$", filename).group(1))


def add_label(frame, text, font):
    # Write a label in the top-left corner of a frame (frame is a (height, width, 3) array)
    surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
    label = font.render(text, True, (255, 255, 255))
    background = pygame.Surface((label.get_width() + 20, label.get_height() + 10))
    background.set_alpha(160)
    surface.blit(background, (10, 10))
    surface.blit(label, (20, 15))
    return np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))


def record_episode(model, label, font, deterministic=True):
    # Play one episode with the given model and return its labeled frames
    env = CarEnv(render_mode="rgb_array")
    obs, info = env.reset()
    frames = []
    total_reward = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        if env.step_count % FRAME_SKIP == 0 or done:
            frames.append(add_label(env.render(), f"{label}   reward: {total_reward:.1f}", font))
        if len(frames) >= MAX_CLIP_SECONDS * VIDEO_FPS:
            break
    frames.extend([frames[-1]] * VIDEO_FPS) # Hold the last frame for 1 second, so the final result is readable
    print(f"{label}: {env.step_count} steps, reward {total_reward:.2f}")
    return frames


pygame.init()
font = pygame.font.SysFont(None, 32)

env = CarEnv()
clips = [(PPO("MlpPolicy", env, seed=0), "Untrained (0 steps)", False)] # Start with a random agent (stochastic: it wanders), to show where it all begins

files = sorted(os.listdir(CHECKPOINT_DIR), key=checkpoint_steps)
for filename in files:
    steps = checkpoint_steps(filename)
    model = PPO.load(os.path.join(CHECKPOINT_DIR, filename))
    if model.observation_space.shape != env.observation_space.shape: # Old runs were trained on a different world (e.g. 3 observations instead of 5)
        sys.exit(f"{RUN_NAME} was trained on an older version of the environment: record a newer run.")
    clips.append((model, f"After {steps:,} steps", True)) # Trained agents: their best action

all_frames = []
for model, label, deterministic in clips:
    all_frames.extend(record_episode(model, label, font, deterministic))

os.makedirs("videos", exist_ok=True)
ImageSequenceClip(all_frames, fps=VIDEO_FPS).write_videofile(OUTPUT_PATH, codec="libx264", logger=None)
pygame.quit()
print(f"Saved {OUTPUT_PATH} ({os.path.getsize(OUTPUT_PATH) / 1_000_000:.1f} MB)")
