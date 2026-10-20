import os
import re
import sys

import numpy as np
import pygame
from moviepy import ImageSequenceClip
from stable_baselines3 import PPO

from car_env import CarEnv, HEIGHT, WIDTH

def latest_run():
    # The most recent training run in models/checkpoints (by modification time)
    runs = [os.path.join("models/checkpoints", name) for name in os.listdir("models/checkpoints")]
    return os.path.basename(max(runs, key=os.path.getmtime))


args = [a for a in sys.argv[1:] if not a.startswith("--")]
RUN_NAME = args[0] if args else latest_run() # Which training run to record: python record.py [run name] [frame skip] [--stages], default = latest
STAGES = "--stages" in sys.argv # --stages: one video with the whole curriculum of a run: <run>_drive, then <run>_lab, then <run> (the city)
OUTPUT_PATH = f"videos/{RUN_NAME}_{'stages' if STAGES else 'progress'}.mp4"
FRAME_SKIP = int(args[1]) if len(args) > 1 else 3 # Keep 1 frame out of N: the video plays at Nx speed (6x was too fast to follow the traffic lights)
STAGE_TITLES = { # --stages: the title shown before each stage
    "_drive": "Stage 1 - learning to drive (city without traffic lights)",
    "_lab": "Stage 2 - traffic lights on a straight road (the lab)",
    "": "Stage 3 - everything together: the city with traffic lights",
}
TITLE_SECONDS = 2 # How long a stage title stays on screen
VIDEO_FPS = 60
RECORD_SEED = 7 # Every clip drives the same random road (a road never used in training), so the clips can be compared
MAX_CLIP_SECONDS = 10 if STAGES else 6 # Cut episodes that last longer (e.g. an agent standing still until the time limit). City routes with traffic lights last 7-9 s at 3x
HOLD_SECONDS = 1 # Hold the last frame of each clip, so the final result is readable
TARGET_SECONDS = 30 # The video (with --stages: each stage) should last about this long: less important clips are dropped to fit
CLEAR_IMPROVEMENT = 3 # With the same outcome, a clip is shown only if its reward is at least this much better than every clip before...
CLEAR_PROGRESS = 0.05 # ...or if it drives at least 5% more of the road


def checkpoint_steps(filename):
    # "ppo_car_v2_30000_steps.zip" -> 30000
    return int(re.search(r"_(\d+)_steps\.zip$", filename).group(1))


SHORT_OUTCOMES = { # The reason why an episode ended (info["reason"] from CarEnv), short enough for a label
    "finish": "finish",
    "off the route (in the city: also a street that is not the route)": "off the route",
    "behind the start line": "off the route",
    "crossed on red": "ran a red light",
    "crossed on a yellow it could stop for": "ran a yellow light",
    "stalled (stopped too long, or too far from the stop line)": "stalled",
    "time out (too slow)": "time out",
}


def world_of(run_name):
    # The world a run learned in: (world, traffic lights). ppo_car_vN_drive = city without lights, ppo_car_vN_lab = the lab, ppo_car_vN = city with lights
    return ("lab" if "_lab" in run_name else "city"), "_drive" not in run_name


def play(model, deterministic, world, traffic_lights, render=False, label="", font=None):
    # Play one episode on the recording road; return (outcome, total reward, steps, frames, share of the road driven)
    env = CarEnv(render_mode="rgb_array" if render else None, world=world, traffic_lights=traffic_lights)
    obs, info = env.reset(seed=RECORD_SEED)
    frames = []
    total_reward = 0
    terminated = truncated = False
    reason = None
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        reason = info["reason"]
        if render and (env.step_count % FRAME_SKIP == 0 or terminated or truncated):
            frames.append(add_label(env.render(), f"{label}   reward: {total_reward:.1f}", font))
        if env.step_count >= MAX_CLIP_SECONDS * VIDEO_FPS * FRAME_SKIP:
            break # Long enough for the video
    if render:
        frames.extend([frames[-1]] * HOLD_SECONDS * VIDEO_FPS)
    progress = env.road_index / (len(env.road_points) - 1)
    outcome = SHORT_OUTCOMES.get(reason, reason) if reason else "not finished" # No reason: the clip was cut before the end
    return outcome, total_reward, env.step_count, frames, progress


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
    if len(candidates) == 1:
        return candidates
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


def title_frames(text, font):
    # A dark screen with a title in the middle, held TITLE_SECONDS
    surface = pygame.Surface((WIDTH, HEIGHT))
    surface.fill((20, 20, 20))
    title = font.render(text, True, (255, 255, 255))
    surface.blit(title, title.get_rect(center=(WIDTH // 2, HEIGHT // 2)))
    frame = np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))
    return [frame] * (TITLE_SECONDS * VIDEO_FPS)


def record_run(run_name, font, with_untrained=True):
    # The frames of one run: its checkpoints played on the recording road, keeping only the clips that tell the story
    world, traffic_lights = world_of(run_name)
    env = CarEnv(world=world, traffic_lights=traffic_lights)
    checkpoint_dir = f"models/checkpoints/{run_name}"
    candidates = [{"model": PPO("MlpPolicy", env, seed=0), "label": "Untrained (0 steps)", "deterministic": False}] if with_untrained else [] # A random agent: where it all begins
    for filename in sorted(os.listdir(checkpoint_dir), key=checkpoint_steps):
        model = PPO.load(os.path.join(checkpoint_dir, filename))
        if model.observation_space.shape != env.observation_space.shape or model.action_space != env.action_space: # Old runs were trained on a different world (e.g. 3 observations instead of 5)
            sys.exit(f"{run_name} was trained on an older version of the environment: record a newer run.")
        candidates.append({"model": model, "label": f"After {checkpoint_steps(filename):,} steps", "deterministic": True})

    for clip in candidates: # 1. Play every checkpoint without drawing (fast), to choose what to show
        clip["outcome"], clip["reward"], clip["steps"], _, clip["progress"] = play(clip["model"], clip["deterministic"], world, traffic_lights)

    frames = []
    for clip in choose_clips(candidates): # 2. Draw only the chosen clips
        print(f'{run_name} - {clip["label"]}: {clip["outcome"]}, reward {clip["reward"]:.1f}')
        label = f'{clip["label"]}: {clip["outcome"]}' # The outcome is known from step 1, so the label can say it from the start
        frames.extend(play(clip["model"], clip["deterministic"], world, traffic_lights, render=True, label=label, font=font)[3])
    return frames


pygame.init()
font = pygame.font.SysFont(None, 32)

if STAGES: # The whole curriculum: every stage starts from the model of the one before, so only the first one starts untrained
    all_frames = []
    for number, (suffix, title) in enumerate(STAGE_TITLES.items()):
        all_frames += title_frames(title, font)
        all_frames += record_run(RUN_NAME + suffix, font, with_untrained=number == 0)
else:
    all_frames = record_run(RUN_NAME, font)

os.makedirs("videos", exist_ok=True)
ImageSequenceClip(all_frames, fps=VIDEO_FPS).write_videofile(OUTPUT_PATH, codec="libx264", logger=None)
pygame.quit()
print(f"Saved {OUTPUT_PATH} ({os.path.getsize(OUTPUT_PATH) / 1_000_000:.1f} MB, {len(all_frames) / VIDEO_FPS:.0f} s)")
