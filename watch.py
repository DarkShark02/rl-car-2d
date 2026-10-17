import glob
import os
import sys

import pygame
from stable_baselines3 import PPO

from car_env import CarEnv


def latest_model():
    # The most recently trained model in models/ (by modification time)
    return max(glob.glob("models/*.zip"), key=os.path.getmtime)


MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else latest_model() # Model to watch: python watch.py [models/<name>.zip], default = latest
EPISODES = 3 # How many episodes to show

env = CarEnv(render_mode="human") # Same world as training, but drawn in a window
model = PPO.load(MODEL_PATH) # Load the trained agent
if model.observation_space.shape != env.observation_space.shape: # Old models were trained on a different world (e.g. 3 observations instead of 5)
    sys.exit(f"{MODEL_PATH} was trained on an older version of the environment "
             f"({model.observation_space.shape[0]} observations, now {env.observation_space.shape[0]}): train a new model or pick another one.")
print(f"Watching {MODEL_PATH}")

for episode in range(EPISODES):
    obs, info = env.reset() # Start a new episode
    total_reward = 0
    done = False
    while not done:
        if pygame.event.peek(pygame.QUIT): # Close the window to stop early
            env.close()
            sys.exit()
        action, _ = model.predict(obs, deterministic=True) # The agent picks its best action
        obs, reward, terminated, truncated, info = env.step(action) # The world moves one step (and redraws itself)
        total_reward += reward
        done = terminated or truncated
    print(f"Episode {episode + 1}: reward {total_reward:.2f}, steps {env.step_count}")

env.close()
