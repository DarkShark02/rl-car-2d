import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from car_env import CarEnv

RUN_PREFIX = "ppo_car_v" # Every training run gets the next free name: ppo_car_v1, ppo_car_v2, ppo_car_v3...
version = 1
while os.path.exists(f"models/{RUN_PREFIX}{version}.zip") or os.path.exists(f"models/checkpoints/{RUN_PREFIX}{version}"):
    version += 1
RUN_NAME = f"{RUN_PREFIX}{version}" # Name of this training run: used for the final model and the checkpoints folder
print(f"Training run: {RUN_NAME}")
TOTAL_STEPS = 30_000 # How long to train: on the straight road the agent learns everything in ~20k steps
CHECKPOINT_EVERY = 2_500 # Save a copy of the agent every N steps, to watch or record its progress later (dense, to capture the learning)
SEED = 0 # Fixed seed: the same code gives the same training, so you can tell if a change really helped

env = CarEnv() # The world the agent will learn in
#env = CarEnv(render_mode="human") # The world the agent will learn in, with a window showing the car and road

model = PPO("MlpPolicy", env, verbose=1, seed=SEED) # The agent: PPO algorithm with a small neural network as its "brain"

checkpoints = CheckpointCallback( # Saves models/checkpoints/<RUN_NAME>/<RUN_NAME>_<steps>_steps.zip during training
    save_freq=CHECKPOINT_EVERY,
    save_path=f"models/checkpoints/{RUN_NAME}",
    name_prefix=RUN_NAME,
)

model.learn(total_timesteps=TOTAL_STEPS, callback=checkpoints) # Train: the agent plays and improves, saving checkpoints along the way

model.save(f"models/{RUN_NAME}") # Save the final trained agent to models/<RUN_NAME>.zip
