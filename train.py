from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from car_env import CarEnv

RUN_NAME = "ppo_car_v2" # Name of this training run: used for the final model and the checkpoints folder
TOTAL_STEPS = 100_000 # How long to train
CHECKPOINT_EVERY = 10_000 # Save a copy of the agent every N steps, to watch or record its progress later

env = CarEnv() # The world the agent will learn in
#env = CarEnv(render_mode="human") # The world the agent will learn in, with a window showing the car and road

model = PPO("MlpPolicy", env, verbose=1) # The agent: PPO algorithm with a small neural network as its "brain"

checkpoints = CheckpointCallback( # Saves models/checkpoints/<RUN_NAME>/<RUN_NAME>_<steps>_steps.zip during training
    save_freq=CHECKPOINT_EVERY,
    save_path=f"models/checkpoints/{RUN_NAME}",
    name_prefix=RUN_NAME,
)

model.learn(total_timesteps=TOTAL_STEPS, callback=checkpoints) # Train: the agent plays and improves, saving checkpoints along the way

model.save(f"models/{RUN_NAME}") # Save the final trained agent to models/<RUN_NAME>.zip
