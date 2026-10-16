from stable_baselines3 import PPO

from car_env import CarEnv

env = CarEnv() # The world the agent will learn in

model = PPO("MlpPolicy", env, verbose=1) # The agent: PPO algorithm with a small neural network as its "brain"

model.learn(total_timesteps=100_000) # Train: the agent plays and improves for 100k steps

model.save("models/ppo_car_v1") # Save the trained agent to models/ppo_car_v1.zip
