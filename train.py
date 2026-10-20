import os
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from car_env import CarEnv

TOTAL_STEPS = 1_000_000 # How long to train: city routes with traffic lights (driving, braking at red, starting again at green)
CHECKPOINT_EVERY = 50_000 # Save a copy of the agent every N steps, to watch or record its progress later (20 checkpoints)
SEED = 0 # Fixed seed: the same code gives the same training, so you can tell if a change really helped

TRAFFIC_LIGHTS = "--no-lights" not in sys.argv # Curriculum: python train.py --no-lights (stage A, learn to drive), then continue with the lights (stage B)
LAB = "--lab" in sys.argv # --lab: a straight road with traffic lights, to study the light rules quickly
args = [a for a in sys.argv[1:] if not a.startswith("--")]
env = CarEnv(world="lab" if LAB else "city", traffic_lights=TRAFFIC_LIGHTS) # The world the agent will learn in
#env = CarEnv(render_mode="human") # The world the agent will learn in, with a window showing the car and road

START_FROM = args[0] if len(args) > 0 and args[0] != "new" else None # python train.py [models/<name>.zip | new] [steps]: keep training an existing agent, or start a new one
TOTAL_STEPS = int(args[1]) if len(args) > 1 else TOTAL_STEPS # Optional: how many more steps

# Run names: the stages of one curriculum share the same N. Driving only: ppo_car_vN_drive -> lab (straight road with lights): ppo_car_vN_lab -> city: ppo_car_vN.
# A name already used gets _2, _3...: a run never overwrites another one
RUN_PREFIX = "ppo_car_v"
SUFFIX = "_lab" if LAB else ("" if TRAFFIC_LIGHTS else "_drive")
def used(name):
    return os.path.exists(f"models/{name}.zip") or os.path.exists(f"models/checkpoints/{name}")
stage_suffix = next((s for s in ("_drive", "_lab") if START_FROM and START_FROM.endswith(s + ".zip")), None) # Continuing from an earlier stage?
if stage_suffix and TRAFFIC_LIGHTS:
    base = os.path.basename(START_FROM)[: -len(stage_suffix + ".zip")] + SUFFIX # Next stage: the same N as the model it starts from
else:
    version = 1
    while any(used(f"{RUN_PREFIX}{version}{suffix}") for suffix in ("", "_drive", "_lab")):
        version += 1 # The first number not used yet, by any stage
    base = f"{RUN_PREFIX}{version}{SUFFIX}"
RUN_NAME, copy = base, 2
while used(RUN_NAME):
    RUN_NAME, copy = f"{base}_{copy}", copy + 1
print(f"Training run: {RUN_NAME}")

if START_FROM: # Continue: the agent keeps everything it learned (and its settings, like gamma)
    model = PPO.load(START_FROM, env=env, verbose=1, seed=SEED)
    print(f"Continuing from {START_FROM} for {TOTAL_STEPS:,} steps")
else:
    model = PPO("MlpPolicy", env, verbose=1, seed=SEED, ent_coef=0.01, gamma=0.998) # The agent: PPO with a small neural network as its "brain". gamma = how much future rewards count: with 60 steps per second the default 0.99 only "sees" ~2 s ahead, too short to wait at a red light

checkpoints = CheckpointCallback( # Saves models/checkpoints/<RUN_NAME>/<RUN_NAME>_<steps>_steps.zip during training
    save_freq=CHECKPOINT_EVERY,
    save_path=f"models/checkpoints/{RUN_NAME}",
    name_prefix=RUN_NAME,
)

model.learn(total_timesteps=TOTAL_STEPS, callback=checkpoints) # Train: the agent plays and improves, saving checkpoints along the way

model.save(f"models/{RUN_NAME}") # Save the final trained agent to models/<RUN_NAME>.zip
