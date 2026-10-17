# rl-car-2d 🚗

A car in a 2D world that **learns to drive on its own** through *reinforcement learning*: nobody teaches it the rules — it tries, fails, and improves through rewards and penalties.

The end goal: stay on the road, stop at red lights, obey traffic signs, and avoid crashes.

> 🚧 Work in progress: I'm building this step by step to learn machine learning.

## How it works

- **Environment**: a straight two-lane road seen from above, drawn with Pygame. The camera follows the car. The simulation runs without a window during training.
- **Observations**: three numbers, normalized to -1..1 — distance to the left edge of the road, distance to the right edge, and offset from the center of the right lane.
- **Actions**: for now the car moves in 4 directions (left/right, forward/back) at a fixed speed. Steering, acceleration and braking come in phase 4.5.
- **Rewards**:
  - \+ for real progress toward the end of the road
  - − for drifting away from the center of the right lane
  - −5 for leaving the road (episode ends)
  - +20 for reaching the end of the road (episode ends)
- **Algorithm**: PPO, via Stable-Baselines3.

The goal is to grow this into a full course with curves, traffic lights, signs, other cars and pedestrians.

## Roadmap

- [x] Phase 1: window, road, and a car you can drive with the keyboard
- [x] Phase 2: off-road detection + camera that follows the car
- [x] Phase 3: sensors (distances to the road edges and lane center; ray sensors will come with other cars)
- [ ] Phase 4: Gymnasium environment + first training run (stay on the road) ← in progress
  - [x] Gymnasium environment (`car_env.py`)
  - [x] First PPO training: the agent reaches the end of the road, centered in its lane
  - [x] Keyboard mode built on the same environment
  - [ ] Video of the training progress
- [ ] Phase 4.5: car physics — speed with throttle and brake, steering
- [ ] Phase 5: traffic lights
- [ ] Phase 6: traffic signs (stop, speed limits)
- [ ] Phase 7: other cars to avoid

## Requirements

- Python 3.12 (Pygame has no wheels for Python 3.14 yet)
- The libraries listed in `requirements.txt` (Pygame, NumPy, Gymnasium, Stable-Baselines3)

## Installation

```bash
git clone https://github.com/DarkShark02/rl-car-2d.git
cd rl-car-2d
python -m venv .venv
# Windows:   .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python main.py     # drive the car yourself (arrow keys or WASD)
python train.py    # train the agent with PPO, saving a checkpoint every 10k steps
python watch.py    # watch the trained agent drive (optional: path to another model)
python record.py   # make a video of the agent at every checkpoint: videos/<run>_progress.mp4
```

Trained models are saved in `models/`, which is not tracked by git.

## What I learned

_I'll update this section at every phase: problems I ran into, rewards that didn't work, and how I fixed them._
