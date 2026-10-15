# rl-car-2d 🚗

A car in a 2D world that **learns to drive on its own** through *reinforcement learning*: nobody teaches it the rules — it tries, fails, and improves through rewards and penalties.

The end goal: stay on the road, stop at red lights, obey traffic signs, and avoid crashes.

> 🚧 Work in progress: I'm building this step by step to learn machine learning.

## How it works

- **Environment**: a top-down 2D road drawn with Pygame. The camera follows the car.
- **Observations**: the car "sees" the world through ray sensors (distance to the road edges), plus its own speed and the state of the traffic lights.
- **Actions**: steer, accelerate, brake.
- **Rewards**: + for moving forward along the road, − for leaving the road, running a red light, or speeding.
- **Algorithm**: PPO, via Stable-Baselines3.

## Roadmap

- [ ] Phase 1: window, road, and a car you can drive with the keyboard
- [ ] Phase 2: off-road detection + camera that follows the car
- [ ] Phase 3: ray sensors
- [ ] Phase 4: Gymnasium environment + first training run (stay on the road)
- [ ] Phase 5: traffic lights
- [ ] Phase 6: traffic signs (stop, speed limits)
- [ ] Phase 7: other cars to avoid

## Requirements

- Python 3.11 or 3.12
- The libraries listed in `requirements.txt`

## Installation

```bash
git clone https://github.com/<your-username>/rl-car-2d.git
cd rl-car-2d
python -m venv .venv
# Windows:   .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

_Coming soon, as the project progresses._

## What I learned

_I'll update this section at every phase: problems I ran into, rewards that didn't work, and how I fixed them._
