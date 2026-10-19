# rl-car-2d 🚗

A car in a 2D world that **learns to drive on its own** through *reinforcement learning*: nobody teaches it the rules — it tries, fails, and improves through rewards and penalties.

The end goal: stay on the road, stop at red lights, obey traffic signs, and avoid crashes.

> 🚧 Work in progress: I'm building this step by step to learn machine learning.

## Training progress

### Phase 4.5 — real car physics
The agent learns to accelerate, steer and stay in its own lane (3x speed).


https://github.com/user-attachments/assets/5dac57af-114c-4ab1-9fa4-d5417ba433e7


### Phase 4 — first steps on a straight road
From a random agent to a perfect run, one clip every 2,500 training steps (3x speed).


https://github.com/user-attachments/assets/b7041657-cf57-4038-a089-1e70d30efc86


## How it works

- **Environment**: a straight two-lane road seen from above, drawn with Pygame. The camera follows the car. The simulation runs without a window during training.
- **Observations**: five numbers, normalized to -1..1 — distance to the left edge of the road, distance to the right edge, offset from the center of the right lane, speed, and direction of the car relative to the road.
- **Actions**: steering (left / straight / right) and pedals (throttle / none / brake). The car has real momentum: it accelerates gradually, slows down by itself, and only turns while moving. No reverse gear yet.
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
- [x] Phase 4: Gymnasium environment + first training run (stay on the road)
  - [x] Gymnasium environment (`car_env.py`)
  - [x] First PPO training: the agent reaches the end of the road, centered in its lane
  - [x] Keyboard mode built on the same environment
  - [x] Video of the training progress
- [x] Phase 4.5: car physics — speed with throttle and brake, steering, a real car sprite
- [ ] Phase 5: curves — a road made of pieces, real ray sensors, progress measured along the road
- [ ] Phase 6: traffic lights and intersections
- [ ] Phase 7: traffic signs (stop, speed limits)
- [ ] Phase 8: other cars and pedestrians
- [ ] Phase 9: parking and maneuvers — reverse gear, parking spots, precise low-speed driving
- [ ] Phase 10: procedural world — a generator that builds a different world every episode (curves, intersections, roundabouts, lights, signs, traffic, parking lots), so the agent learns to drive, not to memorize one track
- [ ] Phase 11: scenery (trees, buildings) and the final training run from scratch, recorded from start to finish

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
python train.py    # train the agent with PPO, saving a checkpoint every 2,500 steps
python watch.py    # watch the latest trained agent drive (optional: path to another model)
python record.py   # make a video of the latest run at every checkpoint: videos/<run>_progress.mp4
```

Trained models are saved in `models/`, which is not tracked by git. Every training run gets a new name (`ppo_car_v1`, `ppo_car_v2`, ...). Models trained before a change to the observations can't be loaded by the new environment: `watch.py` and `record.py` tell you when that happens.

### Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

The tests check the environment rules (rewards, end of episode, wall, time limit) and the Gymnasium API, in under a second.

## What I learned

_I'll update this section at every phase: problems I ran into, rewards that didn't work, and how I fixed them._

### Phases 1–4
- **Keep the simulation separate from the drawing.** Training runs without a window and thousands of times faster, so the rules of the world (movement, edges, rewards) must not depend on anything visual. Even the camera lives only in the drawing code.
- **World vs screen coordinates.** The road is 5000 px long and never moves; the camera just decides which part of it you see (`screen_y = world_y - camera_y`).
- **Observation ≠ reward.** What the agent *sees* (distances to the edges, offset from the lane center) is not the same as how it is *judged*. Giving it "1 = good, 0 = bad" as input would tell it that it failed, but not where or by how much.
- **Reward hacking is real.** Paying the agent for "being centered" would have paid it to stand still. Making per-step penalties bigger than the crash penalty would have taught it to crash on purpose, just to stop losing points.
- **Reward what happened, not what was asked.** Progress is measured from the car's real movement, so pushing against the wall at the bottom earns nothing.
- **Scale matters.** I first chose -2500 / +10000; PPO prefers small numbers, so I divided everything by 500 and kept the same ratios.
- **Learning is not a straight line.** In my run the agent reached the end after only 2,500 steps, then got *worse* at staying centered for a while, and only became perfect after ~15,000 steps. Every run is a bit different, because the network starts from random weights.

### Phase 4.5
- **Physics in seconds, not in steps.** Acceleration, braking, friction and steering are written as "from 0 to top speed in 3 seconds", then converted to steps. Changing the top speed keeps the car feeling the same.
- **True position vs pixels.** With sine and cosine the car moves by fractions of a pixel, but `pygame.Rect` only stores whole numbers. The real position is kept as decimals and the `Rect` follows it. The tests even caught the wall checking the rounded position and letting the car slip half a pixel through it.
- **The agent needs to know its speed and heading.** Two cars in the same spot look identical from the edge distances alone, even if one is stopped and the other is about to leave the road at 30°.
- **Faster car, broken reward.** After the physics update, the agent reached the end in the shortest possible time... driving on the wrong side of the road. At the new top speed, the progress of one step outweighed the lane penalty. The fix: a wrong-way penalty that is always twice the best possible progress per step, computed from the top speed so it can never fall behind again.

## Credits

- Car sprite: [Top Down Car Sprites](https://opengameart.org/node/33247) by Unlucky Studio (OpenGameArt), free for personal and commercial use.

## License

MIT — see [LICENSE](LICENSE).
