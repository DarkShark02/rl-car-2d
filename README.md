# rl-car-2d 🚗

A car in a 2D world that **learns to drive on its own** through *reinforcement learning*: nobody teaches it the rules — it tries, fails, and improves through rewards and penalties.

The end goal: stay on the road, stop at red lights, obey traffic signs, and avoid crashes.

> 🚧 Work in progress: I'm building this step by step to learn machine learning.

## Training progress

### Phase 5 — curves and random roads
A new random road every episode, seen through 7 ray sensors. The agent learns to brake before curves and to keep its lane on roads it has never seen (3x speed).


https://github.com/user-attachments/assets/23e8d47e-54ea-4f6f-97db-24287d0207a9


### Phase 4.5 — real car physics
The agent learns to accelerate, steer and stay in its own lane (3x speed).


https://github.com/user-attachments/assets/5dac57af-114c-4ab1-9fa4-d5417ba433e7


### Phase 4 — first steps on a straight road
From a random agent to a perfect run on a straight road (3x speed).


https://github.com/user-attachments/assets/b7041657-cf57-4038-a089-1e70d30efc86


## How it works

- **Environment**: a two-lane road with curves, seen from above and drawn with Pygame. Every episode the road is generated at random (16 straight and curved pieces, 5000–7300 pixels long), so the agent can't learn one track by heart. The camera follows the car. The simulation runs without a window during training.
- **Observations**: ten numbers, normalized to -1..1:
  - 7 **ray sensors**: from the car, at −90°, −45°, −20°, 0°, +20°, +45°, +90° from its nose, each one measuring how far the road goes before the grass (up to a bit more than the braking distance);
  - offset from the center of the right lane, speed, and direction of the car relative to the road.
- **Actions**: steering (left / straight / right) and pedals (throttle / none / brake). The car has real momentum: it accelerates gradually, slows down by itself, and only turns while moving. No reverse gear yet.
- **Rewards**:
  - \+ for real progress along the road (the same amount per pixel on any road)
  - − for drifting away from the center of the right lane
  - − for every step on the wrong side of the road, or with the wheels on the grass (always more than the best possible progress, so it never pays off)
  - −5 for leaving the road, driving back past the start line, or standing still for 3 seconds (episode ends)
  - +20 for reaching the finish line (episode ends)
- **Algorithm**: PPO, via Stable-Baselines3.

The goal is to grow this into a full world with intersections, traffic lights, signs, other cars and pedestrians.

## Roadmap

- [x] Phase 1: window, road, and a car you can drive with the keyboard
- [x] Phase 2: off-road detection + camera that follows the car
- [x] Phase 3: sensors (distances to the road edges and lane center; replaced by ray sensors in phase 5)
- [x] Phase 4: Gymnasium environment + first training run (stay on the road)
  - [x] Gymnasium environment (`car_env.py`)
  - [x] First PPO training: the agent reaches the end of the road, centered in its lane
  - [x] Keyboard mode built on the same environment
  - [x] Video of the training progress
- [x] Phase 4.5: car physics — speed with throttle and brake, steering, a real car sprite
- [x] Phase 5: curves — a road made of pieces, real ray sensors, progress measured along the road
  - [x] Road described by its center line (a list of points), drawn with exact polygons
  - [x] 7 ray sensors
  - [x] A new random road every episode, plus a fixed test road never used in training
  - [x] The agent finishes 30 out of 30 roads it has never seen, without ever driving on the wrong side
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
python train.py    # train the agent with PPO on random roads (1M steps, ~10 minutes), saving a checkpoint every 50,000 steps
python watch.py    # watch the latest trained agent drive on random roads (optional: path to another model)
python record.py   # make a ~30 s video of the latest run: videos/<run>_progress.mp4
```

Trained models are saved in `models/`, which is not tracked by git. Every training run gets a new name (`ppo_car_v1`, `ppo_car_v2`, ...). Models trained before a change to the observations can't be loaded by the new environment: `watch.py` and `record.py` tell you when that happens.

### Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

The tests check the road geometry, the ray sensors, the random road generator and the environment rules (rewards, end of episode, stalling, time limit), plus the Gymnasium API, in a couple of seconds.

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

### Phase 5
- **A curved road is just a list of points.** Many short straight pieces, one every 5 pixels, along the center of the road. The road is generated like a pen moving and turning, exactly like the car. From the nearest point you get everything: how far along the road the car is, which way the road goes, and how far the car is from the center.
- **Left and right depend on the road, not the screen.** A car *below* a road going to the right is on its *right* side. A projection (sine and cosine again) gives the signed distance from the center line in any direction.
- **Some curves are physically impossible.** Because the car turns by a fixed angle per pixel traveled, it always drives the same circle at full steering, at any speed. My first test road had curves tighter than that circle: no driver could make them. Now the random roads never go below 1.5 times the car's turning radius.
- **Rays must turn with the car.** They are measured from the car's nose, so a curve to the right always looks the same (long rays on the right, short on the left), wherever the road goes on the screen. That's what lets the agent learn a rule instead of a map.
- **Make it work, then make it fast.** The first ray sensors made the simulation 8 times slower. The fix was to search the nearest road point only near the previous one: from 324 to about 2,600 steps per second.
- **One track = overfitting.** Trained on a single road, the agent could learn "turn right after 40 steps" instead of how to drive. So every episode now has a new random road, and the agent is tested on a road it has never seen.
- **Doing nothing can be the best strategy.** On random roads, my first agent learned to... never move. Driving without knowing how to steer meant crashing at the first curve (−5), standing still meant 0. It got stuck in this local minimum for a million steps. Two fixes: standing still for 3 seconds now counts as a failure, and an entropy bonus keeps the agent trying new actions.
- **Look at the failures, not only at the score.** The next agent finished only 19 roads out of 30. Watching the failed episodes showed the car stopped in the middle of the road with the throttle pressed: an old invisible wall from the straight road, which now cut every road going down. Without it, the same agent finished 30 out of 30.

## How this project is built

This is my project for learning reinforcement learning: I chose what to build,
planned the phases, and made the design decisions along the way. Many of the
key changes came from looking at the results and asking "why?": replacing the
endless curves with a real city grid, finding routes with BFS, noticing that the
agent was overfitting a single road, spotting routes and goals that didn't make sense.

I wrote the core logic myself — the game loop, the car physics, the environment
rules and rewards, the road and city route generators — step by step, with an
AI assistant (Claude, by Anthropic) as a tutor: it explained the concepts, asked
guiding questions, reviewed my code and pointed out bugs.

To keep the focus on learning machine learning, I asked it to take care of the
supporting parts: the tests, the training and video scripts, most of the drawing
code, tuning values, and drafting this README. When I got stuck, it also fixed
or completed parts of the logic, explaining each change.

Every lesson in "What I learned" comes from problems I actually ran into.

## Credits

- Car sprite: [Top Down Car Sprites](https://opengameart.org/node/33247) by Unlucky Studio (OpenGameArt), free for personal and commercial use.

## License

MIT — see [LICENSE](LICENSE).
