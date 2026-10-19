import pytest
from gymnasium.utils.env_checker import check_env

from car_env import (
    ACCELERATION, CAR_START_ANGLE, CAR_START_X, CAR_START_Y, MAX_SPEED,
    STEPS_PER_SECOND, CarEnv,
)

# Actions are [steering, pedals]. Steering: 0 = left, 1 = straight, 2 = right. Pedals: 0 = throttle, 1 = none, 2 = brake
THROTTLE = [1, 0]
COAST = [1, 1]
BRAKE = [1, 2]
LEFT_THROTTLE = [0, 0]
RIGHT_THROTTLE = [2, 0]
LEFT_COAST = [0, 1]


STRAIGHT_ROAD = [(5000, 0)] # The rules are tested on a fixed straight road, whatever track the game uses


@pytest.fixture
def env():
    # A fresh environment on the straight road, already reset, for every test
    env = CarEnv(road_pieces=STRAIGHT_ROAD)
    env.reset(seed=0)
    return env


def repeat(env, action, steps):
    # Repeat the same action for a number of steps; return the last step's results
    for _ in range(steps):
        result = env.step(action)
    return result


def run_until_done(env, action):
    # Repeat the same action until the episode ends (the time limit guarantees it does); return the last results and the step count
    steps = 0
    while True:
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        if terminated or truncated:
            return obs, reward, terminated, truncated, steps


# --- Gymnasium API and reset ---

def test_env_follows_gymnasium_api():
    # Official Gymnasium check: spaces, reset/step signatures, observation types and bounds
    check_env(CarEnv(), skip_render_check=True)


def test_reset_puts_car_back_at_start_and_still(env):
    repeat(env, LEFT_THROTTLE, 60)
    obs, info = env.reset()
    assert (env.car.left, env.car.top) == (CAR_START_X, CAR_START_Y)
    assert (env.car_x, env.car_y) == (CAR_START_X, CAR_START_Y)
    assert env.car_angle == CAR_START_ANGLE
    assert env.car_speed == 0
    assert env.step_count == 0
    assert obs[2] == 0 # Lane offset: the car starts centered in the right lane


# --- Speed: throttle, brake, friction ---

def test_car_starts_still_and_does_not_move_without_throttle(env):
    obs, reward, terminated, truncated, info = env.step(COAST)
    assert env.car_speed == 0
    assert (env.car_x, env.car_y) == (CAR_START_X, CAR_START_Y)
    assert reward == 0


def test_throttle_accelerates_gradually_up_to_max_speed(env):
    env.step(THROTTLE)
    assert env.car_speed == pytest.approx(ACCELERATION) # One step: only a little faster
    repeat(env, THROTTLE, 5 * STEPS_PER_SECOND) # 5 seconds of throttle
    assert env.car_speed == MAX_SPEED # Never above the top speed


def test_brake_stops_the_car_and_speed_never_goes_negative(env):
    repeat(env, THROTTLE, 5 * STEPS_PER_SECOND)
    repeat(env, BRAKE, 2 * STEPS_PER_SECOND + 1) # Full brake from top speed: about 2 seconds
    assert env.car_speed == 0
    repeat(env, BRAKE, 10)
    assert env.car_speed == 0 # No reverse gear: braking while stopped does nothing


def test_friction_slows_down_more_gently_than_the_brake(env):
    repeat(env, THROTTLE, 5 * STEPS_PER_SECOND)
    repeat(env, COAST, 2 * STEPS_PER_SECOND) # After 2 seconds without pedals...
    assert 0 < env.car_speed < MAX_SPEED # ...the car is slower, but still moving


# --- Steering ---

def test_car_does_not_turn_when_stopped(env):
    repeat(env, LEFT_COAST, 30)
    assert env.car_angle == CAR_START_ANGLE


def test_steering_left_turns_left(env):
    repeat(env, THROTTLE, 2 * STEPS_PER_SECOND)
    x_before = env.car_x
    repeat(env, LEFT_THROTTLE, 10)
    assert env.car_angle > CAR_START_ANGLE # Angles grow counterclockwise: left = bigger angle
    assert env.car_x < x_before


def test_steering_right_turns_right(env):
    repeat(env, THROTTLE, 2 * STEPS_PER_SECOND)
    x_before = env.car_x
    repeat(env, RIGHT_THROTTLE, 10)
    assert env.car_angle < CAR_START_ANGLE
    assert env.car_x > x_before


# --- Rewards and end of episode ---

def test_moving_forward_centered_earns_progress(env):
    repeat(env, THROTTLE, STEPS_PER_SECOND)
    obs, reward, terminated, truncated, info = env.step(THROTTLE)
    assert reward > 0
    assert not terminated and not truncated


def test_wrong_way_never_pays_off_even_at_top_speed(env):
    env.car_x = env.road.centerx - env.car.width / 2 # Car straddling the center line
    env.car_speed = MAX_SPEED # Top speed: the most progress per step
    obs, reward, terminated, truncated, info = env.step(THROTTLE)
    assert env.car.left < env.road.centerx # Still partly in the opposite lane
    assert reward < 0 # The wrong-way penalty is bigger than the best possible progress


def test_wheels_on_the_grass_cost_a_penalty_but_the_episode_goes_on(env):
    env.car_x = env.road.right - env.car.width + 10 # Right wheels 10 px on the grass, center still on the road
    env.car_speed = MAX_SPEED # Top speed: the most progress per step
    obs, reward, terminated, truncated, info = env.step(THROTTLE)
    assert not terminated # Only the wheels are out: the episode goes on
    assert reward < 0 # The grass penalty is bigger than the best possible progress


@pytest.mark.parametrize("action", [LEFT_THROTTLE, RIGHT_THROTTLE])
def test_off_road_ends_episode_with_penalty(env, action):
    obs, reward, terminated, truncated, steps = run_until_done(env, action)
    assert terminated and not truncated
    assert reward == pytest.approx(-5, abs=0.1) # -5, plus a tiny progress and lane penalty


def test_reaching_the_end_ends_episode_with_bonus(env):
    obs, reward, terminated, truncated, steps = run_until_done(env, THROTTLE)
    assert terminated and not truncated
    assert reward == pytest.approx(20, abs=0.1) # +20, plus the progress of the last step
    assert steps < env.max_steps


def test_perfect_run_total_reward(env):
    total = 0
    while True:
        obs, reward, terminated, truncated, info = env.step(THROTTLE)
        total += reward
        if terminated or truncated:
            break
    assert total == pytest.approx(29.9, abs=0.05) # 10 for the whole road + 20 for arriving


def test_wall_blocks_the_car_without_ending_the_episode(env):
    env.car_angle = 270 # Point the car down, toward the wall
    for _ in range(30):
        obs, reward, terminated, truncated, info = env.step(THROTTLE)
        assert not terminated
    assert env.car.bottom == env.road.bottom # Pushed against the wall
    assert env.car_y == env.car.top # The true position was corrected too, not only the Rect
    assert env.car_speed == 0 # Hitting the wall stops the car


def test_time_limit_truncates_the_episode(env):
    obs, reward, terminated, truncated, steps = run_until_done(env, COAST)
    assert truncated and not terminated
    assert steps == env.max_steps


# --- Observations: [dist_left, dist_right, lane_offset, speed, direction] ---

def test_speed_observation_goes_from_0_to_1(env):
    obs, *_ = env.step(COAST)
    assert obs[3] == 0 # Still
    obs, *_ = repeat(env, THROTTLE, 5 * STEPS_PER_SECOND)
    assert obs[3] == pytest.approx(1) # Top speed


@pytest.mark.parametrize("angle, expected", [
    (90, 0), # Straight up, along the road
    (135, 0.25), # 45° to the left
    (45, -0.25), # 45° to the right
    (450, 0), # A full turn more: still straight
    (-90, -1), # Pointing backward
])
def test_direction_observation_is_relative_to_the_road(env, angle, expected):
    env.car_angle = angle
    assert env.get_obs()[4] == pytest.approx(expected)


@pytest.mark.parametrize("action", [LEFT_THROTTLE, RIGHT_THROTTLE])
def test_observations_stay_in_space_even_off_road(env, action):
    while True:
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs)
        if terminated or truncated:
            break
