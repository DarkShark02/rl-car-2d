import pytest
from gymnasium.utils.env_checker import check_env

from car_env import (
    ACCELERATION, CAR_START_ANGLE, CAR_START_X, CAR_START_Y, MAX_SPEED,
    CAR_WIDTH, OFF_ROAD_PENALTY, ROAD_WIDTH, ROAD_X, STALL_PENALTY, STALL_SECONDS, STEPS_PER_SECOND, RAY_ANGLES, CarEnv,
)

# Actions are [steering, pedals]. Steering: 0 = left, 1 = straight, 2 = right. Pedals: 0 = throttle, 1 = none, 2 = brake
THROTTLE = [1, 0]
COAST = [1, 1]
BRAKE = [1, 2]
LEFT_THROTTLE = [0, 0]
RIGHT_THROTTLE = [2, 0]
LEFT_COAST = [0, 1]


LANE, SPEED, DIRECTION = len(RAY_ANGLES), len(RAY_ANGLES) + 1, len(RAY_ANGLES) + 2 # Positions in the observation, after the rays
ROAD_CENTER_X, ROAD_RIGHT_X = ROAD_X + ROAD_WIDTH / 2, ROAD_X + ROAD_WIDTH # Where the straight test road is
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
    assert obs[LANE] == 0 # Lane offset: the car starts centered in the right lane


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
    env.car_x = ROAD_CENTER_X - env.car.width / 2 # Car straddling the center line
    env.car_speed = MAX_SPEED # Top speed: the most progress per step
    obs, reward, terminated, truncated, info = env.step(THROTTLE)
    assert env.car.left < ROAD_CENTER_X # Still partly in the opposite lane
    assert reward < 0 # The wrong-way penalty is bigger than the best possible progress


def test_wheels_on_the_grass_cost_a_penalty_but_the_episode_goes_on(env):
    env.car_x = ROAD_RIGHT_X - env.car.width + 10 # Right wheels 10 px on the grass, center still on the road
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


def test_driving_back_past_the_start_line_is_off_road(env):
    env.car_angle = 270 # The car turned around: it points down, back toward the start line
    obs, reward, terminated, truncated, steps = run_until_done(env, THROTTLE)
    assert terminated # Behind the start there is no road
    assert reward == pytest.approx(-OFF_ROAD_PENALTY, abs=0.1)
    assert steps < STEPS_PER_SECOND # It happens as soon as the car crosses the start line (30 px behind it)


def test_no_invisible_wall_on_roads_going_down():
    env = CarEnv(road_pieces=[(300, 0), (600, -180), (2000, 0)]) # A U-turn: the road comes back down, below the start
    env.reset(seed=0)
    env.car_x, env.car_y = env.road_points[-100][0] - CAR_WIDTH / 2, env.road_points[-100][1] - 25 # On the way down, below the start line height
    env.car_angle = 270 # Driving down, along the road
    env.road_index = len(env.road_points) - 100
    env.car_speed = MAX_SPEED
    env.step(THROTTLE)
    assert env.car_speed > 0 # Nothing stops the car: the old wall at y = HEIGHT is gone


def test_time_limit_truncates_the_episode(env):
    env.max_steps = 100 # A short time limit, reached while the car is still driving
    obs, reward, terminated, truncated, steps = run_until_done(env, THROTTLE)
    assert truncated and not terminated # Out of time: not a success, not a failure
    assert steps == env.max_steps


def test_standing_still_for_too_long_is_a_failure(env):
    obs, reward, terminated, truncated, steps = run_until_done(env, COAST) # Never pressing the throttle
    assert terminated and not truncated # Stalling ends the episode like going off the road
    assert steps == STALL_SECONDS * STEPS_PER_SECOND
    assert reward == pytest.approx(-STALL_PENALTY, abs=0.05)


def test_driving_resets_the_stall_counter(env):
    repeat(env, COAST, STALL_SECONDS * STEPS_PER_SECOND - 1) # Almost stalled...
    obs, reward, terminated, truncated, info = repeat(env, THROTTLE, STEPS_PER_SECOND) # ...then drive away
    assert not terminated
    assert env.step_noprogress_count < STEPS_PER_SECOND # Moving forward resets the counter


def test_stall_counter_starts_from_zero_at_every_episode(env):
    repeat(env, COAST, STALL_SECONDS * STEPS_PER_SECOND - 10) # Stuck for a while...
    env.reset() # ...new episode
    assert env.step_noprogress_count == 0


# --- Observations: [7 rays, lane_offset, speed, direction] ---

def test_rays_see_the_road_around_the_car(env):
    obs = env.get_obs()
    rays = obs[:len(RAY_ANGLES)]
    straight_ahead = rays[RAY_ANGLES.index(0)]
    right_side = rays[RAY_ANGLES.index(-90)]
    left_side = rays[RAY_ANGLES.index(90)]
    assert straight_ahead == pytest.approx(1) # A long straight road: the forward ray reaches its maximum length
    assert right_side < left_side # The car is in the right lane: the grass is closer on the right
    assert all(0 < ray <= 1 for ray in rays)


def test_rays_turn_with_the_car(env):
    env.car_angle = 0 # The car points right, across the road
    obs = env.get_obs()
    straight_ahead = obs[RAY_ANGLES.index(0)]
    assert straight_ahead < 0.2 # Looking across the road: the grass is close

def test_speed_observation_goes_from_0_to_1(env):
    obs, *_ = env.step(COAST)
    assert obs[SPEED] == 0 # Still
    obs, *_ = repeat(env, THROTTLE, 5 * STEPS_PER_SECOND)
    assert obs[SPEED] == pytest.approx(1) # Top speed


@pytest.mark.parametrize("angle, expected", [
    (90, 0), # Straight up, along the road
    (135, 0.25), # 45° to the left
    (45, -0.25), # 45° to the right
    (450, 0), # A full turn more: still straight
    (-90, -1), # Pointing backward
])
def test_direction_observation_is_relative_to_the_road(env, angle, expected):
    env.car_angle = angle
    assert env.get_obs()[DIRECTION] == pytest.approx(expected)


@pytest.mark.parametrize("action", [LEFT_THROTTLE, RIGHT_THROTTLE])
def test_observations_stay_in_space_even_off_road(env, action):
    while True:
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs)
        if terminated or truncated:
            break
