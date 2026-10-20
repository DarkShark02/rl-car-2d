import pytest
from gymnasium.utils.env_checker import check_env

from car_env import (
    ACCELERATION, CAR_START_ANGLE, CAR_START_X, CAR_START_Y, MAX_SPEED,
    CAR_WIDTH, OFF_ROAD_PENALTY, ROAD_WIDTH, ROAD_X, STALL_PENALTY, STALL_SECONDS, STEPS_PER_SECOND, RAY_ANGLES, CarEnv,
)

# Actions are [steering, pedals]. Steering: 0 = left, 1 = straight, 2 = right. Pedals: 0 = throttle, 1 = none, 2 = brake
THROTTLE = [2, 0] # Steering index 2 = straight (STEER_LEVELS)
COAST = [2, 1]
BRAKE = [2, 2]
LEFT_THROTTLE = [0, 0]
RIGHT_THROTTLE = [4, 0]
LEFT_COAST = [0, 1]


LANE, SPEED, DIRECTION = len(RAY_ANGLES), len(RAY_ANGLES) + 1, len(RAY_ANGLES) + 2 # Positions in the observation, after the rays
ROAD_CENTER_X, ROAD_RIGHT_X = ROAD_X + ROAD_WIDTH / 2, ROAD_X + ROAD_WIDTH # Where the straight test road is
STRAIGHT_ROAD = [(5000, 0)] # The rules are tested on a fixed straight road, whatever track the game uses


@pytest.fixture
def env():
    # A fresh environment on the straight road, already reset, for every test
    env = CarEnv(road_pieces=STRAIGHT_ROAD, start_speed=0) # The rule tests start with the car still
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
    assert reward == pytest.approx(-5, abs=0.3) # -5, plus the lane, grass and wrong-way penalties of that last step


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


def test_running_out_of_time_is_a_failure(env):
    from car_env import TIME_OUT_PENALTY
    env.max_steps = 100 # A short time limit, reached while the car is still driving
    obs, reward, terminated, truncated, steps = run_until_done(env, THROTTLE)
    assert terminated # Out of time: too slow, a failure
    assert steps == env.max_steps
    assert reward == pytest.approx(-TIME_OUT_PENALTY, abs=0.1)


def test_every_end_has_a_reason(env):
    obs, reward, terminated, truncated, info = env.step(THROTTLE)
    assert info["reason"] is None # Still driving
    while True:
        obs, reward, terminated, truncated, info = env.step(LEFT_THROTTLE)
        if terminated or truncated:
            break
    assert info["reason"].startswith("off the route") # Steering left until the grass


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


# --- Traffic lights (city) ---

def city_env_before_a_light(seed=101, color_timer=0, distance=60, speed=0):
    # A city episode with the car in its lane, `distance` px before the first stop line, all lights at the same timer
    from car_env import LANE_WIDTH
    from road import road_angle
    import math
    env = CarEnv(start_speed=0) # The default world is the city
    env.reset(seed=seed)
    stop = env.stop_indices[0]
    index = stop - distance // 5
    x, y = env.road_points[index]
    angle = road_angle(env.road_points, index)
    right = math.radians(angle - 90)
    env.car_x = x + LANE_WIDTH / 2 * math.cos(right) - CAR_WIDTH / 2
    env.car_y = y - LANE_WIDTH / 2 * math.sin(right) - 25
    env.car.left, env.car.top = round(env.car_x), round(env.car_y)
    env.car_angle, env.car_speed, env.road_index = angle, speed, index
    env.light_timers = [color_timer] * len(env.stop_indices)
    return env


def test_city_routes_have_traffic_lights():
    env = CarEnv()
    env.reset(seed=101)
    assert len(env.stop_indices) >= 1
    assert env.stop_indices == sorted(env.stop_indices) # In the order the car meets them


def test_lights_cycle_green_yellow_red():
    from car_env import GREEN_SECONDS, YELLOW_SECONDS, LIGHT_CYCLE_STEPS
    env = CarEnv()
    env.reset(seed=101)
    def color_at(timer):
        env.light_timers = [timer]
        env.stop_indices = env.stop_indices[:1]
        return env.light_colors()[0]
    assert color_at(0) == "green"
    assert color_at(int(GREEN_SECONDS * STEPS_PER_SECOND)) == "yellow"
    assert color_at(int((GREEN_SECONDS + YELLOW_SECONDS) * STEPS_PER_SECOND)) == "red"
    assert color_at(LIGHT_CYCLE_STEPS - 1) == "red"


def test_crossing_on_red_ends_the_episode():
    from car_env import RED_LIGHT_PENALTY, LIGHT_CYCLE_STEPS
    env = city_env_before_a_light(color_timer=LIGHT_CYCLE_STEPS - 200, speed=MAX_SPEED) # Red for a while
    obs, reward, terminated, truncated, steps = run_until_done(env, THROTTLE)
    assert terminated and steps < 20 # It crossed the stop line almost at once
    assert reward == pytest.approx(-RED_LIGHT_PENALTY, abs=0.2)


def test_crossing_on_green_is_fine():
    env = city_env_before_a_light(color_timer=0, speed=MAX_SPEED) # Green for 4 seconds
    for _ in range(20):
        obs, reward, terminated, truncated, info = env.step(THROTTLE)
        assert not terminated


def test_waiting_at_a_red_light_is_not_a_stall():
    from car_env import LIGHT_CYCLE_STEPS
    env = city_env_before_a_light(color_timer=LIGHT_CYCLE_STEPS - 200, distance=40, speed=0) # Stopped before a red light
    for _ in range(STALL_SECONDS * STEPS_PER_SECOND + 10): # Longer than the stall limit, but the light is still red
        obs, reward, terminated, truncated, info = env.step(BRAKE)
        if env.light_colors()[0] != "red":
            break
        assert not terminated


def test_light_observation_is_one_hot():
    from car_env import LIGHT_CYCLE_STEPS
    env = city_env_before_a_light(color_timer=LIGHT_CYCLE_STEPS - 200) # Red
    obs = env.get_obs()
    assert obs.shape == (15,)
    assert list(obs[11:14]) == [0, 0, 1] # green, yellow, red
    assert 0 < obs[10] < 1 # The stop line is close, within the rays' reach


def test_city_without_traffic_lights():
    env = CarEnv(traffic_lights=False)
    obs, info = env.reset(seed=101)
    assert env.stop_indices == []
    assert list(obs[10:15]) == [1, 1, 0, 0, 1] # No light ahead: far away, green, and no change coming


def test_yellow_far_from_the_line_means_stop():
    from car_env import GREEN_SECONDS, RED_LIGHT_PENALTY
    turns_yellow = int(GREEN_SECONDS * STEPS_PER_SECOND)
    slow_far = city_env_before_a_light(color_timer=turns_yellow - 1, distance=500, speed=2) # Slow and far when it turns yellow: room to brake
    slow_far.step(COAST)
    assert slow_far.must_stop_at_yellow[0] # It could stop, so it must

    env = city_env_before_a_light(color_timer=turns_yellow + 10, distance=60, speed=MAX_SPEED) # Yellow (for 2 more seconds), right before the line
    env.must_stop_at_yellow = [True] * len(env.stop_indices) # ...a yellow it could have stopped for
    obs, reward, terminated, truncated, steps = run_until_done(env, THROTTLE)
    assert terminated and steps < 20 # It crossed while the light was still yellow
    assert reward == pytest.approx(-RED_LIGHT_PENALTY, abs=0.3)


def test_yellow_too_close_to_stop_may_go_on():
    from car_env import GREEN_SECONDS
    turns_yellow = int(GREEN_SECONDS * STEPS_PER_SECOND)
    env = city_env_before_a_light(color_timer=turns_yellow - 1, distance=80, speed=MAX_SPEED) # Fast and close: it can't stop in time
    for _ in range(20):
        obs, reward, terminated, truncated, info = env.step(THROTTLE)
        assert not terminated # Crossing on that yellow is allowed
    assert env.must_stop_at_yellow[0] is False


def test_stopping_far_from_a_red_light_is_not_a_stall_but_costs():
    from car_env import LIGHT_CYCLE_STEPS
    env = city_env_before_a_light(color_timer=LIGHT_CYCLE_STEPS - 230, distance=300, speed=0) # Red for almost 4 s, stopped 300 px away
    total = 0
    for _ in range(STALL_SECONDS * STEPS_PER_SECOND + 10): # Longer than the stall limit: no stall, it must not be forced to move on red
        obs, reward, terminated, truncated, info = env.step(BRAKE)
        total += reward
        assert not terminated
    assert total < -1 # But waiting that far away costs something


def test_waiting_costs_more_with_the_nose_far_from_the_line():
    from car_env import LIGHT_CYCLE_STEPS
    near = city_env_before_a_light(color_timer=LIGHT_CYCLE_STEPS - 200, distance=150, speed=0) # Both outside the wait zone
    far = city_env_before_a_light(color_timer=LIGHT_CYCLE_STEPS - 200, distance=300, speed=0)
    near_reward = near.step(BRAKE)[1]
    far_reward = far.step(BRAKE)[1]
    assert far_reward < near_reward < 0


def test_stopping_in_the_wait_zone_pays_once_per_light():
    from car_env import LIGHT_CYCLE_STEPS, STOP_IN_ZONE_REWARD
    env = city_env_before_a_light(color_timer=LIGHT_CYCLE_STEPS - 200, distance=40, speed=0) # Red, nose 15 px from the line
    first = env.step(BRAKE)[1]
    second = env.step(BRAKE)[1]
    assert first == pytest.approx(STOP_IN_ZONE_REWARD, abs=0.1) # The reward, once...
    assert abs(second) < 0.1 # ...and then nothing more for waiting there


def test_driving_at_normal_speed_towards_a_red_light_costs_nothing_extra():
    from car_env import LIGHT_CYCLE_STEPS
    red = city_env_before_a_light(color_timer=LIGHT_CYCLE_STEPS - 200, distance=500, speed=5) # Far enough to brake, at a normal speed: braking early isn't needed
    green = city_env_before_a_light(color_timer=0, distance=500, speed=5)
    assert red.step(COAST)[1] == pytest.approx(green.step(COAST)[1])


def test_lab_is_a_straight_road_with_traffic_lights():
    from car_env import LAB_LENGTH, LAB_MIN_LIGHT_GAP, LAB_MAX_LIGHT_GAP
    env = CarEnv(world="lab")
    for seed in range(5):
        env.reset(seed=seed)
        assert env.road_length == LAB_LENGTH
        gaps = [b - a for a, b in zip([0] + env.stop_indices, env.stop_indices)]
        assert len(env.stop_indices) >= 3
        assert all(LAB_MIN_LIGHT_GAP <= gap * 5 <= LAB_MAX_LIGHT_GAP for gap in gaps)


def test_light_observation_is_the_gap_with_detail_near_the_line():
    from car_env import LIGHT_CYCLE_STEPS
    near = city_env_before_a_light(color_timer=LIGHT_CYCLE_STEPS - 200, distance=45, speed=0).get_obs()[10] # Nose 20 px from the line
    far = city_env_before_a_light(color_timer=LIGHT_CYCLE_STEPS - 200, distance=300, speed=0).get_obs()[10]
    assert 0.1 < near < far < 1 # Close to the line it is still clearly more than 0


def test_countdown_is_hidden_on_red():
    from car_env import LIGHT_CYCLE_STEPS
    env = city_env_before_a_light(color_timer=LIGHT_CYCLE_STEPS - 30) # Red, half a second left
    assert env.get_obs()[14] == 1.0


def test_episodes_start_with_the_car_already_moving():
    from car_env import CAR_START_SPEED
    env = CarEnv()
    env.reset(seed=0)
    assert env.car_speed == CAR_START_SPEED > 0


def test_light_countdown_observation():
    from car_env import GREEN_SECONDS
    env = city_env_before_a_light(color_timer=0) # Green, just turned on
    full = env.get_obs()[14]
    env.light_timers = [int(GREEN_SECONDS * STEPS_PER_SECOND) - 30] * len(env.stop_indices) # Green, half a second left
    almost = env.get_obs()[14]
    assert 0 < almost < full <= 1 # The countdown goes down as the change gets closer


def test_approaching_a_red_light_too_fast_costs_more_the_faster_it_goes(monkeypatch):
    import car_env
    from car_env import LIGHT_CYCLE_STEPS
    monkeypatch.setattr(car_env, "UNDERSPEED_PENALTY", 0) # Only the too-fast side of the braking curve is measured here
    def extra_cost(distance, speed): # What the red light costs, compared to the same step on green
        red = city_env_before_a_light(color_timer=LIGHT_CYCLE_STEPS - 200, distance=distance, speed=speed)
        green = city_env_before_a_light(color_timer=0, distance=distance, speed=speed)
        return green.step(COAST)[1] - red.step(COAST)[1]
    assert extra_cost(400, 2) == pytest.approx(0) # Far and slow: below the braking curve, nothing to pay
    assert 0 < extra_cost(150, 4) < extra_cost(150, MAX_SPEED) # Above the curve: the faster, the more it costs


def test_stopping_short_of_the_wait_zone_costs_more_the_farther_it_is():
    from car_env import LIGHT_CYCLE_STEPS
    def cost(distance): # Stopped at a red light, `distance` px before the line
        return -city_env_before_a_light(color_timer=LIGHT_CYCLE_STEPS - 200, distance=distance, speed=0).step(BRAKE)[1]
    assert cost(300) > cost(110) > 0


def test_weaving_costs_something(env):
    straight = env.step(COAST)[1]
    env.reset(seed=0)
    env.car_angle += 20 # Same place, pointing 20° to the side
    assert env.step(COAST)[1] < straight


def test_crossing_on_red_is_worse_than_stalling():
    from car_env import RED_LIGHT_PENALTY, STALL_PENALTY, OFF_ROAD_PENALTY
    assert RED_LIGHT_PENALTY > max(STALL_PENALTY, OFF_ROAD_PENALTY) # Otherwise a car stopped far from the line may as well run the red


def test_small_steering_turns_less_than_full_steering(env):
    small, full = CarEnv(road_pieces=STRAIGHT_ROAD, start_speed=0), CarEnv(road_pieces=STRAIGHT_ROAD, start_speed=0)
    small.reset(seed=0); full.reset(seed=0)
    small.car_speed = full.car_speed = MAX_SPEED
    start = small.car_angle
    small.step([1, 1]) # A little left
    full.step([0, 1]) # Full left
    assert 0 < small.car_angle - start < full.car_angle - start # Left = the angle grows, less with a little steering
