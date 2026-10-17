import pytest
from gymnasium.utils.env_checker import check_env

from car_env import CAR_START_X, CAR_START_Y, MAX_STEPS, CarEnv

# Actions are [x, y] with 0/1/2 meaning -1/0/+1
STAY = [1, 1]
FORWARD = [1, 0]
BACK = [1, 2]
LEFT = [0, 1]
RIGHT = [2, 1]


@pytest.fixture
def env():
    # A fresh environment, already reset, for every test
    env = CarEnv()
    env.reset(seed=0)
    return env


def run_until_done(env, action):
    # Repeat the same action until the episode ends; return the last step's results and how many steps it took
    steps = 0
    while True:
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        if terminated or truncated:
            return obs, reward, terminated, truncated, steps


def test_env_follows_gymnasium_api():
    # Official Gymnasium check: spaces, reset/step signatures, observation types and bounds
    check_env(CarEnv(), skip_render_check=True)


def test_reset_puts_car_back_at_start(env):
    env.step(LEFT)
    env.step(FORWARD)
    obs, info = env.reset()
    assert (env.car.left, env.car.top) == (CAR_START_X, CAR_START_Y)
    assert env.step_count == 0
    assert obs[2] == 0 # Lane offset: the car starts centered in the right lane


def test_moving_forward_centered_earns_progress(env):
    obs, reward, terminated, truncated, info = env.step(FORWARD)
    assert reward == pytest.approx(0.01)
    assert not terminated and not truncated


def test_standing_still_earns_nothing(env):
    obs, reward, terminated, truncated, info = env.step(STAY)
    assert reward == 0


@pytest.mark.parametrize("action", [LEFT, RIGHT])
def test_off_road_ends_episode_with_penalty(env, action):
    obs, reward, terminated, truncated, steps = run_until_done(env, action)
    assert terminated and not truncated
    assert reward == pytest.approx(-5, abs=0.1) # -5, plus a tiny lane penalty


def test_reaching_the_end_ends_episode_with_bonus(env):
    obs, reward, terminated, truncated, steps = run_until_done(env, FORWARD)
    assert terminated and not truncated
    assert reward == pytest.approx(20, abs=0.1) # +20, plus the progress of the last step
    assert steps == 990


def test_perfect_run_total_reward(env):
    total = 0
    while True:
        obs, reward, terminated, truncated, info = env.step(FORWARD)
        total += reward
        if terminated or truncated:
            break
    assert total == pytest.approx(29.9, abs=0.01) # 10 for the whole road + 20 for arriving


def test_wall_blocks_the_car_without_ending_the_episode(env):
    for _ in range(10):
        obs, reward, terminated, truncated, info = env.step(BACK)
        assert not terminated
    assert env.car.bottom == env.road.bottom # Pushed against the wall
    assert reward == 0 # Against the wall the car doesn't move, so no progress and no penalty


def test_time_limit_truncates_the_episode(env):
    obs, reward, terminated, truncated, steps = run_until_done(env, STAY)
    assert truncated and not terminated
    assert steps == MAX_STEPS


@pytest.mark.parametrize("action", [LEFT, RIGHT])
def test_observations_stay_in_space_even_off_road(env, action):
    while True:
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs)
        if terminated:
            break
