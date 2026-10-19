import numpy as np

from city import find_route, neighbors, random_street_positions
from road import POINT_SPACING


def test_street_positions_start_at_zero_with_gaps_in_range():
    rng = np.random.default_rng(0)
    for _ in range(20):
        positions = random_street_positions(rng, 5, 400, 900)
        assert len(positions) == 5
        assert positions[0] == 0
        for a, b in zip(positions, positions[1:]):
            assert 400 <= b - a <= 900 + POINT_SPACING # A random gap (rounded up to the spacing)
            assert (b - a) % POINT_SPACING == 0


def test_neighbors_stay_inside_the_grid():
    assert sorted(neighbors((1, 2), 3, 4)) == [(0, 2), (1, 1), (1, 3), (2, 2)] # In the middle: 4 neighbors
    assert sorted(neighbors((0, 0), 3, 4)) == [(0, 1), (1, 0)] # Corner: 2
    assert sorted(neighbors((0, 2), 3, 4)) == [(0, 1), (0, 3), (1, 2)] # Border: 3


def test_route_is_a_shortest_path_of_neighboring_intersections():
    rng = np.random.default_rng(0)
    for _ in range(50):
        start = (int(rng.integers(0, 5)), int(rng.integers(0, 5)))
        goal = (int(rng.integers(0, 5)), int(rng.integers(0, 5)))
        route = find_route(start, goal, 5, 5, rng)
        assert route[0] == start and route[-1] == goal
        for a, b in zip(route, route[1:]):
            assert b in neighbors(a, 5, 5) # Every step goes to a neighboring intersection
        manhattan = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
        assert len(route) - 1 == manhattan # As short as possible: no detours
        assert len(set(route)) == len(route) # Never the same intersection twice


def test_routes_are_varied():
    rng = np.random.default_rng(0)
    routes = {tuple(find_route((0, 0), (3, 3), 5, 5, rng)) for _ in range(30)}
    assert len(routes) > 5 # Many different shortest routes between the same two corners


def test_random_city_routes_are_drivable_roads():
    from city import TURN_RADIUS, random_city_pieces
    from car_env import MIN_CURVE_RADIUS, LANE_WIDTH
    rng = np.random.default_rng(0)
    for _ in range(50):
        pieces = random_city_pieces(rng)
        assert pieces[0][1] == 0 and pieces[-1][1] == 0 # Straight at the start and at the end
        assert all(length >= 0 for length, _ in pieces)
        assert all(degrees in (0, 90, -90) for _, degrees in pieces) # City: only straights and 90° turns
    assert TURN_RADIUS - LANE_WIDTH / 2 >= MIN_CURVE_RADIUS # In the right lane, the car turns on a circle it can drive


def test_random_city_routes_always_turn():
    from city import random_city_pieces
    rng = np.random.default_rng(1)
    for _ in range(200):
        pieces = random_city_pieces(rng)
        assert any(degrees != 0 for _, degrees in pieces) # Never a straight route
