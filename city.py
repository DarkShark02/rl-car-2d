import math
from collections import deque

from road import round_up_to_spacing

DIRECTION_ANGLES = {(-1, 0): 90, (1, 0): 270, (0, 1): 0, (0, -1): 180} # (Δrow, Δcolumn) -> angle: up, down, right, left
TURN_RADIUS = 200 # Radius of the center line in a 90° turn (the car, in the right lane, turns on 155 px: wider than MIN_CURVE_RADIUS)
DIRECTION_ANGLES = {(-1, 0): 90, (1, 0): 270, (0, 1): 0, (0, -1): 180} # (Δrow, Δcolumn) -> angle: up, down, right, left

def random_street_positions(rng, streets_count, min_gap, max_gap):
    # Positions of the streets along one axis: the first at 0, then each one a random gap after the previous
    first_street = 0
    positions = [first_street]
    for _ in range(streets_count - 1):
        gap = round_up_to_spacing(int(rng.integers(min_gap, max_gap + 1))) # Random gap, rounded to a multiple of POINT_SPACING
        positions.append(positions[-1] + gap)
    return positions


def neighbors(intersection, rows, columns):
    # The intersections directly connected to this one: up, left, down, right (only those inside the grid)
    row, column = intersection
    candidates = [(row - 1, column), (row, column - 1), (row + 1, column), (row, column + 1)] # Up, left, down, right
    return [(r, c) for r, c in candidates if 0 <= r < rows and 0 <= c < columns] # Keep only those inside the grid


def find_route(start, goal, rows, columns, rng):
    # A shortest route from start to goal (a list of intersections), choosing at random among the many shortest ones
    queue = deque([start]) # Intersections to explore, closest to the start first
    came_from = {start: None} # For every intersection reached: the intersection we came from

    while queue: # An empty deque counts as False: stop when there is nothing left to explore
        current = queue.popleft()
        if current == goal:
            break

        current_neighbors = neighbors(current, rows, columns)
        rng.shuffle(current_neighbors) # Random order: a different shortest route every time
        for neighbor in current_neighbors:
            if neighbor not in came_from: # Not reached yet
                came_from[neighbor] = current
                queue.append(neighbor)

    route = [] # Walk back from the goal to the start, following came_from
    intersection = goal
    while intersection is not None:
        route.append(intersection)
        intersection = came_from[intersection]
    route.reverse() # It was built from the goal backward
    return route


def route_to_pieces(route, xs, ys):
    angles = [] # The direction of every step of the route
    lengths = [] # The length in pixels of every step
    for (row_a, col_a), (row_b, col_b) in zip(route, route[1:]):
        angles.append(DIRECTION_ANGLES[(row_b - row_a, col_b - col_a)]) # The difference between the two intersections tells the direction
        lengths.append(abs(xs[col_b] - xs[col_a]) + abs(ys[row_b] - ys[row_a])) # Distance between the two intersections

    turns = [] # The turn at every intersection in the middle of the route: 0, +90 (left) or -90 (right)
    for in_angle, out_angle in zip(angles, angles[1:]):
        turns.append((out_angle - in_angle + 180) % 360 - 180)

    arc_length = round_up_to_spacing(TURN_RADIUS * math.pi / 2) # Length of a quarter of a circle

    pieces = []
    straight = lengths[0] # The straight piece we are building
    for turn, next_length in zip(turns, lengths[1:]):
        if turn == 0:
            straight += next_length
        else:
            pieces.append((straight - TURN_RADIUS, 0))
            pieces.append((arc_length, turn))
            straight = next_length - TURN_RADIUS
    pieces.append((straight, 0)) # The last straight piece
    return pieces



# A random city
CITY_STREETS = 5 # Streets in each direction: 5 x 5 = 25 intersections
CITY_MIN_GAP, CITY_MAX_GAP = 600, 1000 # Distance between two parallel streets: two turns of TURN_RADIUS take 400, the rest is a straight piece with its center line
CITY_MIN_ROUTE_STEPS = 4 # Start and goal at least this many blocks apart, so every route has some turns


def random_city(rng):
    # A new random city and a route through it: street positions, the route, its road pieces and where it starts
    xs = random_street_positions(rng, CITY_STREETS, CITY_MIN_GAP, CITY_MAX_GAP) # x of the vertical streets
    ys = random_street_positions(rng, CITY_STREETS, CITY_MIN_GAP, CITY_MAX_GAP) # y of the horizontal streets
    while True: # Pick start and goal far enough apart, and never on the same street (that would be a straight route)
        start = (int(rng.integers(CITY_STREETS)), int(rng.integers(CITY_STREETS)))
        goal = (int(rng.integers(CITY_STREETS)), int(rng.integers(CITY_STREETS)))
        rows_apart, columns_apart = abs(start[0] - goal[0]), abs(start[1] - goal[1])
        if rows_apart >= 1 and columns_apart >= 1 and rows_apart + columns_apart >= CITY_MIN_ROUTE_STEPS: # Different row AND column: at least one turn
            break
    route = find_route(start, goal, CITY_STREETS, CITY_STREETS, rng)
    first_step = (route[1][0] - route[0][0], route[1][1] - route[0][1])
    return {
        "xs": xs,
        "ys": ys,
        "route": route,
        "pieces": route_to_pieces(route, xs, ys),
        "start": (xs[start[1]], ys[start[0]], DIRECTION_ANGLES[first_step]), # The route starts at the start intersection, heading to the next one
    }


def random_city_pieces(rng):
    # Only the road pieces of a random city route
    return random_city(rng)["pieces"]
