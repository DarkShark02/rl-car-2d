from collections import deque

from road import round_up_to_spacing


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
