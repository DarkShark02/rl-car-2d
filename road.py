import math

POINT_SPACING = 5 # Distance between two points of the center line, in pixels


def build_center_line(pieces, start_x, start_y, start_angle):
    # pieces: list of (length, degrees). Returns the list of (x, y) points of the road's center line
    x, y, angle = start_x, start_y, start_angle # The "pen": where it is and where it points
    points = [(x, y)] # The starting point is the first point of the road

    for length, degrees in pieces:
        steps = length // POINT_SPACING # How many points this piece creates
        turn = degrees / steps if steps > 0 else 0 # How many degrees the pen turns at each point
        for _ in range(steps):
            angle += turn
            x += POINT_SPACING * math.cos(math.radians(angle)) # Move POINT_SPACING pixels in the pen's direction
            y -= POINT_SPACING * math.sin(math.radians(angle))
            points.append((x, y))

    return points


def nearest_point_index(points, x, y):
    # Index of the point of the center line closest to (x, y)
    nearest_index = 0
    nearest_distance = float('inf') # Start with an infinitely large distance

    for i, (px, py) in enumerate(points):
        distance = math.dist((px, py), (x, y)) # Calculate the distance from the point to (x, y)
        if distance < nearest_distance:
            nearest_distance = distance
            nearest_index = i

    return nearest_index


def road_angle(points, i):
    # Direction of the road at point i, in degrees (0° = right, counterclockwise)
    j = min(i, len(points) - 2) # Use the piece from point j to j+1; at the last point, use the piece that leads to it
    x1, y1 = points[j]
    x2, y2 = points[j + 1]
    right = x2 - x1 # How much the road moves to the right
    up = y1 - y2 # How much it moves up (in pygame, going up means y gets smaller)
    return math.degrees(math.atan2(up, right))


def side_offset(points, i, x, y):
    # How far (x, y) is from the center line at point i: positive = right of the road, negative = left
    right_dir = math.radians(road_angle(points, i) - 90) # The road's right side is 90° clockwise from the road's direction
    px, py = points[i]
    dx = x - px # Arrow from the center line to (x, y): how much it goes right...
    dy_up = py - y # ...and how much it goes up
    return dx * math.cos(right_dir) + dy_up * math.sin(right_dir) # Projection ("shadow") of the arrow on the right direction
