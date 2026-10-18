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