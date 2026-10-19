import math

POINT_SPACING = 5 # Distance between two points of the center line, in pixels

# Random roads
STRAIGHT_START_LENGTH = 300 # The first and the last piece: a straight road to start and to finish
STRAIGHT_MIN_LENGTH, STRAIGHT_MAX_LENGTH = 200, 800 # Length of the random straight pieces, in pixels
CURVE_MIN_DEGREES, CURVE_MAX_DEGREES = 20, 90 # How much a random curve turns
CURVE_EXTRA_LENGTH = 400 # A curve is between its minimum length and this much longer (longer = wider)

RAY_SEARCH_WINDOW = 3 # A ray moves POINT_SPACING pixels per step: its nearest road point moves at most ~1 point, so a tiny search is enough


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


def nearest_point_index(points, x, y, around=None, window=None):
    # Index of the point closest to (x, y). If `around` and `window` are given,
    # search only from around - window to around + window (much faster)
    nearest_index = 0
    nearest_distance = float('inf') # Start with an infinitely large distance

    if around is not None and window is not None:
        start = max(0, around - window)
        end = min(len(points), around + window + 1) # +1 because the end index is exclusive
        search_points = points[start:end]
    else:
        search_points = points
        start = 0

    for i, (px, py) in enumerate(search_points):
        distance = math.dist((px, py), (x, y)) # Calculate the distance from the point to (x, y)
        if distance < nearest_distance:
            nearest_distance = distance
            nearest_index = start + i

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


def offset_line(points, distance):
    # A line parallel to the center line, `distance` pixels to its right (negative = to its left)
    line = []
    for i, (px, py) in enumerate(points):
        right_dir = math.radians(road_angle(points, i) - 90) # Direction "right" of the road at point i
        x = px + distance * math.cos(right_dir) # Move `distance` pixels in that direction
        y = py - distance * math.sin(right_dir) # (in pygame, going up means y gets smaller)
        line.append((x, y))
    return line

def is_on_road(points, x, y, half_width, around, window):
    # True if (x, y) is on the road: closer than half_width to the center line
    index = nearest_point_index(points, x, y, around, window)
    offset = side_offset(points, index, x, y)
    return abs(offset) <= half_width

def ray_length(points, x, y, angle, max_length, step, half_width, around):
    # How far a ray can go from (x, y) in direction `angle` (degrees) before leaving the road, up to max_length
    distance = 0
    index = around
    while distance < max_length:
        distance += step
        ray_x = x + distance * math.cos(math.radians(angle))
        ray_y = y - distance * math.sin(math.radians(angle)) # (in pygame, going up means y gets smaller)
        index = nearest_point_index(points, ray_x, ray_y, around=index, window=RAY_SEARCH_WINDOW) # Small search near the previous tip

        if abs(side_offset(points, index, ray_x, ray_y)) > half_width:
            return distance # The last step was off the road

    return max_length # The ray reached its maximum length without leaving the road

def round_up_to_spacing(length):
    # The smallest multiple of POINT_SPACING that is at least `length` (build_center_line only uses whole multiples of it)
    return POINT_SPACING * math.ceil(length / POINT_SPACING)


def random_road_pieces(rng, pieces_count, min_radius):
    # A random road: a straight piece first, then a random mix of straights and curves, and a straight piece at the end.
    # Every curve is wide enough to be driven: its radius is at least min_radius
    pieces = [(STRAIGHT_START_LENGTH, 0)] # 1. Always start with a straight piece
    for _ in range(pieces_count - 2):
        if rng.random() < 0.5: # 2. Choose at random: straight or curve? (heads or tails)
            length = int(rng.integers(STRAIGHT_MIN_LENGTH, STRAIGHT_MAX_LENGTH + 1)) # 3a. Straight: a random length
            pieces.append((round_up_to_spacing(length), 0))
        else:
            turn = int(rng.integers(CURVE_MIN_DEGREES, CURVE_MAX_DEGREES + 1)) # 3b. Curve: how much it turns...
            degrees = turn * int(rng.choice([-1, 1])) # ...and to which side (+ = left, - = right)
            min_length = round_up_to_spacing(min_radius * math.radians(turn)) # Shorter than this, the curve would be tighter than min_radius
            length = int(rng.integers(min_length, min_length + CURVE_EXTRA_LENGTH + 1)) # Random length, never shorter than min_length
            pieces.append((round_up_to_spacing(length), degrees))
    pieces.append((STRAIGHT_START_LENGTH, 0)) # 4. Always end with a straight piece
    return pieces
