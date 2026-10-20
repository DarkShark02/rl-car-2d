import pygame
import numpy as np
import gymnasium as gym
import math
import os
from city import TURN_RADIUS, random_city
from road import along_offset, build_center_line, nearest_point_index, offset_line, POINT_SPACING, random_road_pieces, ray_length, road_angle, side_offset

# World constants: the single source of truth, also imported by main.py
WIDTH, HEIGHT = 800, 600 # Set the width and height of the window
CAR_WIDTH = 30 # Define the width of the car
CAR_HEIGHT = 50 # Define the height of the car
LANE_WIDTH = CAR_WIDTH * 3 # Define the width of a lane, triple than the car
ROAD_WIDTH = LANE_WIDTH * 2 # Define the width of the road  
RIGHT_LANE_CENTER = WIDTH // 2 + LANE_WIDTH // 2 # Center of the right lane
LEFT_LANE_CENTER = WIDTH // 2 - LANE_WIDTH // 2 # Center of the left lane
ROAD_X = WIDTH // 2 - ROAD_WIDTH // 2 # Left edge of the road, centered horizontally
ROAD_ANGLE = 90 # The road is vertical, pointing up (0° = right, angles grow counterclockwise)
CAR_START_X = RIGHT_LANE_CENTER - CAR_WIDTH // 2 # Starting x position of the car, centered in the right lane
CAR_BOTTOM_MARGIN = 5 # Offset for the car's position from the bottom of the window
CAR_START_Y = HEIGHT - (CAR_HEIGHT + CAR_BOTTOM_MARGIN) # Starting y position of the car, at the bottom
CAR_START_ANGLE = ROAD_ANGLE # Starting angle of the car, aligned with the road
CAR_START_AHEAD = HEIGHT - CAR_START_Y - CAR_HEIGHT // 2 # How far after the start line the car's center starts (30 px)
ROAD_PIECES = [(300, 0), (400, -90), (500, 0), (400, 90), (1000, 0)] # A fixed test road, never used in training: to check that the agent can drive a road it has never seen
ROAD_PIECES_COUNT = 16 # How many pieces a random road has (about 6000-8000 pixels)
RAY_SEARCH_WINDOW = 5 # A ray moves POINT_SPACING pixels per step: its nearest road point moves at most ~1 point, so a tiny search is enough

# Car physics: every value is expressed in seconds, then converted to steps (the game runs at 60 steps per second)
STEPS_PER_SECOND = 60 # Simulation steps in one second of game time
MAX_SPEED = 8 # Top speed, in pixels per step
CAR_START_SPEED = MAX_SPEED / 2 # The car starts already moving: with random actions (brake and friction stronger than the throttle) a car starting still never moves, and the agent never discovers that driving pays
ACCELERATION = MAX_SPEED / (3 * STEPS_PER_SECOND) # Full throttle: from 0 to MAX_SPEED in 3 seconds
BRAKE = MAX_SPEED / (2 * STEPS_PER_SECOND) # Full brake: from MAX_SPEED to 0 in 2 seconds
FRICTION = MAX_SPEED / (6 * STEPS_PER_SECOND) # No pedals: from MAX_SPEED to 0 in 6 seconds
CAR_TURN_RADIUS = 2 * CAR_HEIGHT # Tightest circle the car can drive, steering fully: about twice its length (pixels)
STEER_RATE = math.degrees(1 / CAR_TURN_RADIUS) # Degrees turned per pixel traveled: the same circle at any speed
STEER_LEVELS = [-1, -0.25, 0, 0.25, 1] # Steering actions: full left, a little left, straight, a little right, full right.
# With only full left / straight / full right one step turned the car 4.6° at top speed: it could only zigzag to stay in the lane
MIN_CURVE_RADIUS = 1.5 * CAR_TURN_RADIUS # Random curves are never tighter than this: the car can drive them with some margin
TIME_LIMIT_FACTOR = 4 # Time limit of an episode: 4 times the steps needed at top speed (plus the traffic lights). It's not a race: only a car that crawls along runs out of time
BRAKING_DISTANCE = (MAX_SPEED / 2) * (MAX_SPEED / BRAKE) # Average speed while braking × steps needed to stop from top speed

# Sensors: rays that measure how far the road goes around the car
RAY_LENGTH = 1.25 * BRAKING_DISTANCE # The car sees a bit farther than it needs to stop
RAY_ANGLES = [-90, -45, -20, 0, 20, 45, 90] # Angles of the rays the car uses to "see" the road, relative to the car's direction (degrees, counterclockwise)
RAY_STEP = 5 # How many pixels the rays advance at each step, when checking if they hit the road
CAR_SEARCH_WINDOW = 5 # The car moves at most MAX_SPEED (8) pixels per step, less than 2 points: searching 5 points around its last index is enough

# Rewards and penalties: the single source of truth, also imported by test_car_env.py
OFF_ROAD_PENALTY = 5 # Penalty for going off the road
FINISH_REWARD = 20 # Reward for reaching the end of the road
PROGRESS_REWARD_PER_PIXEL = 10 / 5000 # Reward for each pixel of road driven (5000 px = 10 points), the same on any road
MAX_PROGRESS_PER_STEP = MAX_SPEED * PROGRESS_REWARD_PER_PIXEL # Progress reward of one step at top speed
WRONG_WAY_PENALTY = 2 * MAX_PROGRESS_PER_STEP # Each step in the opposite lane costs twice the best progress: wrong way never pays off
GRASS_PENALTY = 2 * MAX_PROGRESS_PER_STEP # Each step with wheels on the grass costs twice the best progress
STALL_SECONDS = 3 # If the car is stuck for this many seconds, the episode ends
STALL_PENALTY = OFF_ROAD_PENALTY # Being stuck counts as a failure, like going off the road
TIME_OUT_PENALTY = OFF_ROAD_PENALTY # Running out of time (crawling) is a failure too: otherwise driving very slowly would be a free, risk-free strategy
RED_LIGHT_PENALTY = 2 * OFF_ROAD_PENALTY # Crossing a stop line on red (or on a yellow it could stop for): the worst mistake. Worse than a stall, so gambling on the light never pays
LIGHT_WAIT_GAP = CAR_HEIGHT # The wait zone: the car's nose between the stop line and this far from it (one car length)
STOP_IN_ZONE_REWARD = 2 # Once per light: coming to a full stop inside the wait zone at a light it must stop for. ~1/10 of the finish: felt, but not worth hunting for
COMFORT_BRAKE = 0.75 * BRAKE # Braking curve towards a light it must stop for: braking with 3/4 of the full brake, to keep some margin
OVERSPEED_PENALTY = 0.08 # Per step, times (speed - curve speed) / MAX_SPEED: too fast to stop gently in the middle of the wait zone.
# A signal at every step that says how to brake (2 px/step too fast ≈ 0.02, like a step of progress), instead of only the -10 at the line
UNDERSPEED_PENALTY = 0.08 # Per step, before the wait zone, times (half the curve speed - speed) / MAX_SPEED: too slow, it stops short of the zone.
# The other side of the curve: stopped 220 px away ≈ 0.022 per step, 80 px away ≈ 0.012 (ppo_car_v1_lab stopped at 55–110 px and never earned the zone reward)
HEADING_PENALTY = 0.02 # Per step, times the angle between the car and the road / 45°: weaving means always pointing a bit sideways (ppo_car_v1_lab weaved up to 29° near the lights)
LANE_PENALTY = 0.03 # Per step, with the car's center on the lane's edge (touching the grass, 30 px off: 0.02, a bit more than the 0.016 of progress at top speed: the edge never pays off). 0.01 and 0.02 were too weak: the car drifted 10 px right and cut the turns. At 0.05 (v10) driving off-center cost more than it earned and the agent learned to stall
LAB_LENGTH = 3200 # The lab (world="lab"): a straight road with traffic lights, to study the light rules quickly
LAB_MIN_LIGHT_GAP, LAB_MAX_LIGHT_GAP = 500, 900 # Distance between two lights in the lab: random in this range, ~4 lights per road
LIGHT_COLOR_ORDER = ("green", "yellow", "red") # One-hot order of the light color in the observation
GREEN_SECONDS, YELLOW_SECONDS, RED_SECONDS = 4, 2.5, 4 # Traffic light cycle (yellow: longer than the 2 s needed to brake from top speed)
LIGHT_CYCLE_STEPS = int((GREEN_SECONDS + YELLOW_SECONDS + RED_SECONDS) * STEPS_PER_SECOND) # A whole cycle, in steps

# Drawing constants: only needed to show the world on screen (render)
GRASS_COLOR = (60, 140, 60) # Define the color for grass
ROAD_COLOR = (50, 50, 50) # Define the color for the road
LINE_COLOR = (255, 255, 255) # Define the color for lane lines
LINE_WIDTH = 3 # Define the width of the lane lines
LINE_OFFSET = 3 # Define the offset for lane lines from the edge of the road
DASH_LINE_LENGTH = CAR_HEIGHT // 2 # Define the length of the dashed lane lines
DASH_LINE_WIDTH = LINE_WIDTH # Define the width of the dashed lane lines
DASH_LINE_GAP = DASH_LINE_LENGTH // 2 # Define the gap between dashed lane lines
CAR_SCREEN_X = WIDTH // 2 - CAR_WIDTH // 2 # The car is drawn at the center of the screen: in curves you see the road in every direction
CAR_SCREEN_Y = HEIGHT // 2 - CAR_HEIGHT // 2

SIDEWALK_COLOR = (170, 170, 165) # City blocks: the sidewalk around the buildings
BUILDING_COLORS = ((120, 105, 95), (140, 120, 100), (105, 110, 125), (130, 95, 90)) # Roofs, seen from above
SIDEWALK_WIDTH = 15 # From the edge of the street to the buildings
BLOCK_CORNER_RADIUS = TURN_RADIUS - ROAD_WIDTH // 2 # Rounded corners: exactly the inner edge of a 90° turn of the road
STOP_LINE_WIDTH = 2 * LINE_WIDTH # Stop lines at the intersections: thicker than the other lines
STOP_LINE_DISTANCE = TURN_RADIUS # From the center of an intersection to its stop lines: where the rounded corners begin
GOAL_COLOR = (240, 200, 40) # The goal of a city route: a yellow circle at the end of the route, in the right lane
GOAL_RADIUS = LANE_WIDTH // 2 - LINE_OFFSET - LINE_WIDTH - 2 # Fits inside the lane, without covering the white edge line
LIGHT_BOX_COLOR = (25, 25, 25) # Traffic lights: a black box with three lamps, only the active one lit
LIGHT_LAMP_COLORS = {"green": (60, 220, 90), "yellow": (250, 200, 40), "red": (235, 50, 45)}
LIGHT_LAMP_OFF = (70, 70, 70)
LIGHT_LAMP_RADIUS = 5
LIGHT_POLE_COLOR = (90, 90, 95) # The pole
LIGHT_POLE_RADIUS = 6
LIGHT_POLE_SIDE = ROAD_WIDTH // 2 + 12 # The light stands on the sidewalk, on the right of the road...
LIGHT_BACK = 12 # ...a little before the stop line
START_LINE_WIDTH = LINE_WIDTH # The start line is as thick as the edge lines
CHECKER_SIZE = 10 # Side of one square of the checkered finish line, in pixels
CHECKER_ROWS = 2 # How many rows of squares the finish line has
CHECKER_COLORS = ((255, 255, 255), (20, 20, 20)) # White and black squares

CAR_SPRITE_PATH = os.path.join(os.path.dirname(__file__), "assets", "controlled_car.png") # Car image, found next to this file
CAR_SPRITE_ANGLE = 90 # Direction the car points to in the image (up)
CAR_SPRITE_QUALITY = 4 # Keep the car image 4x bigger than drawn: rotating a big image and shrinking it looks much smoother

def world_to_screen(rect, camera_x, camera_y):
    return rect.move(-camera_x, -camera_y) # World -> screen: subtract the camera position on both axes

def move(point, angle, distance):
    # The point `distance` pixels away from `point` in direction `angle` (radians; negative distance = backward)
    x, y = point
    return (x + distance * math.cos(angle), y - distance * math.sin(angle))


def point_to_screen(point, camera_x, camera_y):
    x, y = point
    return (x - camera_x, y - camera_y) # Same as world_to_screen, for a single (x, y) point

class CarEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60} # Gymnasium: supported render modes and speed
    #metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 240} # Gymnasium: supported render modes and speed, increased to 240 for faster training

    def __init__(self, render_mode=None, road_pieces=None, world="city", traffic_lights=True, start_speed=CAR_START_SPEED):
        # what the agent can do and what it sees (action_space, observation_space); the road itself is built at every reset
        self.render_mode = render_mode # None = no drawing (training), "human" = window, "rgb_array" = image for videos
        self.screen = None # Created only the first time render() is called
        self.clock = None
        self.fixed_road_pieces = road_pieces # None = a new random road at every reset; a list of pieces = always that road
        self.world = world # Random roads: "city" (streets and 90° turns), "country" (long curves) or "lab" (straight, with traffic lights)
        self.traffic_lights = traffic_lights # False = a city without traffic lights (curriculum: first learn to drive, then the lights)
        self.start_speed = start_speed # Speed at the start of every episode (the tests use 0)

        self.car = pygame.Rect(CAR_START_X, CAR_START_Y, CAR_WIDTH, CAR_HEIGHT) # Create a rectangle for the car

        self.action_space = gym.spaces.MultiDiscrete([len(STEER_LEVELS), 3]) # Actions: [steering, pedals]. Steering: an index of STEER_LEVELS (2 = straight). Pedals: 0 = throttle, 1 = none, 2 = brake
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(15,), dtype=np.float32)  # Observations: [7 rays, lane offset, speed, direction, distance to the next stop line, next light green/yellow/red, time until it changes], all within -1..1
        self.step_count = 0 # Initialize the step counter
        self.road_index = 0 # Index of the nearest point on the road to the car's center
        self.step_noprogress_count = 0 # Count of consecutive steps with no progress (for stalling detection)

    def build_road(self, pieces, start=(ROAD_X + ROAD_WIDTH // 2, HEIGHT, ROAD_ANGLE)):
        # Everything that depends on the road: rebuilt at every reset, because the road can change.
        # start = (x, y, angle) where the center line begins: by default the bottom of the screen, pointing up
        self.road_points = build_center_line(pieces, *start) # The road's center line
        self.road_length = sum(length for length, degrees in pieces) # Total length of this road: the sum of all its pieces
        self.max_steps = int(self.road_length / MAX_SPEED * TIME_LIMIT_FACTOR) # Time limit: depends on how long this road is
        self.road_drawing_ready = False # The shapes to draw are built only when render() needs them: training never draws

    def build_road_drawing(self):
        # The shapes used to draw this road (only needed by render)
        outer = ROAD_WIDTH / 2 - LINE_OFFSET # From the center line to the outer side of each white edge line
        inner = outer - LINE_WIDTH # ...and to its inner side
        self.left_line_shape = self.line_shape(-outer, -inner) # White line along the left edge (negative = left), as an exact strip
        self.right_line_shape = self.line_shape(inner, outer) # White line along the right edge

        # The road as one shape: along the left edge from start to end, then back along the right edge (like drawing an outline without lifting the pen)
        self.road_polygon = offset_line(self.road_points, -ROAD_WIDTH / 2) + offset_line(self.road_points, ROAD_WIDTH / 2)[::-1]
        start_dir = math.radians(road_angle(self.road_points, 0)) # Direction of the road at the start
        left_start, right_start = offset_line(self.road_points[:2], -outer)[0], offset_line(self.road_points[:2], outer)[0] # Outer sides of the edge lines, at the first point
        near, far = LINE_OFFSET, LINE_OFFSET + START_LINE_WIDTH # The start line begins after the same grey margin as the sides
        self.start_line = [move(left_start, start_dir, near), move(right_start, start_dir, near), move(right_start, start_dir, far), move(left_start, start_dir, far)] # Exact rectangle across the road
        self.finish_squares = self.build_finish_squares() # Checkered squares across the road at its last point
        self.road_drawing_ready = True

    def line_shape(self, side_a, side_b):
        # A white strip running along the road between two side distances, stopping LINE_OFFSET before both ends (the same grey margin as on the sides)
        start_dir = math.radians(road_angle(self.road_points, 0))
        end_dir = math.radians(road_angle(self.road_points, len(self.road_points) - 1))
        def trimmed(line):
            return [move(line[0], start_dir, LINE_OFFSET)] + line[1:-1] + [move(line[-1], end_dir, -LINE_OFFSET)]
        return trimmed(offset_line(self.road_points, side_a)) + trimmed(offset_line(self.road_points, side_b))[::-1] # Along one side, back along the other

    def build_finish_squares(self):
        # Checkered finish line: rows of squares across the road, with a grey margin before the end of the road
        end_x, end_y = self.road_points[-1]
        forward = math.radians(road_angle(self.road_points, len(self.road_points) - 1)) # Direction of the road at the end
        right = forward - math.pi / 2 # Direction "right" of the road
        fx, fy = math.cos(forward), -math.sin(forward) # One pixel forward, in screen coordinates (y goes down)
        rx, ry = math.cos(right), -math.sin(right) # One pixel to the right
        line_span = ROAD_WIDTH / 2 - LINE_OFFSET # The checkers cover the road between the grey margins, like the edge lines
        columns = int(2 * line_span // CHECKER_SIZE)
        square_width = 2 * line_span / columns # Slightly wider than CHECKER_SIZE, so the columns fill the space between the grey margins exactly
        squares = []
        for row in range(CHECKER_ROWS):
            for column in range(columns):
                back = LINE_OFFSET + (row + 1) * CHECKER_SIZE # Rows go back from the end of the road, after a grey margin
                side = -line_span + column * square_width # Columns go from the left margin to the right margin
                corners = []
                for dback, dside in ((0, 0), (0, square_width), (CHECKER_SIZE, square_width), (CHECKER_SIZE, 0)):
                    distance_back = back - dback
                    distance_side = side + dside
                    corners.append((end_x - fx * distance_back + rx * distance_side, end_y - fy * distance_back + ry * distance_side))
                squares.append((corners, CHECKER_COLORS[(row + column) % 2])) # Alternate colors like a chessboard
        return squares

    def draw_road(self, camera_x, camera_y):
        # A single road (country roads and tests): asphalt, white edge lines, dashed center line
        road_shape = [point_to_screen(point, camera_x, camera_y) for point in self.road_polygon]
        pygame.draw.polygon(self.screen, ROAD_COLOR, road_shape) # Fill the whole road (straight cut at the start and at the end)
        
        for shape in (self.left_line_shape, self.right_line_shape): # The two white edge lines
            pygame.draw.polygon(self.screen, LINE_COLOR, [point_to_screen(point, camera_x, camera_y) for point in shape]) # Convert every point to the screen and fill the strip

        dash_points = DASH_LINE_LENGTH // POINT_SPACING # How many center line points one dash covers
        gap_points = DASH_LINE_GAP // POINT_SPACING # How many points the gap between two dashes covers
        first_dash = math.ceil((LINE_OFFSET + START_LINE_WIDTH) / POINT_SPACING) # Dashes start after the start line...
        last_dash = len(self.road_points) - 1 - math.ceil((LINE_OFFSET + CHECKER_ROWS * CHECKER_SIZE) / POINT_SPACING) # ...and stop before the finish line
        for start in range(first_dash, last_dash, dash_points + gap_points): # One dash every (dash + gap) points
            dash = [point_to_screen(point, camera_x, camera_y) for point in self.road_points[start:min(start + dash_points, last_dash) + 1]] # The points of this dash
            if len(dash) > 1: # A line needs at least two points
                pygame.draw.lines(self.screen, LINE_COLOR, False, dash, DASH_LINE_WIDTH) # Draw one dash, following the curve

    def light_colors(self):
        # The color of every traffic light right now, in route order: "green", "yellow" or "red"
        green_end = GREEN_SECONDS * STEPS_PER_SECOND # The cycle: green from 0 to here...
        yellow_end = (GREEN_SECONDS + YELLOW_SECONDS) * STEPS_PER_SECOND # ...then yellow until here, then red until the end
        colors = []
        for timer in self.light_timers:
            if timer < green_end:
                colors.append("green")
            elif timer < yellow_end:
                colors.append("yellow")
            else:
                colors.append("red")
        return colors

    def light_time_left(self, light):
        # Steps until traffic light number `light` changes color (like the countdown some real lights have).
        # In the observations: it solves the yellow dilemma (the v11 test failed only because the car started still and never explored)
        timer = self.light_timers[light]
        green_end = GREEN_SECONDS * STEPS_PER_SECOND
        yellow_end = (GREEN_SECONDS + YELLOW_SECONDS) * STEPS_PER_SECOND
        if timer < green_end:
            return green_end - timer
        if timer < yellow_end:
            return yellow_end - timer
        return LIGHT_CYCLE_STEPS - timer

    def light_gap(self, stop_index):
        # Free space between the car's nose and a stop line, in pixels (negative: the nose is already past it)
        return (stop_index - self.road_index) * POINT_SPACING - CAR_HEIGHT / 2

    def next_light(self):
        # The next traffic light ahead of the car: (its stop line index, its color), or None if there is none left
        for stop_index, color in zip(self.stop_indices, self.light_colors()):
            if stop_index > self.road_index: # Its stop line is still ahead of the car's center
                return stop_index, color
        return None

    def find_stop_indices(self):
        # For every intersection in the middle of the route: the point of the route where the car meets its stop line.
        # The stop line is STOP_LINE_DISTANCE before the center of the intersection, on the street the car arrives from:
        # exactly where a turn begins, so it is always a point of the route.
        xs, ys, route = self.city["xs"], self.city["ys"], self.city["route"]
        indices = []
        for (row_a, col_a), (row_b, col_b) in zip(route[:-2], route[1:-1]): # Every step that arrives at a middle intersection
            center_x, center_y = xs[col_b], ys[row_b]
            back_x, back_y = xs[col_a] - center_x, ys[row_a] - center_y # Pointing back along the street we arrive from
            length = math.hypot(back_x, back_y)
            stop_x = center_x + back_x / length * STOP_LINE_DISTANCE
            stop_y = center_y + back_y / length * STOP_LINE_DISTANCE
            indices.append(nearest_point_index(self.road_points, stop_x, stop_y))
        return indices

    def lab_stop_indices(self):
        # The lab's stop lines: random distances apart (so the agent can't learn their places by heart), in multiples of POINT_SPACING
        indices, distance = [], 0
        while True:
            distance += int(self.np_random.integers(LAB_MIN_LIGHT_GAP, LAB_MAX_LIGHT_GAP + 1)) // POINT_SPACING * POINT_SPACING
            if distance > LAB_LENGTH - LAB_MIN_LIGHT_GAP / 2: # Leave some road after the last light
                return indices
            indices.append(distance // POINT_SPACING)

    def draw_stop_line(self, stop_index, camera_x, camera_y):
        # A thick white line across the right lane, ending at the stop point (the city draws its own with the streets)
        forward = math.radians(road_angle(self.road_points, stop_index))
        right = forward - math.pi / 2
        point = self.road_points[stop_index]
        outer = ROAD_WIDTH / 2 - LINE_OFFSET
        corners = [move(point, right, 0), move(point, right, outer), move(move(point, right, outer), forward, -STOP_LINE_WIDTH), move(point, forward, -STOP_LINE_WIDTH)]
        pygame.draw.polygon(self.screen, LINE_COLOR, [point_to_screen(corner, camera_x, camera_y) for corner in corners])

    def draw_traffic_light(self, stop_index, color, camera_x, camera_y):
        # On the right sidewalk, just before the stop line: a pole and a box with the three lamps in a column,
        # like a traffic light seen from the front: red farthest ahead, then yellow, then green closest to the car
        forward = math.radians(road_angle(self.road_points, stop_index)) # Direction of the cars arriving at this light
        right = forward - math.pi / 2
        box_center = move(move(self.road_points[stop_index], right, LIGHT_POLE_SIDE), forward, -LIGHT_BACK - LIGHT_LAMP_RADIUS * 4)
        to_screen = lambda point: point_to_screen(point, camera_x, camera_y)

        pygame.draw.circle(self.screen, LIGHT_POLE_COLOR, to_screen(move(box_center, right, LIGHT_LAMP_RADIUS + 5)), LIGHT_POLE_RADIUS) # The pole, behind the box

        spacing = 2 * LIGHT_LAMP_RADIUS + 3
        half_along, half_across = 1.5 * spacing + 2, LIGHT_LAMP_RADIUS + 3 # The box is long along the road: the lamps are in a column
        corners = [move(move(box_center, forward, f * half_along), right, r * half_across) for f, r in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        pygame.draw.polygon(self.screen, LIGHT_BOX_COLOR, [to_screen(point) for point in corners])
        for lamp_color, ahead in (("red", spacing), ("yellow", 0), ("green", -spacing)): # Red on top (farthest ahead), green at the bottom
            lit = LIGHT_LAMP_COLORS[lamp_color] if lamp_color == color else LIGHT_LAMP_OFF
            pygame.draw.circle(self.screen, lit, to_screen(move(box_center, forward, ahead)), LIGHT_LAMP_RADIUS)

    def draw_city(self, camera_x, camera_y):
        # The whole city around the route: asphalt everywhere, blocks with rounded corners, white edge lines and dashed center lines
        xs, ys = self.city["xs"], self.city["ys"]
        half = ROAD_WIDTH // 2

        def screen_rect(left, top, right, bottom): # A world rectangle, on the screen
            return pygame.Rect(round(left - camera_x), round(top - camera_y), round(right - left), round(bottom - top))

        city_area = screen_rect(xs[0] - half, ys[0] - half, xs[-1] + half, ys[-1] + half)
        pygame.draw.rect(self.screen, ROAD_COLOR, city_area) # Every street and intersection
        edge = LINE_OFFSET + LINE_WIDTH
        pygame.draw.rect(self.screen, LINE_COLOR, city_area.inflate(-2 * LINE_OFFSET, -2 * LINE_OFFSET), LINE_WIDTH) # Edge line along the outer streets

        for row in range(len(ys) - 1):
            for column in range(len(xs) - 1):
                block = screen_rect(xs[column] + half, ys[row] + half, xs[column + 1] - half, ys[row + 1] - half)
                pygame.draw.rect(self.screen, LINE_COLOR, block.inflate(2 * edge, 2 * edge), LINE_WIDTH, border_radius=BLOCK_CORNER_RADIUS + edge) # Edge line around the block
                pygame.draw.rect(self.screen, SIDEWALK_COLOR, block, border_radius=BLOCK_CORNER_RADIUS) # Sidewalk
                roof = BUILDING_COLORS[(row * 7 + column * 3) % len(BUILDING_COLORS)] # A different roof color for each block
                pygame.draw.rect(self.screen, roof, block.inflate(-2 * SIDEWALK_WIDTH, -2 * SIDEWALK_WIDTH), border_radius=max(0, BLOCK_CORNER_RADIUS - SIDEWALK_WIDTH)) # Buildings

        step = DASH_LINE_LENGTH + DASH_LINE_GAP
        margin = STOP_LINE_WIDTH + DASH_LINE_GAP # Dashes start a little after a stop line and end a little before the next one
        for y in ys: # Dashed center line of every horizontal street, from one stop line to the next
            for x_from, x_to in zip(xs, xs[1:]):
                for x in range(x_from + STOP_LINE_DISTANCE + margin, x_to - STOP_LINE_DISTANCE - margin - DASH_LINE_LENGTH + 1, step):
                    pygame.draw.rect(self.screen, LINE_COLOR, screen_rect(x, y - DASH_LINE_WIDTH / 2, x + DASH_LINE_LENGTH, y + DASH_LINE_WIDTH / 2))
        for x in xs: # ...and of every vertical street
            for y_from, y_to in zip(ys, ys[1:]):
                for y in range(y_from + STOP_LINE_DISTANCE + margin, y_to - STOP_LINE_DISTANCE - margin - DASH_LINE_LENGTH + 1, step):
                    pygame.draw.rect(self.screen, LINE_COLOR, screen_rect(x - DASH_LINE_WIDTH / 2, y, x + DASH_LINE_WIDTH / 2, y + DASH_LINE_LENGTH))

        lane_edge = half - LINE_OFFSET - LINE_WIDTH # From the center line to the inner side of the white edge line
        for row, y in enumerate(ys): # Stop lines: on every street that enters an intersection, across the lane of the cars arriving
            for column, x in enumerate(xs):
                d, w = STOP_LINE_DISTANCE, STOP_LINE_WIDTH
                if column > 0: # Arriving from the left (driving right): the right lane is below the center line
                    pygame.draw.rect(self.screen, LINE_COLOR, screen_rect(x - d - w, y, x - d, y + lane_edge))
                if column < len(xs) - 1: # Arriving from the right (driving left): the right lane is above
                    pygame.draw.rect(self.screen, LINE_COLOR, screen_rect(x + d, y - lane_edge, x + d + w, y))
                if row > 0: # Arriving from above (driving down): the right lane is on the left of the screen
                    pygame.draw.rect(self.screen, LINE_COLOR, screen_rect(x - lane_edge, y - d - w, x, y - d))
                if row < len(ys) - 1: # Arriving from below (driving up): the right lane is on the right
                    pygame.draw.rect(self.screen, LINE_COLOR, screen_rect(x, y + d, x + lane_edge, y + d + w))

        goal_x, goal_y = offset_line(self.road_points[-2:], LANE_WIDTH / 2)[-1] # The goal: a yellow circle at the end of the route, in the right lane
        pygame.draw.circle(self.screen, GOAL_COLOR, (goal_x - camera_x, goal_y - camera_y), GOAL_RADIUS)

    def render(self):
        if self.render_mode is None:
            return # Training: nothing to draw

        if self.screen is None: # First call: set up pygame
            pygame.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
                pygame.display.set_caption("RL Car 2D")
            else:
                self.screen = pygame.Surface((WIDTH, HEIGHT)) # Invisible surface for videos
            self.clock = pygame.time.Clock()

            image = pygame.image.load(CAR_SPRITE_PATH) # Load the car image from the specified path
            car_only = image.subsurface(image.get_bounding_rect()) # Cut away the transparent border around the car
            sprite_width = round(CAR_HEIGHT * car_only.get_width() / car_only.get_height()) # Keep the image proportions (option A)
            self.car_sprite = pygame.transform.smoothscale(car_only, (sprite_width * CAR_SPRITE_QUALITY, CAR_HEIGHT * CAR_SPRITE_QUALITY)) # Shrink it, but keep it CAR_SPRITE_QUALITY times bigger than the car

        if not self.road_drawing_ready: # A new road since the last frame: prepare its shapes
            self.build_road_drawing()

        camera_y = self.car.top - CAR_SCREEN_Y # Camera follows the car, keeping it at a fixed height on screen
        camera_x = self.car.left - CAR_SCREEN_X # Camera follows the car, keeping it at a fixed x position on screen

        self.screen.fill(GRASS_COLOR) # Fill the screen with green {grass}
        if self.city is not None:
            self.draw_city(camera_x, camera_y) # The city's streets already contain the route
        else:
            self.draw_road(camera_x, camera_y)

        if self.city is None: # In the city the route starts in the middle of an intersection: no start line there
            pygame.draw.polygon(self.screen, LINE_COLOR, [point_to_screen(point, camera_x, camera_y) for point in self.start_line]) # White start line across the road
        if self.city is None: # In the city the goal is the yellow circle, drawn with the city
            for corners, color in self.finish_squares: # Checkered finish line
                pygame.draw.polygon(self.screen, color, [point_to_screen(point, camera_x, camera_y) for point in corners])

        if self.city is None: # The city draws its stop lines with the streets
            for stop_index in self.stop_indices:
                self.draw_stop_line(stop_index, camera_x, camera_y)

        rotated = pygame.transform.rotozoom(self.car_sprite, self.car_angle - CAR_SPRITE_ANGLE, 1 / CAR_SPRITE_QUALITY) # 1. Rotate the image like the car (smooth edges) and shrink it to the car size
        car_center = world_to_screen(self.car, camera_x,camera_y).center # 2. Where the car's center is on screen
        self.screen.blit(rotated, rotated.get_rect(center=car_center)) # 3. Paste it with its center on the car's center
        for stop_index, color in zip(self.stop_indices, self.light_colors()): # The traffic lights hang over the road: drawn above the car
            self.draw_traffic_light(stop_index, color, camera_x, camera_y)

        if self.render_mode == "human":
            pygame.event.pump() # Keep the window responsive
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2)) # Image as (height, width, 3)

    def close(self):
        # Gymnasium: free the window when the environment is no longer used
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    def center_offset(self):
        # Signed distance from the car's center to the road's center line: positive = right, negative = left
        return side_offset(self.road_points, self.road_index, self.car.centerx, self.car.centery)

    def get_obs(self):
        # What the agent sees: 7 rays (how far the road goes in each direction), then lane position, speed and direction
        offset = self.center_offset() # Calculate the offset of the car from the center line of the road at the nearest point
        #normalized_dist_left = (ROAD_WIDTH / 2 + offset - CAR_WIDTH / 2) / ROAD_WIDTH # Distance from the car's left side to the left edge of the road
        #normalized_dist_right = (ROAD_WIDTH / 2 - offset - CAR_WIDTH / 2) / ROAD_WIDTH # Distance from the car's right side to the right edge of the road
        normalized_dist_center_right_lane = max(-1.0, min(1.0, (offset - LANE_WIDTH / 2) / (LANE_WIDTH / 2))) # Distance from the center of the right lane, in half lanes: 10 px = 0.22, clearly visible (it was / ROAD_WIDTH: 10 px = 0.055, almost invisible)
        normalized_speed = self.car_speed / MAX_SPEED # Normalize the speed
        current_road_angle = road_angle(self.road_points, self.road_index) # Direction of the road where the car is
        normalized_direction = ((self.car_angle - current_road_angle + 180) % 360 - 180) / 180 # Difference from the road direction, always between -1 and +1

        normalized_ray_length = []
        for angle in RAY_ANGLES:
            length = ray_length(self.road_points, self.car.centerx, self.car.centery, self.car_angle + angle, RAY_LENGTH, RAY_STEP, ROAD_WIDTH / 2, self.road_index) # Calculate the length of the ray in the given direction, considering the road boundaries and the car's position
            normalized_ray_length.append(length / RAY_LENGTH) # Normalize the ray length to be between 0 and 1

        light = self.next_light()
        if light is None: # No traffic light ahead: as if it were far away and green
            normalized_light_distance, light_one_hot, normalized_time_left = 1.0, [1.0, 0.0, 0.0], 1.0
        else:
            stop_index, color = light
            gap = max(0.0, min(self.light_gap(stop_index), RAY_LENGTH)) # Free space in front of the nose, the same measure the rules use
            normalized_light_distance = math.sqrt(gap / RAY_LENGTH) # Square root: more detail close to the line (20 px = 0.18 instead of 0.03), where stopping right matters
            light_one_hot = [1.0 if color == name else 0.0 for name in LIGHT_COLOR_ORDER] # Exactly one 1: the color that is on
            normalized_time_left = self.light_time_left(self.stop_indices.index(stop_index)) / (max(GREEN_SECONDS, RED_SECONDS) * STEPS_PER_SECOND) # Countdown: how soon the color changes (0..1)
            if color == "red":
                normalized_time_left = 1.0 # Hidden on red: useful to decide on green and yellow, but on red it only helps to time the green and run the light

        normalized_values = np.array(normalized_ray_length + [normalized_dist_center_right_lane, normalized_speed, normalized_direction, normalized_light_distance] + light_one_hot + [normalized_time_left], dtype=np.float32)

        return normalized_values # [7 rays (in the order of RAY_ANGLES), lane offset, speed, direction, stop line distance, green, yellow, red, time until the light changes]

    def reset(self, seed=None, options=None):
        # build the road (random or fixed), put the car back at the start (position, angle, speed), reset the step counter
        # return first observation, info
        super().reset(seed=seed)
        self.city = None # Only set in the city world: the streets to draw around the route
        self.stop_indices = [] # Points of the route where there is a stop line with a traffic light (none outside the city)
        if self.fixed_road_pieces is not None:
            self.build_road(self.fixed_road_pieces) # Always the same road (tests, or a road chosen on purpose)
        elif self.world == "city":
            self.city = random_city(self.np_random) # A new random city, and a route through it
            self.build_road(self.city["pieces"], self.city["start"])
            if self.traffic_lights:
                self.stop_indices = self.find_stop_indices() # Where the route crosses a stop line: one traffic light each
            self.max_steps += len(self.stop_indices) * LIGHT_CYCLE_STEPS # Extra time to wait at every traffic light (a whole cycle each)
        elif self.world == "lab":
            self.build_road([(LAB_LENGTH, 0)]) # The lab: one straight road, only the traffic lights to learn
            self.stop_indices = self.lab_stop_indices()
            self.max_steps += len(self.stop_indices) * LIGHT_CYCLE_STEPS
        else:
            self.build_road(random_road_pieces(self.np_random, ROAD_PIECES_COUNT, MIN_CURVE_RADIUS)) # A new random country road: the agent can't learn one road by heart

        self.light_timers = [int(self.np_random.integers(0, LIGHT_CYCLE_STEPS)) for _ in self.stop_indices] # One timer per light, each starting at a random point of its cycle (not synchronized)
        self.must_stop_at_yellow = [True] * len(self.stop_indices) # Decided when each light turns yellow (a light that starts already yellow: the car is far, it must stop)
        self.stopped_in_zone = [False] * len(self.stop_indices) # STOP_IN_ZONE_REWARD is paid once per light

        # The car starts at the beginning of the road, in the right lane, pointing along the road
        start_angle = road_angle(self.road_points, 0)
        forward, right = math.radians(start_angle), math.radians(start_angle - 90)
        start_x, start_y = self.road_points[0]
        center_x = start_x + CAR_START_AHEAD * math.cos(forward) + LANE_WIDTH / 2 * math.cos(right) # A bit after the start line...
        center_y = start_y - CAR_START_AHEAD * math.sin(forward) - LANE_WIDTH / 2 * math.sin(right) # ...in the middle of the right lane
        self.car_x = center_x - CAR_WIDTH / 2 # The true position is the top-left corner of the car's Rect
        self.car_y = center_y - CAR_HEIGHT / 2
        self.car.left = round(self.car_x)
        self.car.top = round(self.car_y)
        self.car_angle = start_angle
        self.car_speed = self.start_speed
        self.step_count = 0
        self.step_noprogress_count = 0
        self.road_index = nearest_point_index(self.road_points, self.car.centerx, self.car.centery) # Full search: a new episode, there is no previous index

        if self.render_mode == "human": # Gymnasium convention: in human mode the window updates by itself
            self.render()
        return self.get_obs(), {}

    def step(self, action):
        # action is a pair of values: [steering, pedals]
        steer = STEER_LEVELS[action[0]] # Steering: -1 = full left ... 0 = straight ... +1 = full right
        pedal = action[1] - 1 # Pedals: -1 = throttle, 0 = none, +1 = brake
        
        if pedal == -1: # Throttle
            self.car_speed += ACCELERATION
        elif pedal == 1: # Brake
            self.car_speed -= BRAKE
        else: # No pedals: the car slows down by itself
            self.car_speed -= FRICTION
        self.car_speed = max(0, min(self.car_speed, MAX_SPEED)) # Keep the speed between 0 and MAX_SPEED

        self.car_angle -= steer * STEER_RATE * self.car_speed # Update the car's angle based on the steering input and speed
        self.car_x += self.car_speed * math.cos(math.radians(self.car_angle)) # Update the car's x position based on its angle and speed
        self.car_y -= self.car_speed * math.sin(math.radians(self.car_angle)) # Update the car's y position based on its angle and speed

        self.car.left = round(self.car_x) # The Rect (whole pixels) follows the true position (decimals): used for drawing and edge checks
        self.car.top = round(self.car_y)

        previous_road_index = self.road_index # To know which stop lines the car crosses in this step
        new_road_index = nearest_point_index(self.road_points, self.car.centerx, self.car.centery, around=self.road_index, window=CAR_SEARCH_WINDOW) # Update the index of the nearest point on the road to the car's center
        progress = (new_road_index - self.road_index) * POINT_SPACING # Calculate the progress made along the road based on the change in road index
        self.road_index = new_road_index # Update the road index for the next step

        offset = self.center_offset() # Calculate the offset of the car from the center line of the road at the nearest point

        reward = progress * PROGRESS_REWARD_PER_PIXEL # Reward for moving forward along the road
        reward -= LANE_PENALTY * abs(offset - LANE_WIDTH / 2) / (LANE_WIDTH // 2) # Penalty for being away from the center of the right lane, scaled by half the lane width
        heading_error = (self.car_angle - road_angle(self.road_points, self.road_index) + 180) % 360 - 180 # Degrees between the car and the road
        reward -= HEADING_PENALTY * abs(heading_error) / 45 # Pointing sideways: weaving costs at every step, not only once far from the lane center
        
        if abs(offset) > ROAD_WIDTH / 2 - CAR_WIDTH / 2: # Off road: part of the car is outside the road boundaries
            reward -= GRASS_PENALTY # Add a penalty for going off the road

        if offset - CAR_WIDTH / 2 < 0: # Wrong way: part of the car is in the opposite lane
            reward -= WRONG_WAY_PENALTY # Add a penalty for being in the wrong lane

        terminated = False # Initialize the terminated flag to False
        truncated = False # Initialize the truncated flag to False
        reason = None # Why the episode ended (returned in info, printed by main.py)

        behind_start = self.road_index == 0 and along_offset(self.road_points, 0, self.car.centerx, self.car.centery) < 0 # Turned around and drove back past the start line

        if abs(offset) > ROAD_WIDTH / 2 or behind_start: # Off road (or behind the start, where there is no road): episode fails
            terminated = True # Set the terminated flag to True
            reward -= OFF_ROAD_PENALTY # Add a penalty for going off the road
            reason = "behind the start line" if behind_start else "off the route (in the city: also a street that is not the route)"

        if self.road_index == len(self.road_points) - 1: # Reached the end of the road: episode succeeds
            terminated = True # Set the terminated flag to True
            reward += FINISH_REWARD # Add a reward for successfully reaching the end of the road
            reason = "finish"

        nose = CAR_HEIGHT // 2 // POINT_SPACING # How many road points the car's nose is ahead of its center
        for k, (stop_index, color) in enumerate(zip(self.stop_indices, self.light_colors())): # Crossing a stop line with the nose on red (or on a yellow the car could stop for) ends the episode
            crossed = previous_road_index + nose < stop_index <= self.road_index + nose
            if crossed and (color == "red" or (color == "yellow" and self.must_stop_at_yellow[k])):
                terminated = True
                reward -= RED_LIGHT_PENALTY
                reason = "crossed on red" if color == "red" else "crossed on a yellow it could stop for"

        light = self.next_light()
        stopping_light = light is not None and light[1] != "green" # A yellow or red light ahead
        if stopping_light:
            gap = self.light_gap(light[0]) # Free space between the car's nose and the stop line
            k = self.stop_indices.index(light[0])
            must_stop = light[1] == "red" or self.must_stop_at_yellow[k]
            if must_stop: # The braking curve: the fastest speed that still stops gently in the middle of the wait zone (v² = 2·a·distance)
                curve_speed = math.sqrt(2 * COMFORT_BRAKE * max(gap - LIGHT_WAIT_GAP / 2, 0))
                reward -= OVERSPEED_PENALTY * max(self.car_speed - curve_speed, 0) / MAX_SPEED
                if gap > LIGHT_WAIT_GAP: # Before the wait zone: much slower than the curve means it will stop too early
                    reward -= UNDERSPEED_PENALTY * max(curve_speed / 2 - self.car_speed, 0) / MAX_SPEED
            if must_stop and self.car_speed == 0 and gap <= LIGHT_WAIT_GAP and not self.stopped_in_zone[k]: # Stopped in the wait zone: well done, once per light
                self.stopped_in_zone[k] = True
                reward += STOP_IN_ZONE_REWARD
        waiting_at_light = stopping_light # Stopped at a yellow or red light: never a stall (being far from the line already costs UNDERSPEED_PENALTY)
        if progress <= 0 and not waiting_at_light: # No progress made in this step (and no good reason for it)
            self.step_noprogress_count += 1 # Increment the count of consecutive steps with no progress
            if self.step_noprogress_count >= STALL_SECONDS * STEPS_PER_SECOND: # If the car has been stuck for too long
                terminated = True # Set the terminated flag to True, ending the episode due to stalling        
                reward -= STALL_PENALTY # Add a penalty for stalling
                reason = "stalled (stopped too long, or too far from the stop line)"
        elif progress > 0:
            self.step_noprogress_count = 0 # Reset the count if progress is made
        
        self.light_timers = [(timer + 1) % LIGHT_CYCLE_STEPS for timer in self.light_timers] # Every light moves one step on in its cycle, then starts again
        for k, timer in enumerate(self.light_timers): # A light just turned yellow: could the car still stop before its line?
            if timer == int(GREEN_SECONDS * STEPS_PER_SECOND):
                gap_now = self.light_gap(self.stop_indices[k])
                braking_distance = self.car_speed ** 2 / (2 * BRAKE)
                self.must_stop_at_yellow[k] = gap_now > braking_distance # Far enough to brake: it must stop (Italian Highway Code, art. 41). Too close: it may go on
        self.step_count += 1 # Increment the step counter

        if self.step_count >= self.max_steps and not terminated: # Out of time: with a generous limit, only a car that crawls gets here. It counts as a failure
            terminated = True
            reward -= TIME_OUT_PENALTY
            reason = "time out (too slow)"

        if self.render_mode == "human": # Gymnasium convention: in human mode the window updates by itself
            self.render()

        return self.get_obs(), reward, terminated, truncated, {"reason": reason} # Observation, reward, terminated, truncated, and why the episode ended (None while it goes on)
