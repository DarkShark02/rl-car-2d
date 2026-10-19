import pygame
import numpy as np
import gymnasium as gym
import math
import os
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
ROAD_PIECES = [(300, 0), (400, -90), (500, 0), (400, 90), (1000, 0)] # A fixed test road, never used in training: to check that the agent can drive a road it has never seen
ROAD_PIECES_COUNT = 16 # How many pieces a random road has (about 6000-8000 pixels)
RAY_SEARCH_WINDOW = 5 # A ray moves POINT_SPACING pixels per step: its nearest road point moves at most ~1 point, so a tiny search is enough

# Car physics: every value is expressed in seconds, then converted to steps (the game runs at 60 steps per second)
STEPS_PER_SECOND = 60 # Simulation steps in one second of game time
MAX_SPEED = 8 # Top speed, in pixels per step
ACCELERATION = MAX_SPEED / (3 * STEPS_PER_SECOND) # Full throttle: from 0 to MAX_SPEED in 3 seconds
BRAKE = MAX_SPEED / (2 * STEPS_PER_SECOND) # Full brake: from MAX_SPEED to 0 in 2 seconds
FRICTION = MAX_SPEED / (6 * STEPS_PER_SECOND) # No pedals: from MAX_SPEED to 0 in 6 seconds
CAR_TURN_RADIUS = 2 * CAR_HEIGHT # Tightest circle the car can drive, steering fully: about twice its length (pixels)
STEER_RATE = math.degrees(1 / CAR_TURN_RADIUS) # Degrees turned per pixel traveled: the same circle at any speed
MIN_CURVE_RADIUS = 1.5 * CAR_TURN_RADIUS # Random curves are never tighter than this: the car can drive them with some margin
TIME_LIMIT_FACTOR = 2 # Time limit of an episode: twice the steps needed to drive the whole road at top speed
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

    def __init__(self, render_mode=None, road_pieces=None):
        # what the agent can do and what it sees (action_space, observation_space); the road itself is built at every reset
        self.render_mode = render_mode # None = no drawing (training), "human" = window, "rgb_array" = image for videos
        self.screen = None # Created only the first time render() is called
        self.clock = None
        self.fixed_road_pieces = road_pieces # None = a new random road at every reset; a list of pieces = always that road

        self.car = pygame.Rect(CAR_START_X, CAR_START_Y, CAR_WIDTH, CAR_HEIGHT) # Create a rectangle for the car

        self.action_space = gym.spaces.MultiDiscrete([3, 3])         # Actions: [steering, pedals]. Steering: 0 = left, 1 = straight, 2 = right. Pedals: 0 = throttle, 1 = none, 2 = brake
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)  # Observations: [angle distance in 7 rays, distance to center of right lane, speed, direction], all normalized between -1 and 1
        self.step_count = 0 # Initialize the step counter
        self.road_index = 0 # Index of the nearest point on the road to the car's center
        self.step_noprogress_count = 0 # Count of consecutive steps with no progress (for stalling detection)

    def build_road(self, pieces):
        # Everything that depends on the road: rebuilt at every reset, because the road can change
        self.road_points = build_center_line(pieces, ROAD_X + ROAD_WIDTH // 2, HEIGHT, ROAD_ANGLE) # The road's center line
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
        
        pygame.draw.polygon(self.screen, LINE_COLOR, [point_to_screen(point, camera_x, camera_y) for point in self.start_line]) # White start line across the road
        for corners, color in self.finish_squares: # Checkered finish line
            pygame.draw.polygon(self.screen, color, [point_to_screen(point, camera_x, camera_y) for point in corners])

        rotated = pygame.transform.rotozoom(self.car_sprite, self.car_angle - CAR_SPRITE_ANGLE, 1 / CAR_SPRITE_QUALITY) # 1. Rotate the image like the car (smooth edges) and shrink it to the car size
        car_center = world_to_screen(self.car, camera_x,camera_y).center # 2. Where the car's center is on screen
        self.screen.blit(rotated, rotated.get_rect(center=car_center)) # 3. Paste it with its center on the car's center

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
        normalized_dist_center_right_lane = (offset - LANE_WIDTH / 2) / ROAD_WIDTH # Distance from the car's center to the center of the right lane
        normalized_speed = self.car_speed / MAX_SPEED # Normalize the speed
        current_road_angle = road_angle(self.road_points, self.road_index) # Direction of the road where the car is
        normalized_direction = ((self.car_angle - current_road_angle + 180) % 360 - 180) / 180 # Difference from the road direction, always between -1 and +1

        normalized_ray_length = []
        for angle in RAY_ANGLES:
            length = ray_length(self.road_points, self.car.centerx, self.car.centery, self.car_angle + angle, RAY_LENGTH, RAY_STEP, ROAD_WIDTH / 2, self.road_index) # Calculate the length of the ray in the given direction, considering the road boundaries and the car's position
            normalized_ray_length.append(length / RAY_LENGTH) # Normalize the ray length to be between 0 and 1

        normalized_values = np.array(normalized_ray_length + [normalized_dist_center_right_lane, normalized_speed, normalized_direction], dtype=np.float32)

        return normalized_values # [7 rays (in the order of RAY_ANGLES), lane offset, speed, direction]

    def reset(self, seed=None, options=None):
        # build the road (random or fixed), put the car back at the start (position, angle, speed), reset the step counter
        # return first observation, info
        super().reset(seed=seed)
        if self.fixed_road_pieces is not None:
            pieces = self.fixed_road_pieces # Always the same road (tests, or a road chosen on purpose)
        else:
            pieces = random_road_pieces(self.np_random, ROAD_PIECES_COUNT, MIN_CURVE_RADIUS) # A new random road: the agent can't learn one road by heart
        self.build_road(pieces)

        self.car.left = CAR_START_X
        self.car.top = CAR_START_Y
        self.car_x = CAR_START_X
        self.car_y = CAR_START_Y
        self.car_angle = CAR_START_ANGLE
        self.car_speed = 0
        self.step_count = 0
        self.step_noprogress_count = 0
        self.road_index = nearest_point_index(self.road_points, self.car.centerx, self.car.centery) # Full search: a new episode, there is no previous index

        if self.render_mode == "human": # Gymnasium convention: in human mode the window updates by itself
            self.render()
        return self.get_obs(), {}

    def step(self, action):
        # action is a pair of values: [steering, pedals]
        steer = action[0] - 1 # Steering: -1 = left, 0 = straight, +1 = right
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

        new_road_index = nearest_point_index(self.road_points, self.car.centerx, self.car.centery, around=self.road_index, window=CAR_SEARCH_WINDOW) # Update the index of the nearest point on the road to the car's center
        progress = (new_road_index - self.road_index) * POINT_SPACING # Calculate the progress made along the road based on the change in road index
        self.road_index = new_road_index # Update the road index for the next step

        offset = self.center_offset() # Calculate the offset of the car from the center line of the road at the nearest point

        reward = progress * PROGRESS_REWARD_PER_PIXEL # Reward for moving forward along the road
        reward -= 0.01 * abs(offset - LANE_WIDTH / 2) / (LANE_WIDTH // 2) # Small penalty for being away from the center of the right lane, scaled by half the lane width
        
        if abs(offset) > ROAD_WIDTH / 2 - CAR_WIDTH / 2: # Off road: part of the car is outside the road boundaries
            reward -= GRASS_PENALTY # Add a penalty for going off the road

        if offset - CAR_WIDTH / 2 < 0: # Wrong way: part of the car is in the opposite lane
            reward -= WRONG_WAY_PENALTY # Add a penalty for being in the wrong lane

        terminated = False # Initialize the terminated flag to False
        truncated = False # Initialize the truncated flag to False

        behind_start = self.road_index == 0 and along_offset(self.road_points, 0, self.car.centerx, self.car.centery) < 0 # Turned around and drove back past the start line

        if abs(offset) > ROAD_WIDTH / 2 or behind_start: # Off road (or behind the start, where there is no road): episode fails
            terminated = True # Set the terminated flag to True
            reward -= OFF_ROAD_PENALTY # Add a penalty for going off the road

        if self.road_index == len(self.road_points) - 1: # Reached the end of the road: episode succeeds
            terminated = True # Set the terminated flag to True
            reward += FINISH_REWARD # Add a reward for successfully reaching the end of the road

        if progress <= 0: # No progress made in this step
            self.step_noprogress_count += 1 # Increment the count of consecutive steps with no progress
            if self.step_noprogress_count >= STALL_SECONDS * STEPS_PER_SECOND: # If the car has been stuck for too long
                terminated = True # Set the terminated flag to True, ending the episode due to stalling        
                reward -= STALL_PENALTY # Add a penalty for stalling
        else:
            self.step_noprogress_count = 0 # Reset the count if progress is made
        
        self.step_count += 1 # Increment the step counter

        if self.step_count >= self.max_steps: # Check if the maximum number of steps has been reached
            truncated = True

        if self.render_mode == "human": # Gymnasium convention: in human mode the window updates by itself
            self.render()

        return self.get_obs(), reward, terminated, truncated, {} # Return the observation, reward, terminated flag, truncated flag, and an empty info dictionary
