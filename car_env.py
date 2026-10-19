import pygame
import numpy as np
import gymnasium as gym
import math
import os
from road import build_center_line, nearest_point_index, POINT_SPACING, road_angle, side_offset

# World constants: the single source of truth, also imported by main.py
WIDTH, HEIGHT = 800, 600 # Set the width and height of the window
CAR_WIDTH = 30 # Define the width of the car
CAR_HEIGHT = 50 # Define the height of the car
LANE_WIDTH = CAR_WIDTH * 3 # Define the width of a lane, triple than the car
ROAD_WIDTH = LANE_WIDTH * 2 # Define the width of the road  
RIGHT_LANE_CENTER = WIDTH // 2 + LANE_WIDTH // 2 # Center of the right lane
LEFT_LANE_CENTER = WIDTH // 2 - LANE_WIDTH // 2 # Center of the left lane
ROAD_LENGTH = 5000  # Define the length of the road
ROAD_X = WIDTH // 2 - ROAD_WIDTH // 2 # Left edge of the road, centered horizontally
ROAD_ANGLE = 90 # The road is vertical, pointing up (0° = right, angles grow counterclockwise)
CAR_START_X = RIGHT_LANE_CENTER - CAR_WIDTH // 2 # Starting x position of the car, centered in the right lane
CAR_BOTTOM_MARGIN = 5 # Offset for the car's position from the bottom of the window
CAR_START_Y = HEIGHT - (CAR_HEIGHT + CAR_BOTTOM_MARGIN) # Starting y position of the car, at the bottom
CAR_START_ANGLE = ROAD_ANGLE # Starting angle of the car, aligned with the road
ROAD_PIECES = [(ROAD_LENGTH, 0)] # The road is a straight vertical line, defined by its length and angle

# Car physics: every value is expressed in seconds, then converted to steps (the game runs at 60 steps per second)
STEPS_PER_SECOND = 60 # Simulation steps in one second of game time
MAX_SPEED = 8 # Top speed, in pixels per step
ACCELERATION = MAX_SPEED / (3 * STEPS_PER_SECOND) # Full throttle: from 0 to MAX_SPEED in 3 seconds
BRAKE = MAX_SPEED / (2 * STEPS_PER_SECOND) # Full brake: from MAX_SPEED to 0 in 2 seconds
FRICTION = MAX_SPEED / (6 * STEPS_PER_SECOND) # No pedals: from MAX_SPEED to 0 in 6 seconds
STEER_RATE = 90 / (1 * STEPS_PER_SECOND) / MAX_SPEED # Full steering at MAX_SPEED: a 90° turn in 1 seconds (degrees per pixel traveled)

MAX_STEPS = (ROAD_LENGTH // MAX_SPEED) * 2 # Maximum number of steps before the episode ends

# Rewards and penalties: the single source of truth, also imported by test_car_env.py
OFF_ROAD_PENALTY = 5 # Penalty for going off the road
FINISH_REWARD = 20 # Reward for reaching the end of the road
ROAD_REWARD = 10 # Reward for reaching the end of the road, scaled by the road length
MAX_PROGRESS_PER_STEP = MAX_SPEED / ROAD_LENGTH * ROAD_REWARD # Progress reward of one step at top speed
WRONG_WAY_PENALTY = 2 * MAX_PROGRESS_PER_STEP # Each step in the opposite lane costs twice the best progress: wrong way never pays off
GRASS_PENALTY = 2 * MAX_PROGRESS_PER_STEP # Each step with wheels on the grass costs twice the best progress

# Drawing constants: only needed to show the world on screen (render)
GRASS_COLOR = (60, 140, 60) # Define the color for grass
ROAD_COLOR = (50, 50, 50) # Define the color for the road
LINE_COLOR = (255, 255, 255) # Define the color for lane lines
LINE_WIDTH = 3 # Define the width of the lane lines
LINE_OFFSET = 3 # Define the offset for lane lines from the edge of the road
DASH_LINE_LENGTH = CAR_HEIGHT // 2 # Define the length of the dashed lane lines
DASH_LINE_WIDTH = LINE_WIDTH # Define the width of the dashed lane lines
DASH_LINE_GAP = DASH_LINE_LENGTH // 2 # Define the gap between dashed lane lines
CAR_SCREEN_Y = CAR_START_Y # Fixed height on screen where the car is shown

CAR_SPRITE_PATH = os.path.join(os.path.dirname(__file__), "assets", "controlled_car.png") # Car image, found next to this file
CAR_SPRITE_ANGLE = 90 # Direction the car points to in the image (up)
CAR_SPRITE_QUALITY = 4 # Keep the car image 4x bigger than drawn: rotating a big image and shrinking it looks much smoother

def world_to_screen(rect, camera_y):
    return rect.move(0, -camera_y) # Convert world coordinates to screen coordinates based on the camera's y position


class CarEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60} # Gymnasium: supported render modes and speed
    #metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 240} # Gymnasium: supported render modes and speed, increased to 240 for faster training

    def __init__(self, render_mode=None):
        # what the agent can do and what it sees (action_space, observation_space)
        # objects that never change: road, lines
        self.render_mode = render_mode # None = no drawing (training), "human" = window, "rgb_array" = image for videos
        self.screen = None # Created only the first time render() is called
        self.clock = None

        self.road_points = build_center_line(ROAD_PIECES, ROAD_X + ROAD_WIDTH // 2, HEIGHT, ROAD_ANGLE) # Create the road's center line points based on the defined pieces
        self.road = pygame.Rect(ROAD_X, HEIGHT - ROAD_LENGTH, ROAD_WIDTH, ROAD_LENGTH) # Create a rectangle for the road
        self.left_line = pygame.Rect(ROAD_X + LINE_OFFSET, HEIGHT - ROAD_LENGTH, LINE_WIDTH, ROAD_LENGTH) # Create a rectangle for the left lane line
        self.right_line = pygame.Rect(ROAD_X + ROAD_WIDTH - LINE_OFFSET - LINE_WIDTH, HEIGHT - ROAD_LENGTH, LINE_WIDTH, ROAD_LENGTH) # Create a rectangle for the right lane line
        self.dash_lines = [] # Initialize a list to hold the dashed lane lines
        for y in range(self.road.bottom - DASH_LINE_LENGTH, self.road.top, -(DASH_LINE_LENGTH + DASH_LINE_GAP)): # Create dashed lane lines along the road
            dash_line = pygame.Rect(WIDTH // 2 - DASH_LINE_WIDTH // 2, y, DASH_LINE_WIDTH, DASH_LINE_LENGTH) # Create a rectangle for a dashed lane line
            self.dash_lines.append(dash_line) # Add the dashed lane line to the list
        self.car = pygame.Rect(CAR_START_X, CAR_START_Y, CAR_WIDTH, CAR_HEIGHT) # Create a rectangle for the car
        
        self.action_space = gym.spaces.MultiDiscrete([3, 3])         # Actions: [steering, pedals]. Steering: 0 = left, 1 = straight, 2 = right. Pedals: 0 = throttle, 1 = none, 2 = brake
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)  # Observations: [distance to left edge, distance to right edge, distance to center of right lane, speed, direction], all normalized between -1 and 1
        self.step_count = 0 # Initialize the step counter
        self.road_index = 0 # Index of the nearest point on the road to the car's center

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


        camera_y = self.car.top - CAR_SCREEN_Y # Camera follows the car, keeping it at a fixed height on screen

        self.screen.fill(GRASS_COLOR) # Fill the screen with green {grass}
        pygame.draw.rect(self.screen, ROAD_COLOR, world_to_screen(self.road, camera_y)) # Draw the road on the screen
        pygame.draw.rect(self.screen, LINE_COLOR, world_to_screen(self.left_line, camera_y)) # Draw the left lane
        pygame.draw.rect(self.screen, LINE_COLOR, world_to_screen(self.right_line, camera_y)) # Draw the right lane
        for dash_line in self.dash_lines: # Draw each dashed lane line
            pygame.draw.rect(self.screen, LINE_COLOR, world_to_screen(dash_line, camera_y)) # Draw a dashed lane line
        
        rotated = pygame.transform.rotozoom(self.car_sprite, self.car_angle - CAR_SPRITE_ANGLE, 1 / CAR_SPRITE_QUALITY) # 1. Rotate the image like the car (smooth edges) and shrink it to the car size
        car_center = world_to_screen(self.car, camera_y).center # 2. Where the car's center is on screen
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
        # compute the distances from the car to the road edges and the center of the right lane
        offset = self.center_offset() # Calculate the offset of the car from the center line of the road at the nearest point
        normalized_dist_left = (ROAD_WIDTH / 2 + offset - CAR_WIDTH / 2) / ROAD_WIDTH # Distance from the car's left side to the left edge of the road
        normalized_dist_right = (ROAD_WIDTH / 2 - offset - CAR_WIDTH / 2) / ROAD_WIDTH # Distance from the car's right side to the right edge of the road
        normalized_dist_center_right_lane = (offset - LANE_WIDTH / 2) / ROAD_WIDTH # Distance from the car's center to the center of the right lane
        normalized_speed = self.car_speed / MAX_SPEED # Normalize the speed
        current_road_angle = road_angle(self.road_points, self.road_index) # Direction of the road where the car is
        normalized_direction = ((self.car_angle - current_road_angle + 180) % 360 - 180) / 180 # Difference from the road direction, always between -1 and +1

        normalized_values = np.array([normalized_dist_left, normalized_dist_right, normalized_dist_center_right_lane, normalized_speed, normalized_direction], dtype=np.float32)

        return normalized_values # Return the normalized distances as the observation

    def reset(self, seed=None, options=None):
        # put the car back at the start (position, angle, speed), reset the step counter
        # return first observation, info
        super().reset(seed=seed)
        self.car.left = CAR_START_X
        self.car.top = CAR_START_Y
        self.car_x = CAR_START_X
        self.car_y = CAR_START_Y
        self.car_angle = CAR_START_ANGLE
        self.car_speed = 0
        self.step_count = 0
        self.road_index = nearest_point_index(self.road_points, self.car.centerx, self.car.centery) # Find the index of the nearest point on the road to the car's center
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

        if self.car_y + CAR_HEIGHT > self.road.bottom: # Wall at the bottom: block the car, the episode goes on
            self.car_y = self.road.bottom - CAR_HEIGHT # Put the car exactly against the wall
            self.car_speed = 0 # Stop the car if it hits the bottom wall
            
        self.car.left = round(self.car_x) # The Rect (whole pixels) follows the true position (decimals): used for drawing and edge checks
        self.car.top = round(self.car_y)

        new_road_index = nearest_point_index(self.road_points, self.car.centerx, self.car.centery) # Update the index of the nearest point on the road to the car's center
        progress = (new_road_index - self.road_index) * POINT_SPACING # Calculate the progress made along the road based on the change in road index
        self.road_index = new_road_index # Update the road index for the next step

        offset = self.center_offset() # Calculate the offset of the car from the center line of the road at the nearest point

        reward = progress / ROAD_LENGTH * ROAD_REWARD # Reward for moving forward, scaled by the road length
        reward -= 0.01 * abs(offset - LANE_WIDTH / 2) / (LANE_WIDTH // 2) # Small penalty for being away from the center of the right lane, scaled by half the lane width
        
        if abs(offset) > ROAD_WIDTH / 2 - CAR_WIDTH / 2: # Off road: part of the car is outside the road boundaries
            reward -= GRASS_PENALTY # Add a penalty for going off the road

        if offset - CAR_WIDTH / 2 < 0: # Wrong way: part of the car is in the opposite lane
            reward -= WRONG_WAY_PENALTY # Add a penalty for being in the wrong lane

        terminated = False # Initialize the terminated flag to False
        truncated = False # Initialize the truncated flag to False

        if abs(offset) > ROAD_WIDTH / 2: # Off road: episode fails
            terminated = True # Set the terminated flag to True
            reward -= OFF_ROAD_PENALTY # Add a penalty for going off the road

        if self.road_index == len(self.road_points) - 1: # Reached the end of the road: episode succeeds
            terminated = True # Set the terminated flag to True
            reward += FINISH_REWARD # Add a reward for successfully reaching the end of the road

        self.step_count += 1 # Increment the step counter
        if self.step_count >= MAX_STEPS: # Check if the maximum number of steps has been reached
            truncated = True

        if self.render_mode == "human": # Gymnasium convention: in human mode the window updates by itself
            self.render()

        return self.get_obs(), reward, terminated, truncated, {} # Return the observation, reward, terminated flag, truncated flag, and an empty info dictionary
