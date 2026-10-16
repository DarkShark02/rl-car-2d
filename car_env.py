import pygame
import numpy as np
import gymnasium as gym

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
CAR_START_X = RIGHT_LANE_CENTER - CAR_WIDTH // 2 # Starting x position of the car, centered in the right lane
CAR_BOTTOM_MARGIN = 5 # Offset for the car's position from the bottom of the window
CAR_START_Y = HEIGHT - (CAR_HEIGHT + CAR_BOTTOM_MARGIN) # Starting y position of the car, at the bottom
CAR_SPEED = 5 # Define the speed of the car
MAX_STEPS = (ROAD_LENGTH // CAR_SPEED) * 2 # Maximum number of steps before the episode ends

# Drawing constants: only needed to show the world on screen (render)
GRASS_COLOR = (60, 140, 60) # Define the color for grass
ROAD_COLOR = (50, 50, 50) # Define the color for the road
LINE_COLOR = (255, 255, 255) # Define the color for lane lines
LINE_WIDTH = 3 # Define the width of the lane lines
LINE_OFFSET = 3 # Define the offset for lane lines from the edge of the road
CAR_COLOR = (255, 0, 0) # Define the color for the car
DASH_LINE_LENGTH = CAR_HEIGHT // 2 # Define the length of the dashed lane lines
DASH_LINE_WIDTH = LINE_WIDTH # Define the width of the dashed lane lines
DASH_LINE_GAP = DASH_LINE_LENGTH // 2 # Define the gap between dashed lane lines
CAR_SCREEN_Y = CAR_START_Y # Fixed height on screen where the car is shown


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

        self.road = pygame.Rect(ROAD_X, HEIGHT - ROAD_LENGTH, ROAD_WIDTH, ROAD_LENGTH) # Create a rectangle for the road
        self.left_line = pygame.Rect(ROAD_X + LINE_OFFSET, HEIGHT - ROAD_LENGTH, LINE_WIDTH, ROAD_LENGTH) # Create a rectangle for the left lane line
        self.right_line = pygame.Rect(ROAD_X + ROAD_WIDTH - LINE_OFFSET - LINE_WIDTH, HEIGHT - ROAD_LENGTH, LINE_WIDTH, ROAD_LENGTH) # Create a rectangle for the right lane line
        self.dash_lines = [] # Initialize a list to hold the dashed lane lines
        for y in range(self.road.bottom - DASH_LINE_LENGTH, self.road.top, -(DASH_LINE_LENGTH + DASH_LINE_GAP)): # Create dashed lane lines along the road
            dash_line = pygame.Rect(WIDTH // 2 - DASH_LINE_WIDTH // 2, y, DASH_LINE_WIDTH, DASH_LINE_LENGTH) # Create a rectangle for a dashed lane line
            self.dash_lines.append(dash_line) # Add the dashed lane line to the list
        self.car = pygame.Rect(CAR_START_X, CAR_START_Y, CAR_WIDTH, CAR_HEIGHT) # Create a rectangle for the car
        
        self.action_space = gym.spaces.MultiDiscrete([3, 3])         # Actions: 0 = left, 1 = stay, 2 = right for x; same for y (0 = forward, 1 = stay, 2 = back)
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)         # Observation: dist_left, dist_right, lane offset, all divided by ROAD_WIDTH so they always stay within -1..1
        self.step_count = 0 # Initialize the step counter

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

        camera_y = self.car.top - CAR_SCREEN_Y # Camera follows the car, keeping it at a fixed height on screen

        self.screen.fill(GRASS_COLOR) # Fill the screen with green {grass}
        pygame.draw.rect(self.screen, ROAD_COLOR, world_to_screen(self.road, camera_y)) # Draw the road on the screen
        pygame.draw.rect(self.screen, LINE_COLOR, world_to_screen(self.left_line, camera_y)) # Draw the left lane
        pygame.draw.rect(self.screen, LINE_COLOR, world_to_screen(self.right_line, camera_y)) # Draw the right lane
        for dash_line in self.dash_lines: # Draw each dashed lane line
            pygame.draw.rect(self.screen, LINE_COLOR, world_to_screen(dash_line, camera_y)) # Draw a dashed lane line
        pygame.draw.rect(self.screen, CAR_COLOR, world_to_screen(self.car, camera_y)) # Draw the car on the screen

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

    def lane_offset(self):
        # signed distance in pixels from the car's center to the center of the right lane (negative = left)
        return self.car.centerx - RIGHT_LANE_CENTER

    def get_obs(self):
        # compute the distances from the car to the road edges and the center of the right lane
        dist_left = self.car.left - self.road.left # Calculate the distance from the car to the left edge of the road
        dist_right = self.road.right - self.car.right # Calculate the distance from the car to the right edge of the road
        dist_center_right_lane = self.lane_offset() # Calculate the distance from the car's center to the center of the right lane

        # Every value is divided by ROAD_WIDTH: no distance on the road can be larger than that, so the result stays within -1..1
        normalized_values = np.array([dist_left, dist_right, dist_center_right_lane], dtype=np.float32) / ROAD_WIDTH # Normalize the distances
        return normalized_values # Return the normalized distances as the observation

    def reset(self, seed=None, options=None):
        # put the car back at the start, reset the step counter
        # return first observation, info
        super().reset(seed=seed)
        self.car.left = CAR_START_X
        self.car.top = CAR_START_Y
        self.step_count = 0
        if self.render_mode == "human": # Gymnasium convention: in human mode the window updates by itself
            self.render()
        return self.get_obs(), {}

    def step(self, action):
        # action is a tuple of two values: (x_action, y_action)
        dx = action[0] - 1 # calculate the change in x position based on the action (0 = left, 1 = stay, 2 = right)
        dy = action[1] - 1 # calculate the change in y position based on the action (0 = forward, 1 = stay, 2 = back)
        
        prev_top = self.car.top # Store the previous top position of the car for reward calculation

        self.car.left += dx * CAR_SPEED # Update the car's x position based on the input and speed
        self.car.top += dy * CAR_SPEED # Update the car's y position

        if self.car.bottom > self.road.bottom: # Wall at the bottom: block the car, the episode goes on
            self.car.bottom = self.road.bottom # Keep the car within the bottom boundary of the road

        reward = (prev_top - self.car.top) / ROAD_LENGTH * 10 # Reward for moving forward, scaled by the road length
        reward -= 0.01 * abs(self.lane_offset()) / (LANE_WIDTH // 2) # Small penalty for being away from the center of the right lane, scaled by half the lane width

        terminated = False # Initialize the terminated flag to False
        truncated = False # Initialize the truncated flag to False

        if self.car.left < self.road.left or self.car.right > self.road.right: # Off road: episode fails
            terminated = True # Set the terminated flag to True
            reward -= 5 # Add a penalty for going off the road

        if self.car.top < self.road.top: # Reached the end of the road: episode succeeds
            terminated = True # Set the terminated flag to True
            reward += 20 # Add a reward for successfully reaching the end of the road

        self.step_count += 1 # Increment the step counter
        if self.step_count >= MAX_STEPS: # Check if the maximum number of steps has been reached
            truncated = True

        if self.render_mode == "human": # Gymnasium convention: in human mode the window updates by itself
            self.render()

        return self.get_obs(), reward, terminated, truncated, {} # Return the observation, reward, terminated flag, truncated flag, and an empty info dictionary
