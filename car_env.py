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

class CarEnv(gym.Env):
    def __init__(self):
        # what the agent can do and what it sees (action_space, observation_space)
        # objects that never change: road, lines
        self.road = pygame.Rect(ROAD_X, HEIGHT - ROAD_LENGTH, ROAD_WIDTH, ROAD_LENGTH) # Create a rectangle for the road
        self.car = pygame.Rect(CAR_START_X, CAR_START_Y, CAR_WIDTH, CAR_HEIGHT) # Create a rectangle for the car
        
        self.action_space = gym.spaces.MultiDiscrete([3, 3])         # Actions: 0 = left, 1 = stay, 2 = right for x; same for y (0 = forward, 1 = stay, 2 = back)
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)         # Observation: dist_left, dist_right, lane offset, all divided by ROAD_WIDTH so they always stay within -1..1
        self.step_count = 0 # Initialize the step counter

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

        return self.get_obs(), reward, terminated, truncated, {} # Return the observation, reward, terminated flag, truncated flag, and an empty info dictionary
