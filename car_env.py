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

class CarEnv(gym.Env):
    def __init__(self):
        # what the agent can do and what it sees (action_space, observation_space)
        # objects that never change: road, lines
        self.road = pygame.Rect(ROAD_X, HEIGHT - ROAD_LENGTH, ROAD_WIDTH, ROAD_LENGTH) # Create a rectangle for the road
        self.car = pygame.Rect(CAR_START_X, CAR_START_Y, CAR_WIDTH, CAR_HEIGHT) # Create a rectangle for the car

    def get_obs(self):
        # compute the distances from the car to the road edges and the center of the right lane
        dist_left = self.car.left - self.road.left # Calculate the distance from the car to the left edge of the road
        dist_right = self.road.right - self.car.right # Calculate the distance from the car to the right edge of the road
        car_center = self.car.left + self.car.width // 2 # Calculate the center x position of the car
        dist_center_right_lane = car_center - RIGHT_LANE_CENTER # Calculate the distance from the car's center to the center of the right lane
        
        normalized_values = np.array([dist_left, dist_right, dist_center_right_lane], dtype=np.float32) / np.array([ROAD_WIDTH, ROAD_WIDTH, LANE_WIDTH // 2], dtype=np.float32) # Normalize the distances
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
        # same as your update section, then compute the reward
        # return observation, reward, terminated, truncated, info
        pass