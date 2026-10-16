import pygame

from car_env import ( # World and drawing constants, defined once in car_env.py
    WIDTH, HEIGHT, CAR_WIDTH, CAR_HEIGHT, ROAD_WIDTH, ROAD_LENGTH, ROAD_X,
    RIGHT_LANE_CENTER, CAR_START_X, CAR_START_Y, CAR_SPEED,
    GRASS_COLOR, ROAD_COLOR, LINE_COLOR, LINE_WIDTH, LINE_OFFSET, CAR_COLOR,
    DASH_LINE_LENGTH, DASH_LINE_WIDTH, DASH_LINE_GAP, CAR_SCREEN_Y,
    world_to_screen,
)

pygame.init() # Initialize Pygame
screen = pygame.display.set_mode((WIDTH, HEIGHT)) # Create the window
pygame.display.set_caption("RL Car 2D") # Set the window title
clock = pygame.time.Clock() # Create a clock to control the frame rate

road = pygame.Rect(ROAD_X, HEIGHT - ROAD_LENGTH, ROAD_WIDTH, ROAD_LENGTH) # Create a rectangle for the road
car = pygame.Rect(CAR_START_X, CAR_START_Y, CAR_WIDTH, CAR_HEIGHT) # Create a rectangle for the car
left_line = pygame.Rect(ROAD_X + LINE_OFFSET, HEIGHT - ROAD_LENGTH, LINE_WIDTH, ROAD_LENGTH) # Create a rectangle for the left lane line
right_line = pygame.Rect(ROAD_X + ROAD_WIDTH - LINE_OFFSET - LINE_WIDTH, HEIGHT - ROAD_LENGTH, LINE_WIDTH, ROAD_LENGTH) # Create a rectangle for the right lane line
dash_lines = [] # Initialize a list to hold the dashed lane lines

for y in range(road.bottom - DASH_LINE_LENGTH, road.top, -(DASH_LINE_LENGTH + DASH_LINE_GAP)): # Create dashed lane lines along the road
    dash_line = pygame.Rect(WIDTH // 2 - DASH_LINE_WIDTH // 2, y, DASH_LINE_WIDTH, DASH_LINE_LENGTH) # Create a rectangle for a dashed lane line
    dash_lines.append(dash_line) # Add the dashed lane line to the list

running = True # Set the running flag to True
while running: # Main game loop
    # 1 input 
    for event in pygame.event.get(): # Get all events
        if event.type == pygame.QUIT: # If the user closes the window
            running = False # Set the running flag to False

    dx, dy = 0, 0 # Initialize the change in x and y positions to 0
    keys = pygame.key.get_pressed() # Get the state of all keyboard keys
    if keys[pygame.K_LEFT] or keys[pygame.K_a]: # If the left arrow key or 'a' key is pressed
        dx -= 1
    
    if keys[pygame.K_RIGHT] or keys[pygame.K_d]: # If the right arrow key or 'd' key is pressed
        dx += 1
    
    if keys[pygame.K_UP] or keys[pygame.K_w]: # If the up arrow key or 'w' key is pressed
        dy -= 1
    
    if keys[pygame.K_DOWN] or keys[pygame.K_s]: # If the down arrow key or 's' key is pressed
        dy += 1

    # 2 update
    car.left += dx * CAR_SPEED # Update the car's x position based on the input and speed
    car.top += dy * CAR_SPEED # Update the car's y position based on the input and speed

    if car.left < road.left or car.right > road.right: # Off road: episode fails, restart
        car.left = CAR_START_X # Move the car back to the start position
        car.top = CAR_START_Y # Move the car back to the start position

    if car.top < road.top: # Reached the end of the road: episode succeeds, restart
        car.left = CAR_START_X # Move the car back to the start position
        car.top = CAR_START_Y # Move the car back to the start position

    if car.bottom > road.bottom: # Wall at the bottom: block the car, the episode goes on
        car.bottom = road.bottom # Keep the car within the bottom boundary of the road

    dist_left = car.left - road.left # Calculate the distance from the car to the left edge of the road
    dist_right = road.right - car.right # Calculate the distance from the car to the right edge of the road
    car_center = car.left + car.width // 2 # Calculate the center x position of the car
    dist_center_right_lane = car_center - RIGHT_LANE_CENTER # Calculate the distance from the car's center to the center of the right lane

    print(f"dist_left: {dist_left}, dist_right: {dist_right}, dist_center_right_lane: {dist_center_right_lane}") # Print the distances for debugging

    # 3 draw
    camera_y = car.top - CAR_SCREEN_Y # Camera follows the car, keeping it at a fixed height on screen

    screen.fill(GRASS_COLOR) # Fill the screen with green {grass}
    
    pygame.draw.rect(screen, ROAD_COLOR, world_to_screen(road, camera_y)) # Draw the road on the screen
    
    pygame.draw.rect(screen, LINE_COLOR, world_to_screen(left_line, camera_y)) # Draw the left lane

    pygame.draw.rect(screen, LINE_COLOR, world_to_screen(right_line, camera_y)) # Draw the right lane

    for dash_line in dash_lines: # Draw each dashed lane line
        pygame.draw.rect(screen, LINE_COLOR, world_to_screen(dash_line, camera_y))

    pygame.draw.rect(screen, CAR_COLOR, world_to_screen(car, camera_y)) # Draw the car on the screen

    pygame.display.flip() # Update the display
    clock.tick(60) # Limit the frame rate to 60 FPS

pygame.quit() # Quit Pygame

