import pygame

WIDTH, HEIGHT = 800, 600 # Set the width and height of the window
GRASS_COLOR = (60, 140, 60) # Define the color for grass 
ROAD_COLOR = (50, 50, 50) # Define the color for the road
ROAD_WIDTH = 120 # Define the width of the road
ROAD_HEIGHT = HEIGHT # Define the height of the road
ROAD_X = WIDTH // 2 - ROAD_WIDTH // 2 # Left edge of the road, centered horizontally
CAR_COLOR = (255, 0, 0) # Define the color for the car
CAR_WIDTH = 30 # Define the width of the car
CAR_HEIGHT = 50 # Define the height of the car
CAR_START_X = WIDTH // 2 - CAR_WIDTH // 2 # Starting x position of the car, centered horizontally
CAR_BOTTOM_MARGIN = 5 # Offset for the car's position from the bottom of the window
CAR_START_Y = HEIGHT - (CAR_HEIGHT + CAR_BOTTOM_MARGIN) # Starting y position of the car, at the bottom
CAR_SPEED = 5 # Define the speed of the car


pygame.init() # Initialize Pygame
screen = pygame.display.set_mode((WIDTH, HEIGHT)) # Create the window
pygame.display.set_caption("RL Car 2D") # Set the window title
clock = pygame.time.Clock() # Create a clock to control the frame rate

road = pygame.Rect(ROAD_X, 0, ROAD_WIDTH, ROAD_HEIGHT) # Create a rectangle for the road
car = pygame.Rect(CAR_START_X, CAR_START_Y, CAR_WIDTH, CAR_HEIGHT) # Create a rectangle for the car

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
    car.x += dx * CAR_SPEED # Update the car's x position based on the input and speed
    car.y += dy * CAR_SPEED # Update the car's y position based on the input and speed

    # 3 draw
    screen.fill(GRASS_COLOR) # Fill the screen with green {grass}

    pygame.draw.rect(screen, ROAD_COLOR, road) # Draw the road rectangle on the screen
    
    pygame.draw.rect(screen, CAR_COLOR, car) # Draw the car rectangle on the screen

    pygame.display.flip() # Update the display
    clock.tick(60) # Limit the frame rate to 60 FPS

pygame.quit() # Quit Pygame