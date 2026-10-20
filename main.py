import pygame

from car_env import CarEnv

env = CarEnv(render_mode="human") # Create the environment with rendering enabled

env.reset() # Start a new episode
total_reward = 0
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
    obs, reward, terminated, truncated, info = env.step([2 * (dx + 1), dy + 1]) # Arrows steer fully: dx -1/0/+1 = steering index 0/2/4 (full left, straight, full right). Update the environment based on the input actions (dx, dy) and get the new observation, reward, and termination status
    total_reward += reward # Accumulate the total reward

    if terminated or truncated: # If the episode has ended
        print(f"Episode finished after {env.step_count} steps, with reward: {total_reward:.2f}, because: {info['reason']}") # Steps, total reward and why it ended
        env.reset() # Reset the environment for a new episode
        total_reward = 0 # Reset the total reward for the new episode

env.close() # Close the environment and free resources when done

