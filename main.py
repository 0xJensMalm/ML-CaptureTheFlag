import pygame
import sys
from agents import AgentA, AgentB
from game_objects import Flag, Obstacle, draw_grid, draw_scoring_zones
from visualization import draw_visualizations

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 1000, 950  # Increased height to accommodate score and controls
GRID_SIZE = 20  # Size of each grid square
HEADER_HEIGHT = 50  # Height for the header containing scores and speed controls
GAME_HEIGHT = 500  # Height of the game view
VISUALIZATION_HEIGHT = HEIGHT - HEADER_HEIGHT - GAME_HEIGHT  # Height reserved for visualizations
ROWS = GAME_HEIGHT // GRID_SIZE
COLS = WIDTH // GRID_SIZE

# Set up the display
window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Capture the Flag Simulation with ML Monitoring')

# Agent starting positions
agent_a_start = (1, ROWS // 2)
agent_b_start = (COLS - 2, ROWS // 2)

# Create agents
agent_a = AgentA(*agent_a_start, (0, 0, 255), GRID_SIZE, ROWS, COLS)
agent_b = AgentB(*agent_b_start, (255, 0, 0), GRID_SIZE, ROWS, COLS)

# Create flag at the center
flag = Flag(COLS // 2, ROWS // 2, GRID_SIZE)

# Create obstacles
obstacles = []

# Add perimeter obstacles (grid outline)
for x in range(COLS):
    obstacles.append(Obstacle(x, 0, GRID_SIZE))  # Top row
    obstacles.append(Obstacle(x, ROWS - 1, GRID_SIZE))  # Bottom row

for y in range(ROWS):
    obstacles.append(Obstacle(0, y, GRID_SIZE))  # Left column
    obstacles.append(Obstacle(COLS - 1, y, GRID_SIZE))  # Right column

# Add symmetric obstacles between agents and the flag
# Example: Vertical walls on both sides
for y in range(ROWS // 2 - 3, ROWS // 2 + 4):
    obstacles.append(Obstacle(COLS // 2 - 5, y, GRID_SIZE))  # Left of the flag
    obstacles.append(Obstacle(COLS // 2 + 5, y, GRID_SIZE))  # Right of the flag

# Define scoring zones for each agent (2x2 area)
agent_a_scoring_zone = [(agent_a_start[0] + dx, agent_a_start[1] + dy) for dx in range(2) for dy in range(-1, 1)]
agent_b_scoring_zone = [(agent_b_start[0] - dx, agent_b_start[1] + dy) for dx in range(2) for dy in range(-1, 1)]

# Scores
scores = {'Agent A': 0, 'Agent B': 0}

clock = pygame.time.Clock()
running = True
paused = False
simulation_speed = 60  # Frames per second

episode = 0  # Keep track of episodes

# Speed control buttons
speed_buttons = [
    {'label': '0x', 'speed': 0, 'rect': pygame.Rect(WIDTH // 2 - 110, 10, 40, 30)},
    {'label': '1x', 'speed': 60, 'rect': pygame.Rect(WIDTH // 2 - 60, 10, 40, 30)},
    {'label': '2x', 'speed': 120, 'rect': pygame.Rect(WIDTH // 2 - 10, 10, 40, 30)},
    {'label': '5x', 'speed': 300, 'rect': pygame.Rect(WIDTH // 2 + 40, 10, 40, 30)},
    {'label': '10x', 'speed': 600, 'rect': pygame.Rect(WIDTH // 2 + 90, 10, 50, 30)},
]

def draw_header(surface, width, height, scores):
    font = pygame.font.SysFont(None, 24)

    # Draw Model A's score
    agent_a_score_text = font.render(f"Model A: {scores['Agent A']}", True, (255, 255, 255))
    surface.blit(agent_a_score_text, (width // 4 - agent_a_score_text.get_width() // 2, 15))

    # Draw Model B's score
    agent_b_score_text = font.render(f"Model B: {scores['Agent B']}", True, (255, 255, 255))
    surface.blit(agent_b_score_text, (3 * width // 4 - agent_b_score_text.get_width() // 2, 15))

    # Draw speed control buttons
    for button in speed_buttons:
        pygame.draw.rect(surface, (70, 70, 70), button['rect'])
        label = font.render(button['label'], True, (255, 255, 255))
        label_rect = label.get_rect(center=button['rect'].center)
        surface.blit(label, label_rect)
        # Highlight the current speed
        if simulation_speed == button['speed']:
            pygame.draw.rect(surface, (255, 255, 0), button['rect'], 2)

while running:
    clock.tick(60)  # Run at 60 FPS to handle events smoothly
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # Controls
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = event.pos
            for button in speed_buttons:
                if button['rect'].collidepoint(mouse_pos):
                    if button['speed'] == 0:
                        paused = True
                    else:
                        paused = False
                        simulation_speed = button['speed']
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                paused = not paused

    if not paused and simulation_speed > 0:
        # Create a shared representation of the environment
        env = {
            'agents': [agent_a, agent_b],
            'flag': flag,
            'obstacles': obstacles,
            'scoring_zones': {
                'Agent A': agent_a_scoring_zone,
                'Agent B': agent_b_scoring_zone
            },
            'scores': scores
        }

        for _ in range(simulation_speed // 60):  # Adjust updates based on speed
            # Agent A's turn
            reward_a, done_a = agent_a.update(env)
            # Agent B's turn
            reward_b, done_b = agent_b.update(env)

            # Check if any agent has scored
            if done_a or done_b:
                episode += 1
                if done_a:
                    print(f"Episode {episode}: {agent_a.name} scored! Total scores: {scores}")
                if done_b:
                    print(f"Episode {episode}: {agent_b.name} scored! Total scores: {scores}")

                # Reset environment
                agent_a.reset(*agent_a_start)
                agent_b.reset(*agent_b_start)
                flag.reset(COLS // 2, ROWS // 2)

                # Log episode rewards
                agent_a.episode_rewards.append(agent_a.total_reward)
                agent_b.episode_rewards.append(agent_b.total_reward)
                agent_a.total_reward = 0
                agent_b.total_reward = 0

    # Drawing
    window.fill((0, 0, 0))  # Clear screen with black

    # Draw the header (scores and speed control)
    header_surface = pygame.Surface((WIDTH, HEADER_HEIGHT))
    header_surface.fill((50, 50, 50))
    draw_header(header_surface, WIDTH, HEADER_HEIGHT, scores)
    window.blit(header_surface, (0, 0))

    # Draw the game view
    game_surface = pygame.Surface((WIDTH, GAME_HEIGHT))
    draw_grid(game_surface, WIDTH, GAME_HEIGHT, GRID_SIZE)
    draw_scoring_zones(game_surface, agent_a_scoring_zone, agent_b_scoring_zone, GRID_SIZE)
    for obstacle in obstacles:
        obstacle.draw(game_surface)
    flag.draw(game_surface)
    agent_a.draw(game_surface)
    agent_b.draw(game_surface)
    window.blit(game_surface, (0, HEADER_HEIGHT))

    # Draw the visualizations
    visualization_surface = pygame.Surface((WIDTH, VISUALIZATION_HEIGHT))
    draw_visualizations(visualization_surface, agent_a, agent_b, WIDTH, VISUALIZATION_HEIGHT)
    window.blit(visualization_surface, (0, HEADER_HEIGHT + GAME_HEIGHT))

    pygame.display.flip()

pygame.quit()
sys.exit()
