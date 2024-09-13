import pygame

class Flag:
    def __init__(self, x, y, grid_size):
        self.x = x
        self.y = y
        self.grid_size = grid_size
        self.carried_by = None  # Agent who has the flag

    def draw(self, surface):
        if not self.carried_by:
            rect = pygame.Rect(
                self.x * self.grid_size + self.grid_size // 4,
                self.y * self.grid_size + self.grid_size // 4,
                self.grid_size // 2,
                self.grid_size // 2
            )
            pygame.draw.rect(surface, (255, 215, 0), rect)  # Gold color for the flag

    def reset(self, x, y):
        self.x = x
        self.y = y
        self.carried_by = None

class Obstacle:
    def __init__(self, x, y, grid_size):
        self.x = x
        self.y = y
        self.grid_size = grid_size

    def draw(self, surface):
        rect = pygame.Rect(
            self.x * self.grid_size,
            self.y * self.grid_size,
            self.grid_size,
            self.grid_size
        )
        pygame.draw.rect(surface, (128, 128, 128), rect)  # Gray color for obstacles

def draw_grid(surface, width, height, grid_size):
    for x in range(0, width, grid_size):
        pygame.draw.line(surface, (200, 200, 200), (x, 0), (x, height))
    for y in range(0, height, grid_size):
        pygame.draw.line(surface, (200, 200, 200), (0, y), (width, y))

def draw_scores(surface, scores):
    font = pygame.font.SysFont(None, 36)
    score_text = f"Agent A: {scores['Agent A']} | Agent B: {scores['Agent B']}"
    img = font.render(score_text, True, (255, 255, 255))
    surface.blit(img, (10, 10))

def draw_visualization_placeholder(surface, width, height, visualization_height):
    rect = pygame.Rect(0, height - visualization_height, width, visualization_height)
    pygame.draw.rect(surface, (50, 50, 50), rect)  # Dark gray background
    font = pygame.font.SysFont(None, 24)
    text = "Training Data Visualizations (Placeholder)"
    img = font.render(text, True, (200, 200, 200))
    surface.blit(img, (10, height - visualization_height + 10))

def draw_scoring_zones(surface, zone_a, zone_b, grid_size):
    # Draw Agent A's scoring zone
    for x, y in zone_a:
        rect = pygame.Rect(
            x * grid_size,
            y * grid_size,
            grid_size,
            grid_size
        )
        pygame.draw.rect(surface, (144, 238, 144), rect)  # Light green color

    # Draw Agent B's scoring zone
    for x, y in zone_b:
        rect = pygame.Rect(
            x * grid_size,
            y * grid_size,
            grid_size,
            grid_size
        )
        pygame.draw.rect(surface, (144, 238, 144), rect)  # Light green color
