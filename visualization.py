import pygame

def draw_visualizations(surface, agent_a, agent_b, width, height):
    half_width = width // 2
    # Background
    surface.fill((50, 50, 50))

    # Draw Agent A's metrics on the left
    draw_agent_metrics(
        surface,
        agent_a,
        x=0,
        y=0,
        width=half_width,
        height=height,
        title="Agent A Metrics"
    )

    # Draw Agent B's metrics on the right
    draw_agent_metrics(
        surface,
        agent_b,
        x=half_width,
        y=0,
        width=half_width,
        height=height,
        title="Agent B Metrics"
    )

def draw_agent_metrics(surface, agent, x, y, width, height, title):
    # Draw title
    font = pygame.font.SysFont(None, 24)
    title_text = font.render(title, True, (255, 255, 255))
    surface.blit(title_text, (x + 10, y + 10))

    # Calculate space for each graph
    graph_height = (height - 40) // 3  # Subtract space for title and margins

    # Draw average reward graph
    draw_line_graph(
        surface,
        agent.episode_rewards,
        x,
        y + 40,
        width,
        graph_height,
        label="Avg Reward"
    )

    # Draw loss graph
    draw_line_graph(
        surface,
        agent.losses,
        x,
        y + 40 + graph_height,
        width,
        graph_height,
        label="Loss"
    )

    # Draw epsilon graph
    draw_line_graph(
        surface,
        agent.epsilon_values,
        x,
        y + 40 + 2 * graph_height,
        width,
        graph_height,
        label="Epsilon"
    )

def draw_line_graph(surface, data, x, y, width, height, label):
    # Background for the graph
    pygame.draw.rect(surface, (30, 30, 30), (x + 10, y, width - 20, height - 10))

    # Draw label
    font = pygame.font.SysFont(None, 20)
    label_text = font.render(label, True, (255, 255, 255))
    surface.blit(label_text, (x + 15, y + 5))

    if len(data) < 2:
        return

    # Normalize data
    max_value = max(data)
    min_value = min(data)
    range_value = max_value - min_value if max_value != min_value else 1

    scaled_data = [
        (i, height - 20 - ((value - min_value) / range_value) * (height - 40))
        for i, value in enumerate(data[-(width - 40):])
    ]

    # Adjust x positions to fit within the graph width
    x_scale = (width - 40) / max(1, len(scaled_data) - 1)
    points = [
        (x + 20 + idx * x_scale, y + y_offset + 10)
        for idx, (i, y_offset) in enumerate(scaled_data)
    ]

    # Draw the lines
    if len(points) > 1:
        pygame.draw.lines(surface, (0, 255, 0), False, points, 2)
