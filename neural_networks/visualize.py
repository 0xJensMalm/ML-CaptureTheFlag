import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-GUI rendering

import matplotlib.pyplot as plt
import numpy as np

def visualize_game_state(grid):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid, cmap='Pastel1', interpolation='nearest')

    # Marking the elements on the grid
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] == 3:  # Flag
                ax.text(j, i, 'F', ha='center', va='center', fontsize=20, color='red')
            elif grid[i, j] == 2:  # Team 1
                ax.text(j, i, '1', ha='center', va='center', fontsize=20, color='blue')
            elif grid[i, j] == 4:  # Team 2
                ax.text(j, i, '2', ha='center', va='center', fontsize=20, color='green')
            elif grid[i, j] == 1:  # Obstacle
                ax.text(j, i, 'X', ha='center', va='center', fontsize=20, color='black')

    ax.set_xticks([])  # Hide x-axis ticks
    ax.set_yticks([])  # Hide y-axis ticks
    return fig  # Return the figure instead of showing it
