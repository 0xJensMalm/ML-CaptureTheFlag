import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-GUI rendering

import matplotlib.pyplot as plt
import numpy as np

def visualize_game_state(grid, team1_points, team2_points):
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='#1e1e1e')
    ax.imshow(grid, cmap='RdYlBu', interpolation='nearest', vmin=0, vmax=4)

    # Marking the elements on the grid
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] == 3:  # Flag
                ax.text(j, i, '🚩', ha='center', va='center', fontsize=10, color='white')
            elif grid[i, j] == 2:  # Team 1
                ax.text(j, i, '1', ha='center', va='center', fontsize=10, color='white')
            elif grid[i, j] == 4:  # Team 2
                ax.text(j, i, '2', ha='center', va='center', fontsize=10, color='white')
            elif grid[i, j] == 1:  # Static base and obstacle
                ax.text(j, i, '■', ha='center', va='center', fontsize=10, color='white')

    ax.set_xticks([])  # Hide x-axis ticks
    ax.set_yticks([])  # Hide y-axis ticks
    ax.set_facecolor('#1e1e1e')
    plt.tight_layout()
    return fig