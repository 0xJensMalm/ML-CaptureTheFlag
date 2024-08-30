import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-GUI rendering

import matplotlib.pyplot as plt
import numpy as np

def visualize_game_state(grid, team1_points, team2_points):
    fig, ax = plt.subplots(figsize=(12, 6))  # Adjust size for horizontal expansion
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
            elif grid[i, j] == 1:  # Static base and obstacle
                ax.text(j, i, 'X', ha='center', va='center', fontsize=20, color='black')

    # Display the scores at the top of the grid
    ax.text(0.5, -0.1, f"Team 1 Points: {team1_points}", ha='center', va='center', fontsize=20, color='blue', transform=ax.transAxes)
    ax.text(0.5, -0.15, f"Team 2 Points: {team2_points}", ha='center', va='center', fontsize=20, color='green', transform=ax.transAxes)

    ax.set_xticks([])  # Hide x-axis ticks
    ax.set_yticks([])  # Hide y-axis ticks
    return fig  # Return the figure instead of showing it
