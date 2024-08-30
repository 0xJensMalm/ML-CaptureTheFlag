import numpy as np

class CaptureTheFlagEnv:
    def __init__(self, grid_size=(10, 10), num_obstacles=5):
        self.grid_size = grid_size
        self.num_obstacles = num_obstacles
        self.reset()

    def reset(self):
        self.grid = np.zeros(self.grid_size)  # Empty grid
        self.place_flag()
        self.place_obstacles()
        self.place_players()
        return self.grid

    def place_flag(self):
        # Place the flag at the center of the grid
        flag_position = (self.grid_size[0] // 2, self.grid_size[1] // 2)
        self.grid[flag_position] = 3  # 3 represents the flag

    def place_obstacles(self):
        # Randomly place obstacles on the grid
        for _ in range(self.num_obstacles):
            obstacle_position = self.random_empty_position()
            self.grid[obstacle_position] = 1  # 1 represents an obstacle

    def place_players(self):
        # Place the two teams at opposite corners
        self.player_positions = {
            'team1': (0, 0),
            'team2': (self.grid_size[0] - 1, self.grid_size[1] - 1)
        }
        self.grid[self.player_positions['team1']] = 2  # 2 represents team1
        self.grid[self.player_positions['team2']] = 4  # 4 represents team2

    def random_empty_position(self):
        while True:
            position = (np.random.randint(0, self.grid_size[0]), np.random.randint(0, self.grid_size[1]))
            if self.grid[position] == 0:
                return position

    def step(self, team, action):
        current_position = self.player_positions[team]
        new_position = self.get_new_position(current_position, action)

        if new_position is None or not self.is_valid_position(new_position):
            # Invalid move (into an obstacle, outside the grid, or invalid action)
            return self.grid, -1, False

        # Move the player
        self.grid[current_position] = 0  # Clear the old position
        self.grid[new_position] = 2 if team == 'team1' else 4  # Update the new position
        self.player_positions[team] = new_position

        # Check if the player has captured the flag
        reward = 1 if self.grid[new_position] == 3 else 0
        done = reward > 0
        return self.grid, reward, done

    def get_new_position(self, position, action):
        # Action: 0 = up, 1 = down, 2 = left, 3 = right
        if action == 0:  # Up
            return (position[0] - 1, position[1])
        elif action == 1:  # Down
            return (position[0] + 1, position[1])
        elif action == 2:  # Left
            return (position[0], position[1] - 1)
        elif action == 3:  # Right
            return (position[0], position[1] + 1)
        else:
            return None  # Handle unexpected actions by returning None

    def is_valid_position(self, position):
        # Check if the new position is within the grid and not an obstacle
        return (0 <= position[0] < self.grid_size[0] and
                0 <= position[1] < self.grid_size[1] and
                self.grid[position] != 1)
