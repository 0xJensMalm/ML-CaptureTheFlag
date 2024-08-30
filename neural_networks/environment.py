import numpy as np

class CaptureTheFlagEnv:
    def __init__(self, grid_size=(10, 20), num_obstacles=6):
        self.grid_size = grid_size
        self.num_obstacles = num_obstacles
        self.team1_points = 0
        self.team2_points = 0
        self.flag_captured = False  # Initialize flag_captured attribute
        self.reset()

    def reset(self):
        self.grid = np.zeros(self.grid_size)  # Empty grid
        self.place_flag()
        self.place_static_bases_and_obstacles()  # Static base structures
        self.place_players()
        self.flag_captured = False  # Reset the flag_captured status when environment resets
        return self.grid

    def place_flag(self):
        # Place the flag at the center of the grid
        flag_position = (self.grid_size[0] // 2, self.grid_size[1] // 2)
        self.grid[flag_position] = 3  # 3 represents the flag

    def place_static_bases_and_obstacles(self):
        # Static obstacles and base structure
        base1_positions = [(3, 0), (4, 0), (5, 0), (6, 0), (3, 1), (6, 1), (3, 2), (6, 2)]
        base2_positions = [(3, 19), (4, 19), (5, 19), (6, 19), (3, 18), (6, 18), (3, 17), (6, 17)]

        # Place base structures
        for pos in base1_positions:
            self.grid[pos] = 1  # 1 represents static obstacle as part of base 1

        for pos in base2_positions:
            self.grid[pos] = 1  # 1 represents static obstacle as part of base 2

        # Additional obstacles
        obstacle_positions = [
            (3, 4), (3, 5), (3, 6), (6, 4), (6, 5), (6, 6),
            (3, 13), (3, 14), (3, 15), (6, 13), (6, 14), (6, 15)
        ]
        for pos in obstacle_positions:
            self.grid[pos] = 1  # 1 represents an obstacle

    def place_players(self):
        # Place the two teams at the center of their bases
        self.player_positions = {
            'team1': (4, 1),  # Team 1 starts inside base 1
            'team2': (5, 18)  # Team 2 starts inside base 2
        }
        self.grid[self.player_positions['team1']] = 2  # 2 represents team1
        self.grid[self.player_positions['team2']] = 4  # 4 represents team2

    def step(self, team, action):
        current_position = self.player_positions[team]
        new_position = self.get_new_position(current_position, action)

        if new_position is None or not self.is_valid_position(new_position):
            # Invalid move (into an obstacle, outside the grid, or invalid action)
            return self.grid, -1, False

        # Check if the player captured the flag
        if self.grid[new_position] == 3:
            reward = 1
            done = False  # Not done until they return to base
            self.grid[new_position] = 2 if team == 'team1' else 4
            self.player_positions[team] = new_position
            self.flag_captured = True
            return self.grid, reward, done

        # Move the player
        self.grid[current_position] = 0  # Clear the old position
        self.grid[new_position] = 2 if team == 'team1' else 4  # Update the new position
        self.player_positions[team] = new_position

        # Check if the player returned the flag to base
        if self.flag_captured and (
            (team == 'team1' and new_position == (4, 1)) or 
            (team == 'team2' and new_position == (5, 18))
        ):
            reward = 5  # Higher reward for returning the flag to the base
            done = True
            self.flag_captured = False
            if team == 'team1':
                self.team1_points += 1
            else:
                self.team2_points += 1
            return self.grid, reward, done

        return self.grid, -1, False

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
