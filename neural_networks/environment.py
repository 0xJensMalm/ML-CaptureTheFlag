import numpy as np

class CaptureTheFlagEnv:
    def __init__(self, grid_size=(10, 20), num_obstacles=6):
        self.grid_size = grid_size
        self.num_obstacles = num_obstacles
        self.team1_points = 0
        self.team2_points = 0
        self.flag_captured = False
        self.reset()

    def reset(self):
        self.grid = np.zeros(self.grid_size)
        self.place_flag()
        self.place_static_bases_and_obstacles()
        self.place_players()
        self.flag_captured = False
        self.flag_carrier = None
        return self.grid

    def place_flag(self):
        flag_position = (self.grid_size[0] // 2, self.grid_size[1] // 2)
        self.grid[flag_position] = 3
        self.flag_position = flag_position

    def place_static_bases_and_obstacles(self):
        base1_positions = [(3, 0), (4, 0), (5, 0), (6, 0), (3, 1), (6, 1), (3, 2), (6, 2)]
        base2_positions = [(3, 19), (4, 19), (5, 19), (6, 19), (3, 18), (6, 18), (3, 17), (6, 17)]

        for pos in base1_positions + base2_positions:
            self.grid[pos] = 1

        obstacle_positions = [
            (3, 4), (3, 5), (3, 6), (6, 4), (6, 5), (6, 6),
            (3, 13), (3, 14), (3, 15), (6, 13), (6, 14), (6, 15)
        ]
        for pos in obstacle_positions:
            self.grid[pos] = 1

    def place_players(self):
        self.player_positions = {
            'team1': (4, 1),
            'team2': (5, 18)
        }
        self.grid[self.player_positions['team1']] = 2
        self.grid[self.player_positions['team2']] = 4

    def step(self, team, action):
        current_position = self.player_positions[team]
        new_position = self.get_new_position(current_position, action)

        if new_position is None or not self.is_valid_position(new_position):
            return self.grid, -1, False

        reward = self.calculate_reward(team, current_position, new_position)
        
        self.update_grid(team, current_position, new_position)
        
        done = self.check_game_over(team, new_position)

        return self.grid, reward, done

    def calculate_reward(self, team, current_position, new_position):
        reward = -0.1  # Small negative reward for each step to encourage efficiency

        if self.grid[new_position] == 3:  # Capturing the flag
            reward += 1
            self.flag_captured = True
            self.flag_carrier = team

        if self.flag_carrier == team:
            # Reward for moving closer to base when carrying the flag
            base_position = (4, 1) if team == 'team1' else (5, 18)
            old_distance = self.manhattan_distance(current_position, base_position)
            new_distance = self.manhattan_distance(new_position, base_position)
            reward += 0.1 * (old_distance - new_distance)
        else:
            # Reward for moving closer to the flag when not carrying it
            old_distance = self.manhattan_distance(current_position, self.flag_position)
            new_distance = self.manhattan_distance(new_position, self.flag_position)
            reward += 0.1 * (old_distance - new_distance)

        return reward

    def update_grid(self, team, current_position, new_position):
        self.grid[current_position] = 0
        self.grid[new_position] = 2 if team == 'team1' else 4
        self.player_positions[team] = new_position

        if self.flag_captured and self.flag_carrier == team:
            self.flag_position = new_position

    def check_game_over(self, team, position):
        if self.flag_captured and self.flag_carrier == team:
            if (team == 'team1' and position == (4, 1)) or (team == 'team2' and position == (5, 18)):
                if team == 'team1':
                    self.team1_points += 1
                else:
                    self.team2_points += 1
                self.flag_captured = False
                self.flag_carrier = None
                return True
        return False

    def get_new_position(self, position, action):
        if action == 0:  # Up
            return (position[0] - 1, position[1])
        elif action == 1:  # Down
            return (position[0] + 1, position[1])
        elif action == 2:  # Left
            return (position[0], position[1] - 1)
        elif action == 3:  # Right
            return (position[0], position[1] + 1)
        else:
            return None

    def is_valid_position(self, position):
        return (0 <= position[0] < self.grid_size[0] and
                0 <= position[1] < self.grid_size[1] and
                self.grid[position] != 1)

    def manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])