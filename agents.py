import pygame
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque

# Define action constants
ACTIONS = [
    (-1, 0),  # Left
    (1, 0),   # Right
    (0, -1),  # Up
    (0, 1),   # Down
]
ACTION_SPACE = [0, 1, 2, 3]

# Hyperparameters
GAMMA = 0.99
BATCH_SIZE = 64
LR = 0.001
MEMORY_SIZE = 10000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

class Agent:
    def __init__(self, x, y, color, grid_size, rows, cols, name):
        self.x = x  # Grid position x
        self.y = y  # Grid position y
        self.color = color
        self.grid_size = grid_size
        self.rows = rows
        self.cols = cols
        self.has_flag = False
        self.name = name  # Agent's name

        self.state_size = rows * cols  # Flattened grid
        self.action_size = len(ACTION_SPACE)
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.gamma = GAMMA
        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_END
        self.epsilon_decay = EPSILON_DECAY
        self.model = DQN(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)

        # Metrics for monitoring
        self.episode_rewards = []
        self.losses = []
        self.epsilon_values = []
        self.total_reward = 0

    def draw(self, surface):
        rect = pygame.Rect(
            self.x * self.grid_size,
            self.y * self.grid_size,
            self.grid_size,
            self.grid_size
        )
        pygame.draw.rect(surface, self.color, rect)

        if self.has_flag:
            # Draw a slightly opaque yellow overlay
            overlay = pygame.Surface((self.grid_size, self.grid_size), pygame.SRCALPHA)
            overlay.fill((255, 255, 0, 128))  # Yellow with 50% opacity
            surface.blit(overlay, (self.x * self.grid_size, self.y * self.grid_size))

    def move(self, action, obstacles):
        dx, dy = ACTIONS[action]
        new_x = self.x + dx
        new_y = self.y + dy

        # Check for boundaries
        if 0 <= new_x < self.cols and 0 <= new_y < self.rows:
            # Check for obstacles
            if not any(ob.x == new_x and ob.y == new_y for ob in obstacles):
                self.x = new_x
                self.y = new_y

    def reset(self, x, y):
        self.x = x
        self.y = y
        self.has_flag = False
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        self.epsilon_values.append(self.epsilon)

    def get_state(self, env):
        grid = np.zeros((self.rows, self.cols), dtype=int)

        # Mark obstacles
        for ob in env['obstacles']:
            grid[ob.y][ob.x] = 1

        # Mark flag
        if not env['flag'].carried_by:
            grid[env['flag'].y][env['flag'].x] = 2
        elif env['flag'].carried_by == self:
            grid[self.y][self.x] = 5  # Agent carrying the flag

        # Mark agents
        for agent in env['agents']:
            if agent == self:
                grid[self.y][self.x] = 3  # Self
            else:
                grid[agent.y][agent.x] = 4  # Opponent

        # Flatten the grid
        state = grid.flatten()
        state = torch.FloatTensor(state)
        return state

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(ACTION_SPACE)
        with torch.no_grad():
            q_values = self.model(state)
            return torch.argmax(q_values).item()

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.stack(next_states)
        dones = torch.FloatTensor(dones)

        # Compute Q(s_t, a)
        q_values = self.model(states).gather(1, actions)

        # Compute Q(s_{t+1}, a)
        next_q_values = self.model(next_states).max(1)[0].detach()
        expected_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = F.mse_loss(q_values.squeeze(), expected_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Log the loss
        self.losses.append(loss.item())

    def update(self, env):
        state = self.get_state(env)

        action = self.act(state)
        old_x, old_y = self.x, self.y
        self.move(action, env['obstacles'])
        reward = -0.01  # Small penalty for each move to encourage efficiency
        done = False

        # Update flag status
        if self.x == env['flag'].x and self.y == env['flag'].y and not env['flag'].carried_by:
            env['flag'].carried_by = self
            self.has_flag = True
            reward += 10  # Reward for picking up the flag

        # Check if agent has returned to scoring zone with the flag
        scoring_zone = env['scoring_zones'][self.name]
        if self.has_flag and (self.x, self.y) in scoring_zone:
            reward += 100  # Large reward for scoring
            env['scores'][self.name] += 1
            done = True

        # Prepare next state
        next_state = self.get_state(env)

        # Remember experience
        self.remember(state, action, reward, next_state, done)

        # Learn from experiences
        self.replay()

        self.total_reward += reward

        return reward, done

class AgentA(Agent):
    def __init__(self, x, y, color, grid_size, rows, cols):
        super().__init__(x, y, color, grid_size, rows, cols, name='Agent A')

class AgentB(Agent):
    def __init__(self, x, y, color, grid_size, rows, cols):
        super().__init__(x, y, color, grid_size, rows, cols, name='Agent B')
