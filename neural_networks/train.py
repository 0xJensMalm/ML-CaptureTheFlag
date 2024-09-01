import numpy as np
from collections import deque
import random
from .model import create_model
from .environment import CaptureTheFlagEnv
from .visualize_training import plot_training_progress

class ExperienceReplay:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

def train_model(model, env, team, num_epochs=1000, batch_size=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
    losses = []
    epsilon = epsilon_start
    experience_replay = ExperienceReplay(10000)

    for epoch in range(num_epochs):
        state = env.reset()
        done = False
        total_reward = 0
        epoch_loss = 0

        while not done:
            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                action = np.argmax(model.predict(state.reshape(1, 10, 20, 1)))

            next_state, reward, done = env.step(team, action)
            total_reward += reward

            experience_replay.add((state, action, reward, next_state, done))

            if len(experience_replay.buffer) >= batch_size:
                batch = experience_replay.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = np.array(states).reshape(batch_size, 10, 20, 1)
                next_states = np.array(next_states).reshape(batch_size, 10, 20, 1)

                current_q_values = model.predict(states)
                next_q_values = model.predict(next_states)

                target_q_values = current_q_values.copy()
                for i in range(batch_size):
                    if dones[i]:
                        target_q_values[i, actions[i]] = rewards[i]
                    else:
                        target_q_values[i, actions[i]] = rewards[i] + 0.99 * np.max(next_q_values[i])

                history = model.fit(states, target_q_values, epochs=1, verbose=0)
                epoch_loss += history.history['loss'][0]

            state = next_state

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        losses.append(epoch_loss)
        print(f"Epoch {epoch + 1}/{num_epochs} - Total Reward: {total_reward:.2f} - Loss: {epoch_loss:.4f} - Epsilon: {epsilon:.4f}")

    plot_training_progress(losses, title=f"Training Progress for {team}")
    return model

if __name__ == "__main__":
    input_shape = (10, 20, 1)
    env = CaptureTheFlagEnv(grid_size=(10, 20))
    model1 = create_model(input_shape, learning_rate=0.001)
    model2 = create_model(input_shape, learning_rate=0.001)
    
    model1 = train_model(model1, env, 'team1')
    model2 = train_model(model2, env, 'team2')