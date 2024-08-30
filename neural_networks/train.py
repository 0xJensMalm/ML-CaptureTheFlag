from .model1 import create_model1
from .model2 import create_model2
from .environment import CaptureTheFlagEnv
from .visualize_training import plot_training_progress
import numpy as np

def train_model(model, env, team, num_epochs=1000):
    losses = []

    for epoch in range(num_epochs):
        state = env.reset()
        done = False
        total_reward = 0
        epoch_loss = 0

        while not done:
            # Predict the action based on the current state
            action = np.argmax(model.predict(state.reshape(1, 10, 20, 1)))  # Adjusted to the correct input shape
            state, reward, done = env.step(team, action)
            total_reward += reward

            # Simulate loss calculation (replace with real implementation)
            loss = np.abs(np.random.randn())  # Example random loss
            epoch_loss += loss
            model.fit(state.reshape(1, 10, 20, 1), np.array([action]), epochs=1, verbose=0)  # Adjusted input shape

        losses.append(epoch_loss)
        print(f"Epoch {epoch + 1}/{num_epochs} - Total Reward: {total_reward} - Loss: {epoch_loss}")

    # After training, plot the training progress
    plot_training_progress(losses, title=f"Training Progress for {team}")

if __name__ == "__main__":
    input_shape = (10, 20, 1)  # Adjusted input shape to match the grid size
    env = CaptureTheFlagEnv(grid_size=(10, 20))  # Use the correct grid size
    model1 = create_model1(input_shape)
    model2 = create_model2(input_shape)
    
    # Train the models for both teams
    train_model(model1, env, 'team1')
    train_model(model2, env, 'team2')
