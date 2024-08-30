from flask import Flask, render_template, send_file
from neural_networks.environment import CaptureTheFlagEnv
from neural_networks.visualize import visualize_game_state
from neural_networks.visualize_training import plot_training_progress
from neural_networks.model1 import create_model1
from neural_networks.model2 import create_model2
import io
import time
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from threading import Thread
import logging

app = Flask(__name__)

env = CaptureTheFlagEnv(grid_size=(10, 20)) #Grid size
model1_losses = []
model2_losses = []

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def train_model_continuously():
    global model1_losses, model2_losses
    input_shape = (10, 20, 1)  # Adjust input shape to match the grid size of 10x20
    model1 = create_model1(input_shape)
    model2 = create_model2(input_shape)
    
    epoch_count = 0
    max_steps_per_episode = 50  # Limit the number of steps to prevent infinite loops
    epsilon = 0.1  # Exploration factor

    while True:
        state = env.reset()
        state = state.reshape(1, 10, 20, 1)  # Reshape to match the grid size
        done = False
        epoch_loss_1 = 0
        epoch_loss_2 = 0
        step_count = 0

        logging.info(f"Starting new episode (Epoch {epoch_count + 1})")

        while not done and step_count < max_steps_per_episode:
            step_count += 1

            # Model 1 (Team 1)
            if np.random.rand() < epsilon:
                action1 = np.random.choice([0, 1, 2, 3])  # Random action for exploration
                logging.info(f"Step {step_count}: Team 1 takes random action: {action1}")
            else:
                action1 = np.argmax(model1.predict(state))
                logging.info(f"Step {step_count}: Team 1 Action: {action1}")

            state, reward1, done = env.step('team1', action1)
            state = state.reshape(1, 10, 20, 1)  # Reshape after step
            epoch_loss_1 += np.abs(np.random.randn())  # Placeholder for actual loss computation

            target1 = model1.predict(state)
            target1[0, action1] = reward1 + 0.95 * np.max(target1)  # Update target with discounted reward
            model1.fit(state, target1, epochs=1, verbose=0)

            logging.info(f"Team 1 Position: {env.player_positions['team1']}, Reward: {reward1}, Done: {done}")

            if done:
                logging.info("Episode ended by Team 1 capturing the flag.")
                break

            # Model 2 (Team 2)
            if np.random.rand() < epsilon:
                action2 = np.random.choice([0, 1, 2, 3])  # Random action for exploration
                logging.info(f"Step {step_count}: Team 2 takes random action: {action2}")
            else:
                action2 = np.argmax(model2.predict(state))
                logging.info(f"Step {step_count}: Team 2 Action: {action2}")

            state, reward2, done = env.step('team2', action2)
            state = state.reshape(1, 10, 20, 1)  # Reshape after step
            epoch_loss_2 += np.abs(np.random.randn())  # Placeholder for actual loss computation

            target2 = model2.predict(state)
            target2[0, action2] = reward2 + 0.95 * np.max(target2)  # Update target with discounted reward
            model2.fit(state, target2, epochs=1, verbose=0)

            logging.info(f"Team 2 Position: {env.player_positions['team2']}, Reward: {reward2}, Done: {done}")

            if done:
                logging.info("Episode ended by Team 2 capturing the flag.")
                break

        model1_losses.append(epoch_loss_1)
        model2_losses.append(epoch_loss_2)

        # Limit the number of stored losses to avoid memory issues
        model1_losses = model1_losses[-100:]
        model2_losses = model2_losses[-100:]

        logging.info(f"Epoch {epoch_count + 1} completed. Losses - Team 1: {epoch_loss_1}, Team 2: {epoch_loss_2}")
        epoch_count += 1

        if step_count >= max_steps_per_episode:
            logging.warning(f"Max steps reached in Epoch {epoch_count}. Episode terminated to prevent infinite loop.")
        
        time.sleep(1)  # Simulate time delay for training


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/game-view')
def game_view():
    try:
        fig = visualize_game_state(env.grid, env.team1_points, env.team2_points)
        output = io.BytesIO()
        FigureCanvas(fig).print_png(output)
        return send_file(io.BytesIO(output.getvalue()), mimetype='image/png')
    except Exception as e:
        logging.error(f"Error generating game view: {e}")
        return "An error occurred while generating the game view.", 500


@app.route('/nn1-training')
def nn1_training():
    fig = plot_training_progress(model1_losses, title="Training Progress for Model 1")
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return send_file(io.BytesIO(output.getvalue()), mimetype='image/png')

@app.route('/nn2-training')
def nn2_training():
    fig = plot_training_progress(model2_losses, title="Training Progress for Model 2")
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return send_file(io.BytesIO(output.getvalue()), mimetype='image/png')

if __name__ == "__main__":
    training_thread = Thread(target=train_model_continuously)
    training_thread.start()

    app.run(debug=True)
