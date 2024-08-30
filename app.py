# app.py
from flask import Flask, render_template, jsonify
from neural_networks.environment import CaptureTheFlagEnv
from neural_networks.model import create_model
import numpy as np
from threading import Thread, Lock
import logging
import os
import tensorflow as tf
import time

app = Flask(__name__)

# Configuration
GRID_SIZE = (10, 20)
MAX_STEPS_PER_EPISODE = 50
EPSILON = 0.1
GAMMA = 0.95
LEARNING_RATE = 0.001
UPDATE_INTERVAL = 0.1  # 100ms between updates

env = CaptureTheFlagEnv(grid_size=GRID_SIZE)
model_losses = {
    'team1': [],
    'team2': []
}
models = None

# Action heatmap data
action_heatmap = {
    'team1': np.zeros((GRID_SIZE[0], GRID_SIZE[1], 4)),
    'team2': np.zeros((GRID_SIZE[0], GRID_SIZE[1], 4))
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

env_lock = Lock()
model_lock = Lock()
heatmap_lock = Lock()

def create_models():
    input_shape = (*GRID_SIZE, 1)
    return {
        'team1': create_model(input_shape, LEARNING_RATE),
        'team2': create_model(input_shape, LEARNING_RATE)
    }

def train_model_continuously():
    global models, env, model_losses, action_heatmap
    models = create_models()
    epoch_count = 0

    while True:
        with env_lock:
            state = env.reset()
        state = state.reshape(1, *GRID_SIZE, 1)
        done = False
        epoch_losses = {'team1': 0, 'team2': 0}
        step_count = 0

        while not done and step_count < MAX_STEPS_PER_EPISODE:
            step_count += 1

            for team in ['team1', 'team2']:
                with model_lock:
                    action = np.argmax(models[team].predict(state)) if np.random.rand() > EPSILON else np.random.choice([0, 1, 2, 3])
                
                # Update action heatmap
                with heatmap_lock:
                    player_pos = env.player_positions[team]
                    action_heatmap[team][player_pos[0], player_pos[1], action] += 1

                with env_lock:
                    next_state, reward, done = env.step(team, action)
                next_state = next_state.reshape(1, *GRID_SIZE, 1)

                with model_lock:
                    target = models[team].predict(state)
                    next_q_values = models[team].predict(next_state)
                    target[0, action] = reward + GAMMA * np.max(next_q_values)
                    history = models[team].fit(state, target, epochs=1, verbose=0)
                    epoch_losses[team] += history.history['loss'][0]

                state = next_state

                if done:
                    break

            if done:
                break

        with model_lock:
            for team in ['team1', 'team2']:
                model_losses[team].append(epoch_losses[team])
                model_losses[team] = model_losses[team][-100:]  # Keep only last 100 losses

        epoch_count += 1
        time.sleep(UPDATE_INTERVAL)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/game-state')
def game_state():
    with env_lock:
        return jsonify({
            'grid': env.grid.tolist(),
            'team1_points': env.team1_points,
            'team2_points': env.team2_points
        })

@app.route('/training-data')
def training_data():
    with model_lock:
        return jsonify(model_losses)

@app.route('/action-heatmap')
def get_action_heatmap():
    with heatmap_lock:
        return jsonify({
            'team1': action_heatmap['team1'].tolist(),
            'team2': action_heatmap['team2'].tolist()
        })

def run_flask():
    app.run(debug=False, use_reloader=False, port=5001)

if __name__ == "__main__":
    flask_thread = Thread(target=run_flask)
    flask_thread.start()
    time.sleep(2)
    train_model_continuously()