import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-GUI rendering

import matplotlib.pyplot as plt

def plot_training_progress(losses, title="Training Progress"):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(losses, label='Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    return fig  # Return the figure instead of showing it
