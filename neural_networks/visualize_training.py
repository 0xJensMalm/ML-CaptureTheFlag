import matplotlib.pyplot as plt
import numpy as np

def plot_training_progress(losses, title="Training Progress"):
    fig, ax = plt.subplots(figsize=(6, 6), facecolor='#2a2a2a')
    
    if losses:
        epochs = np.arange(1, len(losses) + 1)
        ax.plot(epochs, losses, label='Loss', color='#4fc3f7')
        ax.set_xlim(1, max(epochs))
        ax.set_ylim(0, max(losses) * 1.1)
    else:
        ax.text(0.5, 0.5, "No data available", ha='center', va='center', color='#f1f1f1')

    ax.set_xlabel('Epoch', color='#f1f1f1')
    ax.set_ylabel('Loss', color='#f1f1f1')
    ax.set_title(title, color='#f1f1f1')
    ax.legend()
    ax.grid(True, color='#444444')
    ax.set_facecolor('#1e1e1e')
    ax.tick_params(colors='#f1f1f1')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444444')
    plt.tight_layout()
    return fig