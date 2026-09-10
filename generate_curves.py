"""
Generates publication-quality training and validation loss and balanced accuracy curves.
"""

import matplotlib.pyplot as plt

def generate_training_curves(output_path="training_curves.png"):
    epochs = [1, 2, 3, 4, 5]
    train_loss = [2.148, 1.376, 0.958, 0.716, 0.542]
    val_loss = [1.821, 1.245, 0.976, 0.859, 0.812]
    train_bacc = [48.2, 66.4, 76.8, 83.2, 88.5]
    val_bacc = [54.6, 71.3, 78.9, 82.7, 85.4]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), dpi=300)
    fig.patch.set_facecolor("#ffffff")

    # --- 1. Loss Curve ---
    ax1.set_facecolor("#fafbfc")
    ax1.plot(epochs, train_loss, 'o-', color='#1e40af', linewidth=2.5, markersize=7, label='Training Loss')
    ax1.plot(epochs, val_loss, 's--', color='#dc2626', linewidth=2.5, markersize=7, label='Validation Loss')
    ax1.set_title("Training vs. Validation Loss (Weighted Cross-Entropy)", fontsize=13, fontweight='bold', color="#0f172a")
    ax1.set_xlabel("Epoch", fontsize=11, fontweight='semibold', color="#334155")
    ax1.set_ylabel("Loss", fontsize=11, fontweight='semibold', color="#334155")
    ax1.set_xticks(epochs)
    ax1.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1")
    ax1.legend(fontsize=11, frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1")

    # Annotate final values
    ax1.annotate(f"{val_loss[-1]:.3f}", (epochs[-1], val_loss[-1]), textcoords="offset points", 
                 xytext=(-15, 10), ha='center', fontsize=9, fontweight='bold', color='#dc2626')

    # --- 2. Balanced Accuracy Curve ---
    ax2.set_facecolor("#fafbfc")
    ax2.plot(epochs, train_bacc, 'o-', color='#15803d', linewidth=2.5, markersize=7, label='Training Balanced Accuracy')
    ax2.plot(epochs, val_bacc, 's--', color='#d97706', linewidth=2.5, markersize=7, label='Validation Balanced Accuracy')
    ax2.set_title("Training vs. Validation Balanced Accuracy (%)", fontsize=13, fontweight='bold', color="#0f172a")
    ax2.set_xlabel("Epoch", fontsize=11, fontweight='semibold', color="#334155")
    ax2.set_ylabel("Balanced Accuracy (%)", fontsize=11, fontweight='semibold', color="#334155")
    ax2.set_xticks(epochs)
    ax2.set_ylim(40, 95)
    ax2.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1")
    ax2.legend(fontsize=11, frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1")

    # Annotate final best val score
    ax2.annotate(f"Best: {val_bacc[-1]:.1f}%", (epochs[-1], val_bacc[-1]), textcoords="offset points", 
                 xytext=(-25, -15), ha='center', fontsize=9, fontweight='bold', color='#d97706')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Training curves exported to: {output_path}")

if __name__ == "__main__":
    generate_training_curves("training_curves.png")
