"""
Generates publication-quality training and validation loss and balanced accuracy curves.
Adheres to top-tier machine learning conference visualization standards (NeurIPS/ICLR style).
Can load dynamic metrics from training_history.json or fallback to recorded benchmark runs.
"""

import os
import json
import matplotlib.pyplot as plt


def generate_training_curves(output_path="training_curves.png", history_file="training_history.json", history_dict=None):
    if history_dict is not None:
        history = history_dict
    elif os.path.exists(history_file):
        with open(history_file, "r") as f:
            history = json.load(f)
        print(f"[INFO] Loaded training history from '{history_file}'.")
    else:
        raise FileNotFoundError(
            f"Training history file '{history_file}' not found. "
            "Execute training first or provide a history_dict."
        )

    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    train_bacc = [b * 100 if b <= 1.0 else b for b in history["train_bacc"]]
    val_bacc = [b * 100 if b <= 1.0 else b for b in history["val_bacc"]]
    epochs = list(range(1, len(train_loss) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4.8), dpi=300)
    fig.patch.set_facecolor("#ffffff")

    # =========================================================================
    # SUBPLOT 1: LOSS CURVE (Weighted Cross-Entropy)
    # =========================================================================
    ax1.set_facecolor("#fafbfc")
    ax1.plot(epochs, train_loss, 'o-', color='#1d4ed8', linewidth=2.4, markersize=7, 
             label='Training Loss (Weighted CE)', zorder=4)
    ax1.plot(epochs, val_loss, 's--', color='#dc2626', linewidth=2.4, markersize=7, 
             label='Validation Loss (Weighted CE)', zorder=4)

    # Shaded area showing convergence gap
    ax1.fill_between(epochs, train_loss, val_loss, color='#93c5fd', alpha=0.15, zorder=2)

    ax1.set_title("Training vs. Validation Loss (Weighted Cross-Entropy)", 
                  fontsize=12.5, fontweight='bold', color="#0f172a", pad=12)
    ax1.set_xlabel("Epoch", fontsize=10.5, fontweight='bold', color="#334155", labelpad=8)
    ax1.set_ylabel("Loss", fontsize=10.5, fontweight='bold', color="#334155", labelpad=8)
    ax1.set_xticks(epochs)
    ax1.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1", zorder=1)
    ax1.legend(fontsize=9.5, frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", loc="upper right")

    # Final annotation on loss
    ax1.annotate(f"Final Val: {val_loss[-1]:.3f}", (epochs[-1], val_loss[-1]), textcoords="offset points", 
                 xytext=(-20, 12), ha='center', fontsize=9, fontweight='bold', color='#dc2626',
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="#fef2f2", edgecolor="#fca5a5", lw=0.8))

    # =========================================================================
    # SUBPLOT 2: BALANCED ACCURACY CURVE
    # =========================================================================
    ax2.set_facecolor("#fafbfc")
    ax2.plot(epochs, train_bacc, 'o-', color='#15803d', linewidth=2.4, markersize=7, 
             label='Train Balanced Accuracy (%)', zorder=4)
    ax2.plot(epochs, val_bacc, 's--', color='#d97706', linewidth=2.4, markersize=7, 
             label='Validation Balanced Accuracy (%)', zorder=4)

    # Shaded fill
    ax2.fill_between(epochs, train_bacc, val_bacc, color='#86efac', alpha=0.15, zorder=2)

    ax2.set_title("Training vs. Validation Balanced Accuracy (%)", 
                  fontsize=12.5, fontweight='bold', color="#0f172a", pad=12)
    ax2.set_xlabel("Epoch", fontsize=10.5, fontweight='bold', color="#334155", labelpad=8)
    ax2.set_ylabel("Balanced Accuracy (%)", fontsize=10.5, fontweight='bold', color="#334155", labelpad=8)
    ax2.set_xticks(epochs)
    ax2.set_ylim(40, 95)
    ax2.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1", zorder=1)
    ax2.legend(fontsize=9.5, frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", loc="lower right")

    # Final annotation on best accuracy
    ax2.annotate(f"Best: {val_bacc[-1]:.1f}%", (epochs[-1], val_bacc[-1]), textcoords="offset points", 
                 xytext=(-25, -20), ha='center', fontsize=9, fontweight='bold', color='#b45309',
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="#fefce8", edgecolor="#fef08a", lw=0.8))

    # Subtle spine cleanup
    for ax in [ax1, ax2]:
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color("#94a3b8")
            ax.spines[spine].set_linewidth(0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Professional training curves exported to: {output_path}")


if __name__ == "__main__":
    generate_training_curves("training_curves.png")
