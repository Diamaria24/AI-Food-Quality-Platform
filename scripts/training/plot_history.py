import json
import matplotlib.pyplot as plt
from pathlib import Path


HISTORY_FILE = Path(
    "outputs/history/mobilenet_history.json"
)

with open(HISTORY_FILE) as f:

    history = json.load(f)


# =========================
# Accuracy Plot
# =========================

plt.figure(figsize=(8,6))

plt.plot(

    history["accuracy"],
    label="Train Accuracy",
    linewidth=2
)

plt.plot(

    history["val_accuracy"],
    label="Validation Accuracy",
    linewidth=2
)

plt.title(
    "Model Accuracy"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.legend()

plt.grid(True)

plt.savefig(

    "outputs/plots/mobilenet_accuracy_curve.png",

    dpi=300,

    bbox_inches="tight"
)

plt.show()


# =========================
# Loss Plot
# =========================

plt.figure(figsize=(8,6))

plt.plot(

    history["loss"],
    label="Train Loss",
    linewidth=2
)

plt.plot(

    history["val_loss"],
    label="Validation Loss",
    linewidth=2
)

plt.title(
    "Model Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.legend()

plt.grid(True)

plt.savefig(

    "outputs/plots/mobilenet_loss_curve.png",

    dpi=300,

    bbox_inches="tight"
)

plt.show()

print()
print("Plots Saved.")