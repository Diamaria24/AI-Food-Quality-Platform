import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input
)

# =========================================================
# Paths
# =========================================================

MODEL_PATH = "models/trained/mobilenet.keras"

IMAGE_FOLDER = Path(
    "datasets/multiclass/test/Tomato_Rotten"
)

SAVE_PATH = (
    "outputs/plots/gradcam_tomato_rotten.jpg"
)

# Automatically select first image
IMAGE_PATH = list(
    IMAGE_FOLDER.glob("*")
)[0]

print()
print("Image :", IMAGE_PATH)

# =========================================================
# Load Model
# =========================================================

model = load_model(
    MODEL_PATH
)

# Print layers once if needed
# for layer in model.layers:
#     print(layer.name)

# =========================================================
# Load Image
# =========================================================

img = image.load_img(

    str(IMAGE_PATH),

    target_size=(224, 224)
)

img_array = image.img_to_array(
    img
)

input_img = np.expand_dims(

    img_array,

    axis=0
)

input_img = preprocess_input(
    input_img
)

# =========================================================
# Find Last Conv Layer Automatically
# =========================================================

last_conv_layer = None

for layer in reversed(model.layers):

    if isinstance(
        layer,
        tf.keras.layers.Conv2D
    ):

        last_conv_layer = layer
        break

print()
print(
    "Last Conv Layer :",
    last_conv_layer.name
)

# =========================================================
# GradCAM Model
# =========================================================

grad_model = tf.keras.models.Model(

    inputs=model.inputs,

    outputs=[

        last_conv_layer.output,

        model.output
    ]
)

# =========================================================
# Compute Gradients
# =========================================================

with tf.GradientTape() as tape:

    conv_outputs, predictions = grad_model(
        input_img
    )

    pred_index = tf.argmax(
        predictions[0]
    )

    class_channel = predictions[
        :,
        pred_index
    ]

grads = tape.gradient(

    class_channel,

    conv_outputs
)

pooled_grads = tf.reduce_mean(

    grads,

    axis=(0, 1, 2)
)

conv_outputs = conv_outputs[0]

heatmap = tf.reduce_sum(

    pooled_grads * conv_outputs,

    axis=-1
)

heatmap = np.maximum(

    heatmap,

    0
)

heatmap /= (

    np.max(heatmap) + 1e-8
)

# =========================================================
# Overlay Heatmap
# =========================================================

img_cv = cv2.imread(
    str(IMAGE_PATH)
)

heatmap = cv2.resize(

    heatmap,

    (

        img_cv.shape[1],

        img_cv.shape[0]
    )
)

heatmap = np.uint8(
    255 * heatmap
)

heatmap = cv2.applyColorMap(

    heatmap,

    cv2.COLORMAP_JET
)

superimposed_img = cv2.addWeighted(

    img_cv,

    0.6,

    heatmap,

    0.4,

    0
)

# =========================================================
# Save
# =========================================================

cv2.imwrite(

    SAVE_PATH,

    superimposed_img
)

# =========================================================
# Display
# =========================================================

plt.figure(figsize=(8,8))

plt.imshow(

    cv2.cvtColor(

        superimposed_img,

        cv2.COLOR_BGR2RGB
    )
)

plt.axis("off")

plt.title(
    "GradCAM Visualization"
)

plt.show()

print()
print(
    "Saved :",
    SAVE_PATH
)