import random
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input
)

MODEL_PATH = (
    "models/trained/mobilenet.keras"
)

TEST_FOLDER = Path(
    "datasets/multiclass/test"
)

model = load_model(
    MODEL_PATH
)

classes = [

    "Apple_Fresh",
    "Apple_Rotten",
    "Banana_Fresh",
    "Banana_Rotten",
    "Strawberry_Fresh",
    "Strawberry_Rotten",
    "Tomato_Fresh",
    "Tomato_Rotten"

]

all_images = []

for folder in TEST_FOLDER.iterdir():

    for img in folder.glob("*"):

        all_images.append(img)

random.shuffle(
    all_images
)

samples = all_images[:12]

plt.figure(figsize=(15,10))

for i, img_path in enumerate(samples):

    img = image.load_img(

        img_path,

        target_size=(224,224)
    )

    img_array = image.img_to_array(
        img
    )

    x = np.expand_dims(

        img_array,

        axis=0
    )

    x = preprocess_input(
        x
    )

    pred = model.predict(

        x,

        verbose=0
    )

    pred_class = classes[
        np.argmax(pred)
    ]

    confidence = np.max(
        pred
    )

    actual = img_path.parent.name

    plt.subplot(

        3,
        4,
        i+1
    )

    plt.imshow(img)

    color = (

        "green"

        if pred_class == actual

        else "red"
    )

    plt.title(

        f"A:{actual}\nP:{pred_class}\n{confidence:.2%}",

        fontsize=8,

        color=color
    )

    plt.axis("off")

plt.tight_layout()

plt.savefig(

    "outputs/plots/sample_predictions.png",

    dpi=300
)

plt.show()

print()
print("Sample Predictions Saved.")