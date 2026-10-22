
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

# --------------------------------------------------
# Model configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "trained"
    / "mobilenet.keras"
)

CLASS_NAMES = [
    "Apple_Fresh",
    "Apple_Rotten",
    "Banana_Fresh",
    "Banana_Rotten",
    "Strawberry_Fresh",
    "Strawberry_Rotten",
    "Tomato_Fresh",
    "Tomato_Rotten",
]

IMAGE_SIZE = (224, 224)

_model = None


# --------------------------------------------------
# Load model once
# --------------------------------------------------

def get_model():
    global _model

    if _model is None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Freshness model not found: {MODEL_PATH}"
            )

        _model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

        if _model.output_shape[-1] != len(CLASS_NAMES):
            raise ValueError(
                "Model output classes do not match "
                "the configured class labels."
            )

    return _model


# --------------------------------------------------
# Predict food freshness
# --------------------------------------------------

def predict_rgb_freshness(image_path):
    """
    Classify an RGB image into one of eight classes.

    Supported foods:
    Apple, Banana, Strawberry, Tomato.

    Output:
    Food type and Fresh/Rotten classification.
    """

    model = get_model()

    # Match original MobileNet training preprocessing.
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize(IMAGE_SIZE)

        image_array = np.asarray(
            image,
            dtype=np.float32
        )

    # Normalize from [0, 255] to [-1, 1].
    image_array = (
        tf.keras.applications.mobilenet_v2
        .preprocess_input(image_array)
    )

    batch = np.expand_dims(
        image_array,
        axis=0
    )

    # Eight-class softmax prediction.
    scores = model.predict(
        batch,
        verbose=0
    )[0]

    predicted_index = int(np.argmax(scores))
    predicted_class = CLASS_NAMES[predicted_index]

    food, freshness = predicted_class.split("_")

    model_score = float(scores[predicted_index])

    return {
        "prediction": predicted_class,
        "food": food,
        "freshness": freshness,
        "model_score": round(model_score, 4),
        "model": "RGB_Freshness_MobileNet",
        "supported_foods": [
            "Apple",
            "Banana",
            "Strawberry",
            "Tomato"
        ],
        "class_scores": {
            name: round(float(score), 4)
            for name, score in zip(CLASS_NAMES, scores)
        },
        "warning": (
            "Experimental RGB freshness classification only. "
            "The model supports Apple, Banana, Strawberry "
            "and Tomato images. Softmax scores are not "
            "calibrated confidence estimates. Predictions "
            "must not be used as food-safety certification."
        )
    }
