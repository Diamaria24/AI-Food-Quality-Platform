
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Trained MobileMold model
MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "trained"
    / "rgb_mold"
    / "mobilemold_mobilenetv2_best.keras"
)

_model = None


def get_model():
    """Load the trained model once and reuse it."""
    global _model

    if _model is None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"MobileMold model not found: {MODEL_PATH}"
            )

        _model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

    return _model


def predict_rgb_mold(image_path):
    """
    Predict mold from a microscope-assisted RGB image.

    Class 0: No Mold
    Class 1: Mold
    """

    model = get_model()

    # Match the Kaggle training preprocessing
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize((224, 224))
        image_array = np.asarray(
            image, dtype=np.float32
        )

    # MobileNetV2 normalization: [0, 255] -> [-1, 1]
    image_array = (
        tf.keras.applications.mobilenet_v2
        .preprocess_input(image_array)
    )

    batch = np.expand_dims(image_array, axis=0)

    # Sigmoid output: probability-like model score
    mold_score = float(
        model.predict(batch, verbose=0)[0][0]
    )

    predicted_label = (
        "Mold" if mold_score >= 0.5 else "No Mold"
    )

    return {
        "prediction": predicted_label,
        "mold_score": round(mold_score, 4),
        "threshold": 0.5,
        "model": "MobileMold_MobileNetV2",
        "supported_foods": [
            "tomato", "orange", "onion", "carrot"
        ],
        "image_type": "Microscope-assisted RGB",
        "warning": (
            "Experimental prediction only. "
            "The model was trained on microscope-assisted "
            "smartphone images and is not validated for "
            "ordinary phone photographs or food-safety decisions."
        )
    }
