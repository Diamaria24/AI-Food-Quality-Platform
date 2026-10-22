
from pathlib import Path
from functools import lru_cache

import joblib
import numpy as np

from spectral import open_image
from scipy.signal import savgol_filter


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_ROOT = (
    PROJECT_ROOT
    / "models"
    / "trained"
    / "hsi"
)

REFERENCE_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "raw"
    / "hyperspectral"
)

REFERENCE_HEADERS = {
    "eggplant": (
        REFERENCE_ROOT
        / "eggplant_hsi"
        / "Eggplant_Crop"
        / "Eggplant_Reflectance_Data.hdr"
    ),
    "cabbage": (
        REFERENCE_ROOT
        / "cabbage_hsi"
        / "Cabbage_Crop"
        / "Cabbage_Reflectance_Data.hdr"
    ),
}

CLASS_NAMES = {
    1: "Low",
    2: "Medium",
    3: "High",
}


@lru_cache(maxsize=2)
def load_model(food):
    if food not in REFERENCE_HEADERS:
        raise ValueError("Unsupported food.")

    model_path = MODEL_ROOT / food / "rf_pipeline.joblib"

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    return joblib.load(model_path)


@lru_cache(maxsize=2)
def reference_wavelengths(food):
    image = open_image(str(REFERENCE_HEADERS[food]))

    return np.asarray(
        image.metadata["wavelength"],
        dtype=np.float64,
    )


def validate_cube(food, image):
    if image.nbands != 277:
        raise ValueError(
            "Expected 277 spectral bands."
        )

    if image.metadata.get("interleave", "").lower() != "bil":
        raise ValueError(
            "Only BIL-interleaved ENVI files are supported."
        )

    wavelengths = image.metadata.get("wavelength")

    if wavelengths is None:
        raise ValueError(
            "Wavelength metadata is required."
        )

    actual = np.asarray(wavelengths, dtype=np.float64)
    expected = reference_wavelengths(food)

    if not np.allclose(actual, expected, atol=0.001, rtol=0):
        raise ValueError(
            "Spectral wavelengths do not match training data."
        )


def predict_hsi_nitrogen(
    food,
    reflectance_header,
    ground_truth_header,
    max_pixels=10000,
    chunk_rows=8,
):
    food = food.lower()

    artifact = load_model(food)

    image = open_image(str(reflectance_header))
    gt_image = open_image(str(ground_truth_header))

    validate_cube(food, image)

    if gt_image.nbands != 1:
        raise ValueError(
            "Ground-truth image must have exactly one band."
        )

    if image.shape[:2] != gt_image.shape[:2]:
        raise ValueError(
            "Reflectance and ground-truth dimensions differ."
        )

    # ENVI memory maps avoid loading entire cubes into RAM.
    cube = image.open_memmap(interleave="bip")
    gt = gt_image.open_memmap(interleave="bip")

    height, width, _ = image.shape

    # Pass 1: determine the global normalization scale.
    absolute_max = -np.inf
    finite_found = False

    for start in range(0, height, chunk_rows):
        end = min(start + chunk_rows, height)

        block = np.asarray(
            cube[start:end],
            dtype=np.float32,
        )

        finite = block[np.isfinite(block)]

        if finite.size:
            finite_found = True
            absolute_max = max(
                absolute_max,
                float(np.max(finite)),
            )

    if not finite_found:
        raise ValueError(
            "No finite reflectance values found."
        )

    if absolute_max > 1000:
        scale = 4095.0
    elif absolute_max > 10:
        scale = absolute_max
    else:
        scale = 1.0

    # Pass 2: collect bounded, ground-truth-labeled spectra.
    selected = []
    selected_count = 0

    for start in range(0, height, chunk_rows):
        end = min(start + chunk_rows, height)

        gt_block = np.asarray(
            gt[start:end, :, 0]
        )

        valid_mask = np.isin(
            gt_block,
            [1, 2, 3],
        )

        if not np.any(valid_mask):
            continue

        spectra = np.asarray(
            cube[start:end],
            dtype=np.float32,
        )[valid_mask]

        finite_mask = np.isfinite(spectra).all(axis=1)
        spectra = spectra[finite_mask]

        remaining = max_pixels - selected_count

        if remaining <= 0:
            break

        spectra = spectra[:remaining]

        if len(spectra):
            selected.append(spectra)
            selected_count += len(spectra)

    if not selected:
        raise ValueError(
            "No valid labeled spectra found."
        )

    X = np.concatenate(selected, axis=0)

    X = X / scale

    X = savgol_filter(
        X,
        window_length=artifact["savgol_window"],
        polyorder=artifact["savgol_polyorder"],
        axis=1,
    )

    X = artifact["scaler"].transform(X)
    X = artifact["pca"].transform(X)

    predictions = artifact["classifier"].predict(X)

    counts = {
        CLASS_NAMES[label]: int(
            np.sum(predictions == label)
        )
        for label in [1, 2, 3]
    }

    total = len(predictions)

    percentages = {
        name: round(count / total * 100, 2)
        for name, count in counts.items()
    }

    return {
        "food": food.title(),
        "analysis": "Experimental N2 category classification",
        "model": "Random Forest + 30-component PCA",
        "pixels_analyzed": total,
        "predicted_counts": counts,
        "predicted_percentages": percentages,
        "warning": (
            "Experimental pixel-level classification only. "
            "Ground-truth labels were used to select pixels. "
            "This is not an independently validated nitrogen "
            "measurement or food-safety assessment."
        ),
    }
