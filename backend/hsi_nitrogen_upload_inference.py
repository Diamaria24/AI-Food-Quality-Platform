
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
from spectral import open_image

from backend.hsi_foreground import select_foreground_spectra
from backend.hsi_nitrogen_inference import (
    load_model,
    validate_cube,
    CLASS_NAMES,
)


def predict_uploaded_hsi_nitrogen(
    food,
    reflectance_header,
    max_pixels=10000,
    chunk_rows=8,
    foreground_threshold=0.05,
):
    """
    Experimental nitrogen-category classification
    without ground-truth labels.

    The reflectance_header must reference a compatible
    ENVI spectral binary file.

    Uses bounded-memory processing and the existing
    trained scaler, PCA and Random Forest.
    """

    food = food.lower()

    if food not in ("eggplant", "cabbage"):
        raise ValueError("Unsupported food.")

    if max_pixels <= 0 or chunk_rows <= 0:
        raise ValueError(
            "max_pixels and chunk_rows must be positive."
        )

    artifact = load_model(food)

    image = open_image(str(Path(reflectance_header)))

    validate_cube(food, image)

    cube = image.open_memmap(interleave="bip")

    height, width, bands = image.shape

    # Pass 1: reproduce the original global
    # normalization used during model training.
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

    
    # Pass 2: spatially distributed foreground sampling.
    # Select rows across the full image instead of
    # stopping after the first eligible pixels.

    selected = []
    selected_count = 0

    # Distribute sampling across the image height.
    # The number of sampled rows is bounded by max_pixels.
    rows_to_sample = min(height, max_pixels)

    row_indices = np.unique(
        np.linspace(
            0,
            height - 1,
            num=rows_to_sample,
            dtype=int,
        )
    )

    # Allocate approximately equal capacity per row.
    base_quota = max_pixels // len(row_indices)
    extra = max_pixels % len(row_indices)

    for i, row_index in enumerate(row_indices):
        quota = base_quota + (1 if i < extra else 0)

        block = np.asarray(
            cube[row_index:row_index + 1],
            dtype=np.float32,
        )

        spectra = block.reshape(-1, bands)
        normalized = spectra / scale

        foreground, _ = select_foreground_spectra(
            normalized,
            threshold=foreground_threshold,
        )

        if len(foreground) == 0:
            continue

        # Distribute selected pixels across each row.
        take = min(quota, len(foreground))

        column_indices = np.linspace(
            0,
            len(foreground) - 1,
            num=take,
            dtype=int,
        )

        selected.append(
            foreground[column_indices]
        )

        selected_count += take


    X = np.concatenate(selected, axis=0)

    # Original model preprocessing.
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
        for label in (1, 2, 3)
    }

    total = len(predictions)

    percentages = {
        name: round(count / total * 100, 2)
        for name, count in counts.items()
    }

    return {
        "food": food.title(),
        "analysis": (
            "Experimental N2 category classification"
        ),
        "model": (
            "Random Forest + 30-component PCA"
        ),
        "selection_method": (
            "Automatic spectral-intensity foreground selection"
        ),
        "foreground_threshold": foreground_threshold,
        "ground_truth_required": False,
        "pixels_analyzed": total,
        "predicted_counts": counts,
        "predicted_percentages": percentages,
        "warning": (
            "Experimental pixel-level classification. "
            "Foreground pixels were selected automatically "
            "without ground-truth labels. "
            "The model has not been independently validated "
            "on new specimens. Results are not a validated "
            "nitrogen measurement, freshness determination, "
            "or food-safety assessment."
        ),
    }
