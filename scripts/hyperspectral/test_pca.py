"""
test_pca.py

Memory-efficient PCA testing for:

1. Papaya
2. Eggplant
3. Cabbage
"""

from pathlib import Path

import numpy as np

from loaders import HyperspectralLoader
from preprocessing import HSIProcessor
from pca_analysis import HSI_PCA


# ==========================================================
# PROJECT ROOT
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ==========================================================
# DATASET ROOT
# ==========================================================

HYPERSPECTRAL_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "raw"
    / "hyperspectral"
)


# ==========================================================
# OUTPUT DIRECTORY
# ==========================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "graphs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# DATASET PATHS
# ==========================================================

DATASETS = {

    "Papaya": (
        HYPERSPECTRAL_ROOT
        / "papaya_hsi"
        / "Garillos-Manliguez_Hyperspectral Images of Papaya"
        / "HSI_RAW_DATA--cropped-subregions-32x32"
        / "1-10-a-s1.hdr"
    ),

    "Eggplant": (
        HYPERSPECTRAL_ROOT
        / "eggplant_hsi"
        / "Eggplant_Crop"
        / "Eggplant_Reflectance_Data.hdr"
    ),

    "Cabbage": (
        HYPERSPECTRAL_ROOT
        / "cabbage_hsi"
        / "Cabbage_Crop"
        / "Cabbage_Reflectance_Data.hdr"
    )
}


# ==========================================================
# SETTINGS
# ==========================================================

TARGET_VARIANCE = 0.95

MAX_COMPONENTS = 50

SAMPLE_SIZE = 10000

RANDOM_STATE = 42


# ==========================================================
# PROCESS PAPAYA
# ==========================================================

def process_papaya(dataset_path):

    print()
    print("=" * 70)
    print("PAPAYA")
    print("=" * 70)

    print()
    print("Loading dataset...")

    img, cube = (
        HyperspectralLoader.load_hdr(
            dataset_path
        )
    )

    print()
    print("=" * 70)
    print("RAW DATA")
    print("=" * 70)

    print("Shape :", cube.shape)
    print("Dtype :", cube.dtype)

    # ------------------------------------------------------
    # PREPROCESS COMPLETE CUBE
    # ------------------------------------------------------

    print()
    print("Preprocessing...")

    processed_cube = (
        HSIProcessor.process(
            cube
        )
    )

    print()
    print("=" * 70)
    print("PROCESSED DATA")
    print("=" * 70)

    print("Shape :", processed_cube.shape)
    print("Dtype :", processed_cube.dtype)

    print(
        "Minimum :",
        np.min(processed_cube)
    )

    print(
        "Maximum :",
        np.max(processed_cube)
    )

    # ------------------------------------------------------
    # PCA
    # ------------------------------------------------------

    print()
    print("Starting PCA...")

    (
        pca,
        pixels,
        transformed,
        selected_components,
        cumulative_variance
    ) = HSI_PCA.fit(
        processed_cube,
        variance_threshold=TARGET_VARIANCE,
        max_components=MAX_COMPONENTS,
        max_samples=SAMPLE_SIZE,
        random_state=RANDOM_STATE
    )

    print()
    print("=" * 70)
    print("PAPAYA PCA RESULTS")
    print("=" * 70)

    print(
        "Pixel matrix :",
        pixels.shape
    )

    print(
        "PCA matrix   :",
        transformed.shape
    )

    print(
        "Components   :",
        selected_components
    )

    # ------------------------------------------------------
    # EXPLAINED VARIANCE
    # ------------------------------------------------------

    variance = (
        pca.explained_variance_ratio_
    )

    print()
    print("Explained variance:")

    for i, value in enumerate(
        variance
    ):

        print(
            f"PC{i + 1:02d}: "
            f"{value * 100:.2f}%"
        )

    actual_variance = (
        cumulative_variance[
            selected_components - 1
        ]
    )

    print(
        "\nCumulative variance:",
        f"{actual_variance * 100:.2f}%"
    )

    # ------------------------------------------------------
    # GRAPHS
    # ------------------------------------------------------

    HSI_PCA.plot_explained_variance(
        cumulative_variance,
        selected_components,
        "Papaya",
        OUTPUT_DIR
    )

    HSI_PCA.plot_pca_scatter(
        transformed,
        "Papaya",
        OUTPUT_DIR
    )

    print()
    print("Papaya PCA completed successfully.")


# ==========================================================
# PROCESS LARGE DATASET
# ==========================================================

def process_large_dataset(
    dataset_name,
    dataset_path
):

    print()
    print("=" * 70)
    print(dataset_name.upper())
    print("=" * 70)

    print()
    print("Opening dataset using memory-efficient sampling...")

    # ------------------------------------------------------
    # SAMPLE DIRECTLY FROM HDR
    # ------------------------------------------------------

    (
        img,
        samples,
        cube_shape
    ) = HyperspectralLoader.sample_hdr(
        dataset_path,
        sample_size=SAMPLE_SIZE,
        random_state=RANDOM_STATE
    )

    print()
    print("=" * 70)
    print("RAW DATA")
    print("=" * 70)

    print(
        "Original cube :",
        cube_shape
    )

    print(
        "Sample matrix :",
        samples.shape
    )

    print(
        "Dtype         :",
        samples.dtype
    )

    # ------------------------------------------------------
    # PREPROCESS ONLY SAMPLED PIXELS
    # ------------------------------------------------------

    print()
    print("Preprocessing sampled pixels...")

    processed_samples = (
        HSIProcessor.process_samples(
            samples
        )
    )

    print()
    print("=" * 70)
    print("PROCESSED DATA")
    print("=" * 70)

    print(
        "Shape :",
        processed_samples.shape
    )

    print(
        "Dtype :",
        processed_samples.dtype
    )

    print(
        "Minimum :",
        np.min(processed_samples)
    )

    print(
        "Maximum :",
        np.max(processed_samples)
    )

    # ------------------------------------------------------
    # PCA
    # ------------------------------------------------------

    print()
    print("Starting PCA...")

    (
        pca,
        pixels,
        transformed,
        selected_components,
        cumulative_variance
    ) = HSI_PCA.fit_samples(
        processed_samples,
        variance_threshold=TARGET_VARIANCE,
        max_components=MAX_COMPONENTS,
        max_samples=SAMPLE_SIZE,
        random_state=RANDOM_STATE
    )

    # ------------------------------------------------------
    # RESULTS
    # ------------------------------------------------------

    print()
    print("=" * 70)
    print(
        f"{dataset_name.upper()} PCA RESULTS"
    )
    print("=" * 70)

    print(
        "Pixel matrix :",
        pixels.shape
    )

    print(
        "PCA matrix   :",
        transformed.shape
    )

    print(
        "Components   :",
        selected_components
    )

    # ------------------------------------------------------
    # EXPLAINED VARIANCE
    # ------------------------------------------------------

    variance = (
        pca.explained_variance_ratio_
    )

    print()
    print("Explained variance:")

    for i, value in enumerate(
        variance
    ):

        print(
            f"PC{i + 1:02d}: "
            f"{value * 100:.2f}%"
        )

    actual_variance = (
        cumulative_variance[
            selected_components - 1
        ]
    )

    print(
        "\nCumulative variance:",
        f"{actual_variance * 100:.2f}%"
    )

    # ------------------------------------------------------
    # GRAPHS
    # ------------------------------------------------------

    HSI_PCA.plot_explained_variance(
        cumulative_variance,
        selected_components,
        dataset_name,
        OUTPUT_DIR
    )

    HSI_PCA.plot_pca_scatter(
        transformed,
        dataset_name,
        OUTPUT_DIR
    )

    print()
    print(
        f"{dataset_name} PCA completed successfully."
    )


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    # ------------------------------------------------------
    # PAPAYA
    # ------------------------------------------------------

    papaya_path = DATASETS["Papaya"]

    try:

        process_papaya(
            papaya_path
        )

    except Exception as e:

        print()
        print("=" * 70)
        print("ERROR PROCESSING PAPAYA")
        print("=" * 70)

        print(
            type(e).__name__,
            ":",
            e
        )


    # ------------------------------------------------------
    # EGGPLANT
    # ------------------------------------------------------

    eggplant_path = DATASETS["Eggplant"]

    try:

        process_large_dataset(
            "Eggplant",
            eggplant_path
        )

    except Exception as e:

        print()
        print("=" * 70)
        print("ERROR PROCESSING EGGPLANT")
        print("=" * 70)

        print(
            type(e).__name__,
            ":",
            e
        )


    # ------------------------------------------------------
    # CABBAGE
    # ------------------------------------------------------

    cabbage_path = DATASETS["Cabbage"]

    try:

        process_large_dataset(
            "Cabbage",
            cabbage_path
        )

    except Exception as e:

        print()
        print("=" * 70)
        print("ERROR PROCESSING CABBAGE")
        print("=" * 70)

        print(
            type(e).__name__,
            ":",
            e
        )


    # ------------------------------------------------------
    # COMPLETE
    # ------------------------------------------------------

    print()
    print("=" * 70)
    print("ALL PCA TESTS COMPLETED")
    print("=" * 70)