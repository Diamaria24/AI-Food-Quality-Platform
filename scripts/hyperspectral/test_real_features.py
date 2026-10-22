"""
test_real_features.py

Connects the actual hyperspectral datasets to:

    Loading
        ↓
    Preprocessing
        ↓
    PCA
        ↓
    Feature Extraction

Datasets:
    Papaya
    Eggplant
    Cabbage

The resulting feature matrices are saved as CSV files.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ==========================================================
# IMPORT PATH
# ==========================================================

CURRENT_DIR = Path(__file__).resolve().parent

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))


from loaders import HyperspectralLoader
from preprocessing import HSIProcessor
from pca_analysis import HSI_PCA
from feature_extraction import HSIFeatureExtractor


# ==========================================================
# PROJECT DIRECTORIES
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

HYPERSPECTRAL_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "raw"
    / "hyperspectral"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "hyperspectral"
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
# PCA SETTINGS
# ==========================================================

TARGET_VARIANCE = 0.95

MAX_COMPONENTS = 50

SAMPLE_SIZE = 10000

RANDOM_STATE = 42


# ==========================================================
# SAVE FEATURES
# ==========================================================

def save_features(
    dataset_name,
    pca_data,
    output_path
):
    """
    Extract features from PCA data and save them
    as a CSV file.
    """

    print()
    print("=" * 70)
    print("FEATURE EXTRACTION")
    print("=" * 70)

    print(
        "PCA input shape :",
        pca_data.shape
    )

    # ------------------------------------------------------
    # Extract features
    # ------------------------------------------------------

    features = HSIFeatureExtractor.extract(
        pca_data,
        include_pca=True
    )

    # ------------------------------------------------------
    # Generate feature names
    # ------------------------------------------------------

    feature_names = (
        HSIFeatureExtractor.feature_names(
            pca_data.shape[1]
        )
    )

    # ------------------------------------------------------
    # Create DataFrame
    # ------------------------------------------------------

    dataframe = pd.DataFrame(
        features,
        columns=feature_names
    )

    # ------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------

    dataframe.to_csv(
        output_path,
        index=False
    )

    # ------------------------------------------------------
    # Information
    # ------------------------------------------------------

    print(
        "Feature matrix shape :",
        dataframe.shape
    )

    print(
        "Number of features   :",
        len(feature_names)
    )

    print(
        "Output file          :",
        output_path
    )

    print(
        "NaN values           :",
        dataframe.isna().sum().sum()
    )

    print(
        "Infinite values      :",
        np.isinf(
            dataframe.to_numpy()
        ).sum()
    )

    return dataframe


# ==========================================================
# PROCESS PAPAYA
# ==========================================================

def process_papaya():

    dataset_name = "Papaya"

    dataset_path = DATASETS[
        dataset_name
    ]

    print()
    print("=" * 70)
    print("PAPAYA")
    print("=" * 70)

    # ------------------------------------------------------
    # Load
    # ------------------------------------------------------

    print()
    print("Loading dataset...")

    img, cube = (
        HyperspectralLoader.load_hdr(
            dataset_path
        )
    )

    print(
        "Raw shape :",
        cube.shape
    )

    print(
        "Raw dtype :",
        cube.dtype
    )

    # ------------------------------------------------------
    # Preprocess
    # ------------------------------------------------------

    print()
    print("Preprocessing...")

    processed_cube = (
        HSIProcessor.process(
            cube
        )
    )

    print(
        "Processed shape :",
        processed_cube.shape
    )

    # ------------------------------------------------------
    # PCA
    # ------------------------------------------------------

    print()
    print("Running PCA...")

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

    print(
        "PCA output shape :",
        transformed.shape
    )

    print(
        "Selected PCs     :",
        selected_components
    )

    print(
        "Variance retained:",
        f"{cumulative_variance[selected_components - 1] * 100:.2f}%"
    )

    # ------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------

    output_path = (
        OUTPUT_DIR
        / "papaya_features.csv"
    )

    save_features(
        dataset_name,
        transformed,
        output_path
    )


# ==========================================================
# PROCESS LARGE DATASETS
# ==========================================================

def process_large_dataset(
    dataset_name
):

    dataset_path = DATASETS[
        dataset_name
    ]

    print()
    print("=" * 70)
    print(dataset_name.upper())
    print("=" * 70)

    # ------------------------------------------------------
    # Sample directly from HDR
    # ------------------------------------------------------

    print()
    print(
        "Sampling pixels directly from HDR..."
    )

    (
        img,
        samples,
        cube_shape
    ) = HyperspectralLoader.sample_hdr(
        dataset_path,
        sample_size=SAMPLE_SIZE,
        random_state=RANDOM_STATE
    )

    print(
        "Original cube :",
        cube_shape
    )

    print(
        "Sample shape  :",
        samples.shape
    )

    # ------------------------------------------------------
    # Preprocess sampled pixels
    # ------------------------------------------------------

    print()
    print(
        "Preprocessing sampled pixels..."
    )

    processed_samples = (
        HSIProcessor.process_samples(
            samples
        )
    )

    print(
        "Processed shape :",
        processed_samples.shape
    )

    # ------------------------------------------------------
    # PCA
    # ------------------------------------------------------

    print()
    print("Running PCA...")

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

    print(
        "PCA output shape :",
        transformed.shape
    )

    print(
        "Selected PCs     :",
        selected_components
    )

    print(
        "Variance retained:",
        f"{cumulative_variance[selected_components - 1] * 100:.2f}%"
    )

    # ------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------

    output_path = (
        OUTPUT_DIR
        / f"{dataset_name.lower()}_features.csv"
    )

    save_features(
        dataset_name,
        transformed,
        output_path
    )


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    # ------------------------------------------------------
    # PAPAYA
    # ------------------------------------------------------

    try:

        process_papaya()

    except Exception as e:

        print()
        print("=" * 70)
        print("ERROR: PAPAYA")
        print("=" * 70)

        print(
            type(e).__name__,
            ":",
            e
        )


    # ------------------------------------------------------
    # EGGPLANT
    # ------------------------------------------------------

    try:

        process_large_dataset(
            "Eggplant"
        )

    except Exception as e:

        print()
        print("=" * 70)
        print("ERROR: EGGPLANT")
        print("=" * 70)

        print(
            type(e).__name__,
            ":",
            e
        )


    # ------------------------------------------------------
    # CABBAGE
    # ------------------------------------------------------

    try:

        process_large_dataset(
            "Cabbage"
        )

    except Exception as e:

        print()
        print("=" * 70)
        print("ERROR: CABBAGE")
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
    print(
        "REAL FEATURE EXTRACTION COMPLETED"
    )
    print("=" * 70)