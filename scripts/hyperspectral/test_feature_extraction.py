"""
test_feature_extraction.py

Test the hyperspectral feature extraction module.

This test uses a small synthetic PCA matrix first.
This avoids loading the large hyperspectral datasets
while verifying that feature extraction works correctly.
"""

import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------
# Make scripts/hyperspectral importable
# ---------------------------------------------------------

CURRENT_DIR = Path(__file__).resolve().parent

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))


from feature_extraction import HSIFeatureExtractor


def main():

    print("=" * 70)
    print("HYPERSPECTRAL FEATURE EXTRACTION TEST")
    print("=" * 70)

    # -----------------------------------------------------
    # Create example PCA data
    # -----------------------------------------------------

    np.random.seed(42)

    samples = 100
    components = 10

    pca_data = np.random.randn(
        samples,
        components
    ).astype(np.float32)

    print()
    print("=" * 70)
    print("INPUT PCA DATA")
    print("=" * 70)

    print("Samples    :", pca_data.shape[0])
    print("Components :", pca_data.shape[1])
    print("Shape      :", pca_data.shape)
    print("Dtype      :", pca_data.dtype)

    # -----------------------------------------------------
    # Extract features
    # -----------------------------------------------------

    print()
    print("Extracting features...")

    features = HSIFeatureExtractor.extract(
        pca_data,
        include_pca=True
    )

    # -----------------------------------------------------
    # Display results
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("FEATURE EXTRACTION RESULTS")
    print("=" * 70)

    print("Input shape     :", pca_data.shape)
    print("Output shape    :", features.shape)
    print("Output dtype    :", features.dtype)

    print()
    print("Number of features :", features.shape[1])

    # -----------------------------------------------------
    # Feature names
    # -----------------------------------------------------

    names = HSIFeatureExtractor.feature_names(
        components
    )

    print()
    print("FEATURE NAMES")
    print("=" * 70)

    for index, name in enumerate(names):

        print(
            f"{index + 1:02d}. {name}"
        )

    # -----------------------------------------------------
    # Show first sample
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("FIRST SAMPLE FEATURES")
    print("=" * 70)

    for name, value in zip(
        names,
        features[0]
    ):

        print(
            f"{name:10s}: {value:.6f}"
        )

    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    if not np.isfinite(features).all():

        raise ValueError(
            "Feature matrix contains NaN or infinite values."
        )

    print()
    print("=" * 70)
    print("VALIDATION")
    print("=" * 70)

    print("NaN values       :", np.isnan(features).sum())
    print("Infinite values  :", np.isinf(features).sum())

    print()
    print("=" * 70)
    print("FEATURE EXTRACTION COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()