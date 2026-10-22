from pathlib import Path

import numpy as np

from loaders import HyperspectralLoader
from preprocessing import HSIProcessor


# ---------------------------------------------------------
# PAPAYA DATASET
# ---------------------------------------------------------

papaya_path = Path(
    "datasets/raw/hyperspectral/"
    "papaya_hsi/"
    "Garillos-Manliguez_Hyperspectral Images of Papaya/"
    "HSI_RAW_DATA--cropped-subregions-32x32/"
    "1-10-a-s1.hdr"
)


# Load cube
cube = HyperspectralLoader.load_hdr(papaya_path)

print("=" * 60)
print("RAW PAPAYA CUBE")
print("=" * 60)

print("Shape :", cube.shape)
print("Dtype :", cube.dtype)
print("Minimum :", np.min(cube))
print("Maximum :", np.max(cube))


# ---------------------------------------------------------
# REMOVE INVALID VALUES
# ---------------------------------------------------------

clean_cube = HSIProcessor.remove_nan_inf(cube)

print("\n" + "=" * 60)
print("AFTER INVALID VALUE REMOVAL")
print("=" * 60)

print("NaN values :", np.isnan(clean_cube).sum())
print("Infinite values :", np.isinf(clean_cube).sum())


# ---------------------------------------------------------
# NORMALIZATION
# ---------------------------------------------------------

normalized_cube = HSIProcessor.normalize(clean_cube)

print("\n" + "=" * 60)
print("AFTER NORMALIZATION")
print("=" * 60)

print("Minimum :", normalized_cube.min())
print("Maximum :", normalized_cube.max())


# ---------------------------------------------------------
# SPECTRAL SMOOTHING
# ---------------------------------------------------------

smoothed_cube = HSIProcessor.spectral_smoothing(
    normalized_cube,
    window_length=7,
    polyorder=2
)

print("\n" + "=" * 60)
print("AFTER SPECTRAL SMOOTHING")
print("=" * 60)

print("Shape :", smoothed_cube.shape)
print("Dtype :", smoothed_cube.dtype)


# ---------------------------------------------------------
# MEAN CENTERING
# ---------------------------------------------------------

centered_cube = HSIProcessor.mean_center(
    smoothed_cube
)

print("\n" + "=" * 60)
print("AFTER MEAN CENTERING")
print("=" * 60)

print("Shape :", centered_cube.shape)

print(
    "Mean of centered data :",
    np.mean(centered_cube)
)


# ---------------------------------------------------------
# COMPLETE PIPELINE
# ---------------------------------------------------------

processed_cube = HSIProcessor.process(cube)

print("\n" + "=" * 60)
print("COMPLETE PREPROCESSING PIPELINE")
print("=" * 60)

print("Original shape :", cube.shape)
print("Processed shape :", processed_cube.shape)
print("Processed dtype :", processed_cube.dtype)

print("Processed minimum :", processed_cube.min())
print("Processed maximum :", processed_cube.max())

print("\nPreprocessing completed successfully.")