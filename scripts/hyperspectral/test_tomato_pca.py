"""
test_tomato_pca.py

Tomato fungal hyperspectral PCA analysis.

Dataset:
    Tomato HSI fungal spectral dataset

Classes:
    - Healthy
    - Alternaria alternata
    - Alternaria solani
    - Botrytis cinerea
    - Fusarium oxysporum

Input:
    MATLAB .mat hyperspectral cubes

Cube shape:
    (140, 280, 550)

Spectral range:
    400-950 nm

Pipeline:
    MAT file
        ↓
    Scale correction
        ↓
    NaN / Inf cleaning
        ↓
    Savitzky-Golay smoothing
        ↓
    Mean centering
        ↓
    Pixel sampling
        ↓
    PCA
        ↓
    Automatic 95% variance selection
        ↓
    PCA feature CSV
        ↓
    Visualization
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.io import loadmat
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

TOMATO_DIR = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "raw",
    "hyperspectral",
    "tomato_hsi"
)

OUTPUT_FEATURE_DIR = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "processed",
    "hyperspectral",
    "tomato_pca"
)

OUTPUT_GRAPH_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "graphs",
    "tomato_pca"
)

os.makedirs(OUTPUT_FEATURE_DIR, exist_ok=True)
os.makedirs(OUTPUT_GRAPH_DIR, exist_ok=True)


# PCA configuration
TARGET_VARIANCE = 0.95
MAX_COMPONENTS = 50

# Number of pixels sampled from each MAT file
MAX_PIXELS_PER_FILE = 5000

# Reproducibility
RANDOM_STATE = 42

# Savitzky-Golay parameters
SAVGOL_WINDOW = 7
SAVGOL_POLYORDER = 2


# ============================================================
# CLASS MAPPING
# ============================================================

CLASS_NAMES = {
    "Alternaria_alternata": "Alternaria alternata",
    "Alternaria_solani": "Alternaria solani",
    "Botrytis_cinerea": "Botrytis cinerea",
    "Fusarium_oxysporum": "Fusarium oxysporum",
    "Healthy": "Healthy",
}


# ============================================================
# PRINT HEADER
# ============================================================

def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# FIND MATLAB VARIABLE
# ============================================================

def find_data_variable(mat_data):
    """
    Find the hyperspectral data variable inside a MAT file.

    Ignores MATLAB metadata fields beginning with '__'.
    """

    candidates = []

    for key, value in mat_data.items():

        if key.startswith("__"):
            continue

        if isinstance(value, np.ndarray):

            if value.ndim == 3:
                candidates.append((key, value))

    if not candidates:
        raise ValueError(
            "No 3-dimensional hyperspectral array found in MAT file."
        )

    # Prefer the largest 3D array
    candidates.sort(
        key=lambda item: item[1].size,
        reverse=True
    )

    return candidates[0]


# ============================================================
# CLASS FROM FILENAME
# ============================================================

def get_class_from_filename(filename):

    base = os.path.basename(filename)

    for key, class_name in CLASS_NAMES.items():

        if base.startswith(key):
            return class_name

    return "Unknown"


# ============================================================
# SCALE DETECTION
# ============================================================

def detect_scale(cube):

    maximum = float(np.nanmax(cube))

    if maximum <= 1.5:
        return "0-1-like", 1.0

    elif maximum <= 4095.0 * 1.05:
        return "4095-like", 4095.0

    else:
        return "unknown", maximum


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_cube(cube):

    cube = np.asarray(
        cube,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Remove invalid values
    # --------------------------------------------------------

    cube = np.nan_to_num(
        cube,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # --------------------------------------------------------
    # Detect original scale
    # --------------------------------------------------------

    detected_scale, scale_value = detect_scale(cube)

    # --------------------------------------------------------
    # Convert 4095-like data to 0-1 scale
    # --------------------------------------------------------

    if detected_scale == "4095-like":

        cube = cube / 4095.0

    elif detected_scale == "0-1-like":

        cube = cube

    else:

        minimum = np.min(cube)
        maximum = np.max(cube)

        if maximum > minimum:

            cube = (
                cube - minimum
            ) / (
                maximum - minimum
            )

    # --------------------------------------------------------
    # Savitzky-Golay smoothing
    # --------------------------------------------------------

    cube = savgol_filter(
        cube,
        window_length=SAVGOL_WINDOW,
        polyorder=SAVGOL_POLYORDER,
        axis=2
    )

    # --------------------------------------------------------
    # Mean centering along spectral dimension
    # --------------------------------------------------------

    band_means = np.mean(
        cube,
        axis=(0, 1),
        keepdims=True
    )

    cube = cube - band_means

    return cube.astype(np.float32)


# ============================================================
# PIXEL PREPARATION
# ============================================================

def prepare_pixels(cube, max_pixels=5000):

    height, width, bands = cube.shape

    pixels = cube.reshape(
        height * width,
        bands
    )

    # --------------------------------------------------------
    # Remove invalid rows
    # --------------------------------------------------------

    valid_mask = np.all(
        np.isfinite(pixels),
        axis=1
    )

    pixels = pixels[valid_mask]

    print(
        f"Total pixels   : {height * width}"
    )

    print(
        f"Valid pixels   : {len(pixels)}"
    )

    # --------------------------------------------------------
    # Sampling
    # --------------------------------------------------------

    if len(pixels) > max_pixels:

        rng = np.random.default_rng(
            RANDOM_STATE
        )

        indices = rng.choice(
            len(pixels),
            size=max_pixels,
            replace=False
        )

        pixels = pixels[indices]

        print(
            f"Sampled pixels : {len(pixels)}"
        )

    else:

        print(
            "No sampling required."
        )

    return pixels.astype(np.float32)


# ============================================================
# PROCESS ONE FILE
# ============================================================

def process_file(filename):

    print_header(
        f"PROCESSING: {os.path.basename(filename)}"
    )

    # --------------------------------------------------------
    # Load MAT file
    # --------------------------------------------------------

    mat_data = loadmat(
        filename
    )

    variable_name, cube = find_data_variable(
        mat_data
    )

    cube = np.asarray(
        cube,
        dtype=np.float32
    )

    print(
        f"Variable       : {variable_name}"
    )

    print(
        f"Original shape : {cube.shape}"
    )

    print(
        f"Original dtype : {cube.dtype}"
    )

    # --------------------------------------------------------
    # Check cube dimensions
    # --------------------------------------------------------

    if cube.ndim != 3:

        raise ValueError(
            f"Expected 3D cube, got {cube.ndim}D"
        )

    bands = cube.shape[2]

    if bands != 550:

        print(
            f"WARNING: Expected 550 bands, found {bands}"
        )

    # --------------------------------------------------------
    # Raw statistics
    # --------------------------------------------------------

    raw_min = float(np.min(cube))
    raw_max = float(np.max(cube))
    raw_mean = float(np.mean(cube))

    print()
    print("RAW DATA")
    print("-" * 50)

    print(
        f"Minimum        : {raw_min}"
    )

    print(
        f"Maximum        : {raw_max}"
    )

    print(
        f"Mean           : {raw_mean}"
    )

    print(
        f"NaN values     : {np.isnan(cube).sum()}"
    )

    print(
        f"Infinite values: {np.isinf(cube).sum()}"
    )

    scale_name, scale_value = detect_scale(
        cube
    )

    print(
        f"Detected scale : {scale_name}"
    )

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    print()
    print("PREPROCESSING...")

    processed = preprocess_cube(
        cube
    )

    print()
    print("PROCESSED DATA")
    print("-" * 50)

    print(
        f"Shape          : {processed.shape}"
    )

    print(
        f"Dtype          : {processed.dtype}"
    )

    print(
        f"Minimum        : {processed.min()}"
    )

    print(
        f"Maximum        : {processed.max()}"
    )

    print(
        f"Mean           : {processed.mean()}"
    )

    # --------------------------------------------------------
    # Prepare pixels
    # --------------------------------------------------------

    print()
    print("PREPARING PIXELS...")

    pixels = prepare_pixels(
        processed,
        MAX_PIXELS_PER_FILE
    )

    print(
        f"PCA input      : {pixels.shape}"
    )

    return pixels


# ============================================================
# FIT PCA
# ============================================================

def fit_pca(all_pixels):

    print_header(
        "FITTING TOMATO PCA"
    )

    samples, bands = all_pixels.shape

    maximum_pcs = min(
        MAX_COMPONENTS,
        samples,
        bands
    )

    print(
        f"Samples        : {samples}"
    )

    print(
        f"Spectral bands : {bands}"
    )

    print(
        f"Maximum PCs    : {maximum_pcs}"
    )

    # --------------------------------------------------------
    # Fit maximum allowed PCA
    # --------------------------------------------------------

    pca_full = PCA(
        n_components=maximum_pcs,
        random_state=RANDOM_STATE
    )

    transformed_full = pca_full.fit_transform(
        all_pixels
    )

    explained = (
        pca_full.explained_variance_ratio_
    )

    cumulative = np.cumsum(
        explained
    )

    # --------------------------------------------------------
    # Find components needed for 95%
    # --------------------------------------------------------

    components_required = (
        np.searchsorted(
            cumulative,
            TARGET_VARIANCE
        ) + 1
    )

    components_required = min(
        components_required,
        maximum_pcs
    )

    retained_variance = cumulative[
        components_required - 1
    ]

    print()
    print("=" * 65)
    print("TOMATO PCA COMPONENT SELECTION")
    print("=" * 65)

    print(
        f"Target variance        : "
        f"{TARGET_VARIANCE * 100:.2f}%"
    )

    print(
        f"Components required    : "
        f"{components_required}"
    )

    print(
        f"Maximum allowed        : "
        f"{maximum_pcs}"
    )

    print(
        f"Components selected    : "
        f"{components_required}"
    )

    print(
        f"Actual retained        : "
        f"{retained_variance * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Final PCA transformation
    # --------------------------------------------------------

    pca = PCA(
        n_components=components_required,
        random_state=RANDOM_STATE
    )

    transformed = pca.fit_transform(
        all_pixels
    )

    print()
    print(
        "Explained variance:"
    )

    for i, value in enumerate(
        pca.explained_variance_ratio_,
        start=1
    ):

        print(
            f"PC{i:02d}: "
            f"{value * 100:.2f}%"
        )

    print()
    print(
        f"Cumulative variance: "
        f"{np.sum(pca.explained_variance_ratio_) * 100:.2f}%"
    )

    return pca, transformed


# ============================================================
# SAVE PCA RESULTS
# ============================================================

def save_pca_features(
    transformed,
    labels,
    source_files,
    output_name
):

    number_components = transformed.shape[1]

    columns = [
        f"PC{i}"
        for i in range(1, number_components + 1)
    ]

    dataframe = pd.DataFrame(
        transformed,
        columns=columns
    )

    dataframe["class"] = labels
    dataframe["source_file"] = source_files

    output_path = os.path.join(
        OUTPUT_FEATURE_DIR,
        output_name
    )

    dataframe.to_csv(
        output_path,
        index=False
    )

    print()
    print(
        f"Saved PCA features:"
    )

    print(
        output_path
    )

    print(
        f"Shape: {dataframe.shape}"
    )

    return dataframe


# ============================================================
# EXPLAINED VARIANCE GRAPH
# ============================================================

def save_explained_variance_graph(
    pca
):

    explained = (
        pca.explained_variance_ratio_
    )

    cumulative = np.cumsum(
        explained
    )

    components = np.arange(
        1,
        len(cumulative) + 1
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        components,
        cumulative,
        marker="o",
        markersize=3
    )

    plt.axhline(
        TARGET_VARIANCE,
        linestyle="--",
        label="95% variance"
    )

    selected = np.searchsorted(
        cumulative,
        TARGET_VARIANCE
    ) + 1

    plt.axvline(
        selected,
        linestyle="--",
        label=f"Selected components: {selected}"
    )

    plt.xlabel(
        "Number of Principal Components"
    )

    plt.ylabel(
        "Cumulative Explained Variance"
    )

    plt.title(
        "Tomato HSI - PCA Explained Variance"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_GRAPH_DIR,
        "tomato_pca_explained_variance.png"
    )

    plt.savefig(
        output_path,
        dpi=200
    )

    plt.close()

    print(
        f"Saved graph: {output_path}"
    )


# ============================================================
# PC1 VS PC2 GRAPH
# ============================================================

def save_pc_projection(
    transformed,
    labels
):

    if transformed.shape[1] < 2:

        print(
            "Not enough PCA components "
            "for PC1 vs PC2 graph."
        )

        return

    plt.figure(
        figsize=(10, 7)
    )

    unique_classes = sorted(
        set(labels)
    )

    for class_name in unique_classes:

        mask = np.array(
            labels
        ) == class_name

        plt.scatter(
            transformed[mask, 0],
            transformed[mask, 1],
            s=8,
            alpha=0.5,
            label=class_name
        )

    plt.xlabel(
        "Principal Component 1"
    )

    plt.ylabel(
        "Principal Component 2"
    )

    plt.title(
        "Tomato HSI - PCA Projection: PC1 vs PC2"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_GRAPH_DIR,
        "tomato_pca_pc1_pc2.png"
    )

    plt.savefig(
        output_path,
        dpi=200
    )

    plt.close()

    print(
        f"Saved graph: {output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "TOMATO HSI PCA ANALYSIS"
    )

    print(
        f"Dataset directory:"
    )

    print(
        TOMATO_DIR
    )

    # --------------------------------------------------------
    # Find MAT files
    # --------------------------------------------------------

    mat_files = sorted(
        glob.glob(
            os.path.join(
                TOMATO_DIR,
                "*.mat"
            )
        )
    )

    if not mat_files:

        print(
            "ERROR: No MAT files found."
        )

        return

    print()
    print(
        f"MAT files found: {len(mat_files)}"
    )

    # --------------------------------------------------------
    # Process each file
    # --------------------------------------------------------

    all_pixels = []
    all_labels = []
    all_source_files = []

    for filename in mat_files:

        try:

            class_name = get_class_from_filename(
                filename
            )

            print()
            print(
                f"Class: {class_name}"
            )

            pixels = process_file(
                filename
            )

            all_pixels.append(
                pixels
            )

            all_labels.extend(
                [class_name] * len(pixels)
            )

            all_source_files.extend(
                [os.path.basename(filename)] *
                len(pixels)
            )

        except Exception as error:

            print()
            print(
                f"ERROR PROCESSING "
                f"{os.path.basename(filename)}"
            )

            print(
                type(error).__name__,
                ":",
                error
            )

    # --------------------------------------------------------
    # Check successful files
    # --------------------------------------------------------

    if not all_pixels:

        print(
            "No files were successfully processed."
        )

        return

    # --------------------------------------------------------
    # Combine pixels
    # --------------------------------------------------------

    print_header(
        "COMBINING TOMATO PCA INPUT"
    )

    combined_pixels = np.vstack(
        all_pixels
    )

    print(
        f"Combined PCA input: "
        f"{combined_pixels.shape}"
    )

    print(
        f"Total samples     : "
        f"{len(combined_pixels)}"
    )

    # --------------------------------------------------------
    # Fit PCA
    # --------------------------------------------------------

    pca, transformed = fit_pca(
        combined_pixels
    )

    # --------------------------------------------------------
    # Save features
    # --------------------------------------------------------

    dataframe = save_pca_features(
        transformed,
        all_labels,
        all_source_files,
        "tomato_pca_features.csv"
    )

    # --------------------------------------------------------
    # Save graphs
    # --------------------------------------------------------

    save_explained_variance_graph(
        pca
    )

    save_pc_projection(
        transformed,
        all_labels
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print_header(
        "TOMATO PCA VALIDATION"
    )

    print(
        f"Original spectral bands : "
        f"{combined_pixels.shape[1]}"
    )

    print(
        f"Selected PCA components : "
        f"{transformed.shape[1]}"
    )

    print(
        f"Compression ratio       : "
        f"{combined_pixels.shape[1] / transformed.shape[1]:.2f}:1"
    )

    print(
        f"PCA matrix shape        : "
        f"{transformed.shape}"
    )

    print(
        f"NaN values              : "
        f"{np.isnan(transformed).sum()}"
    )

    print(
        f"Infinite values         : "
        f"{np.isinf(transformed).sum()}"
    )

    print()
    print(
        "CLASS DISTRIBUTION"
    )

    print(
        dataframe["class"].value_counts()
    )

    print_header(
        "TOMATO PCA COMPLETED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()