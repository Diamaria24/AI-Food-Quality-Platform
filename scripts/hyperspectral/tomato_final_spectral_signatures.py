"""
======================================================================
TOMATO HSI FINAL SPECTRAL SIGNATURE ANALYSIS
======================================================================

Purpose:
    Extract scientifically comparable class-wise spectral signatures
    from the original Tomato hyperspectral MAT files.

Dataset:
    10 MATLAB hyperspectral cubes
    5 classes
    2 samples per class
    Cube shape: 140 x 280 x 550

Classes:
    Healthy
    Alternaria_alternata
    Alternaria_solani
    Botrytis_cinerea
    Fusarium_oxysporum

Important:
    The original dataset does NOT contain wavelength calibration.
    Therefore the x-axis is reported as Band 1 ... Band 550.

Pipeline:
    Raw MAT cube
        ->
    cube validation
        ->
    per-cube scale normalization
        ->
    spectral sampling
        ->
    class-wise mean spectrum
        ->
    standard deviation
        ->
    coefficient of variation
        ->
    class separation analysis
        ->
    spectral plots
        ->
    CSV outputs

No PCA is performed.
No classifier is trained.
No train/test split is created.
No artificial labels are created.

======================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")


# =====================================================================
# CONFIGURATION
# =====================================================================

INPUT_DIR = (
    r"D:\Projects\AI-Food-Quality-Platform"
    r"\datasets\raw\hyperspectral\tomato_hsi"
)

OUTPUT_DIR = (
    r"D:\Projects\AI-Food-Quality-Platform"
    r"\outputs\tomato_spectral_signatures"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

EXPECTED_HEIGHT = 140
EXPECTED_WIDTH = 280
EXPECTED_BANDS = 550

SAMPLES_PER_FILE = 2000

RANDOM_STATE = 42

# Number of bands
BANDS = np.arange(1, EXPECTED_BANDS + 1)


# =====================================================================
# CLASS MAPPING
# =====================================================================

CLASS_MAPPING = {
    "Alternaria_alternata": "Alternaria_alternata",
    "Alternaria_solani": "Alternaria_solani",
    "Botrytis_cinerea": "Botrytis_cinerea",
    "Fusarium_oxysporum": "Fusarium_oxysporum",
    "Healthy": "Healthy",
}


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def identify_mat_variable(mat_data):
    """
    Identify the hyperspectral cube variable.

    The inspected Tomato files contain variables such as:
        Day_one_1
        Day_one_2

    This function does not hard-code the variable name.
    It identifies the first 3-dimensional numeric array.
    """

    candidates = []

    for key, value in mat_data.items():

        if key.startswith("__"):
            continue

        if isinstance(value, np.ndarray):

            if value.ndim == 3 and np.issubdtype(
                value.dtype, np.number
            ):
                candidates.append((key, value))

    if len(candidates) == 0:
        raise ValueError(
            "No 3-dimensional numerical hyperspectral cube "
            "was found in the MAT file."
        )

    if len(candidates) > 1:

        print("WARNING: Multiple 3D numerical variables found.")
        print("Using:", candidates[0][0])

    return candidates[0]


def detect_class(filename):
    """
    Determine class from filename.
    """

    base = os.path.splitext(filename)[0]

    for class_name in CLASS_MAPPING:

        if base.startswith(class_name):
            return class_name

    raise ValueError(
        f"Could not determine Tomato class from filename: {filename}"
    )


def robust_minmax_normalize(cube):
    """
    Normalize each hyperspectral cube independently.

    This is necessary because QC showed that some cubes have values
    approximately in the 0-0.8 range while others are approximately
    in the 0-4095 range.

    The transformation is:

        normalized = (x - min) / (max - min)

    This makes signatures comparable in relative spectral shape.

    Constant cubes are rejected.
    """

    cube = cube.astype(np.float32, copy=False)

    finite_mask = np.isfinite(cube)

    if not finite_mask.all():

        raise ValueError(
            "Cube contains NaN or Inf values."
        )

    cube_min = float(cube.min())
    cube_max = float(cube.max())

    if cube_max <= cube_min:

        raise ValueError(
            "Cube has no usable dynamic range."
        )

    normalized = (
        cube - cube_min
    ) / (
        cube_max - cube_min
    )

    return normalized, cube_min, cube_max


def sample_spectra(cube, n_samples, rng):
    """
    Randomly sample pixels from a cube.

    Input:
        H x W x Bands

    Output:
        n_samples x Bands
    """

    height, width, bands = cube.shape

    total_pixels = height * width

    n_samples = min(n_samples, total_pixels)

    indices = rng.choice(
        total_pixels,
        size=n_samples,
        replace=False
    )

    rows = indices // width
    cols = indices % width

    spectra = cube[rows, cols, :]

    return spectra.astype(np.float32)


# =====================================================================
# START
# =====================================================================

print_header(
    "TOMATO HSI FINAL SPECTRAL SIGNATURE ANALYSIS"
)

print("""
Input:
    Original Tomato MATLAB hyperspectral cubes

Output:
    Class-wise normalized spectral signatures

Important:
    No wavelength calibration is available.
    Therefore spectral positions are reported as band indices.

No PCA.
No classifier.
No train/test split.
""")


# =====================================================================
# FILE DISCOVERY
# =====================================================================

print_header("DISCOVERING TOMATO MAT FILES")

mat_files = sorted(
    [
        f
        for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(".mat")
    ]
)

if len(mat_files) != 10:

    raise ValueError(
        f"Expected 10 MAT files, found {len(mat_files)}."
    )

for filename in mat_files:
    print(filename)

print()
print("Total MAT files:", len(mat_files))


# =====================================================================
# RANDOM GENERATOR
# =====================================================================

rng = np.random.default_rng(RANDOM_STATE)


# =====================================================================
# STORAGE
# =====================================================================

class_spectra = {}

file_statistics = []

all_sample_records = []


# =====================================================================
# LOAD AND PROCESS EACH CUBE
# =====================================================================

print_header("LOADING AND NORMALIZING RAW TOMATO CUBES")

for filename in mat_files:

    path = os.path.join(INPUT_DIR, filename)

    print()
    print("-" * 70)
    print("FILE:", filename)

    # ---------------------------------------------------------------
    # CLASS
    # ---------------------------------------------------------------

    class_name = detect_class(filename)

    print("Class:", class_name)

    # ---------------------------------------------------------------
    # LOAD MAT
    # ---------------------------------------------------------------

    mat_data = loadmat(path)

    variable_name, cube = identify_mat_variable(mat_data)

    print("Variable:", variable_name)
    print("Original shape:", cube.shape)
    print("Original dtype:", cube.dtype)

    # ---------------------------------------------------------------
    # SHAPE VALIDATION
    # ---------------------------------------------------------------

    if cube.shape != (
        EXPECTED_HEIGHT,
        EXPECTED_WIDTH,
        EXPECTED_BANDS,
    ):

        raise ValueError(
            f"Unexpected cube shape for {filename}: "
            f"{cube.shape}. "
            f"Expected "
            f"{EXPECTED_HEIGHT} x "
            f"{EXPECTED_WIDTH} x "
            f"{EXPECTED_BANDS}."
        )

    # ---------------------------------------------------------------
    # NUMERICAL VALIDATION
    # ---------------------------------------------------------------

    nan_count = int(np.isnan(cube).sum())
    inf_count = int(np.isinf(cube).sum())

    if nan_count > 0 or inf_count > 0:

        raise ValueError(
            f"{filename} contains invalid values. "
            f"NaN={nan_count}, Inf={inf_count}"
        )

    raw_min = float(cube.min())
    raw_max = float(cube.max())
    raw_mean = float(cube.mean())

    print("Raw min:", raw_min)
    print("Raw max:", raw_max)
    print("Raw mean:", raw_mean)
    print("NaN:", nan_count)
    print("Inf:", inf_count)

    # ---------------------------------------------------------------
    # NORMALIZATION
    # ---------------------------------------------------------------

    normalized_cube, scale_min, scale_max = (
        robust_minmax_normalize(cube)
    )

    print("Normalization:")
    print("    Min used:", scale_min)
    print("    Max used:", scale_max)

    print(
        "Normalized range:",
        float(normalized_cube.min()),
        "to",
        float(normalized_cube.max())
    )

    # ---------------------------------------------------------------
    # SAMPLE SPECTRA
    # ---------------------------------------------------------------

    spectra = sample_spectra(
        normalized_cube,
        SAMPLES_PER_FILE,
        rng
    )

    print("Sampled spectra:", spectra.shape)

    # ---------------------------------------------------------------
    # STORE CLASS DATA
    # ---------------------------------------------------------------

    if class_name not in class_spectra:
        class_spectra[class_name] = []

    class_spectra[class_name].append(spectra)

    # ---------------------------------------------------------------
    # FILE STATISTICS
    # ---------------------------------------------------------------

    file_statistics.append(
        {
            "file": filename,
            "class": class_name,
            "variable": variable_name,
            "height": EXPECTED_HEIGHT,
            "width": EXPECTED_WIDTH,
            "spectral_bands": EXPECTED_BANDS,
            "raw_min": raw_min,
            "raw_max": raw_max,
            "raw_mean": raw_mean,
            "normalized_min": float(
                normalized_cube.min()
            ),
            "normalized_max": float(
                normalized_cube.max()
            ),
            "sampled_spectra": len(spectra),
        }
    )

    # Free memory
    del cube
    del normalized_cube
    del spectra


# =====================================================================
# VERIFY CLASSES
# =====================================================================

print_header("VERIFYING CLASS DATA")

expected_classes = set(CLASS_MAPPING.keys())

found_classes = set(class_spectra.keys())

print("Expected classes:")
for c in sorted(expected_classes):
    print("   ", c)

print()
print("Found classes:")
for c in sorted(found_classes):
    print("   ", c)

if expected_classes != found_classes:

    raise ValueError(
        "Class mismatch detected."
    )


# =====================================================================
# COMBINE SPECTRA BY CLASS
# =====================================================================

print_header("COMBINING SAMPLED SPECTRA BY CLASS")

combined_class_spectra = {}

for class_name in sorted(class_spectra):

    combined = np.vstack(
        class_spectra[class_name]
    )

    combined_class_spectra[class_name] = combined

    print(
        f"{class_name:25s}:",
        combined.shape
    )


# =====================================================================
# CALCULATE CLASS-WISE SIGNATURES
# =====================================================================

print_header("CALCULATING CLASS-WISE SPECTRAL SIGNATURES")

signature_rows = []

for class_name in sorted(combined_class_spectra):

    spectra = combined_class_spectra[class_name]

    mean_spectrum = np.mean(
        spectra,
        axis=0
    )

    std_spectrum = np.std(
        spectra,
        axis=0
    )

    median_spectrum = np.median(
        spectra,
        axis=0
    )

    q25 = np.percentile(
        spectra,
        25,
        axis=0
    )

    q75 = np.percentile(
        spectra,
        75,
        axis=0
    )

    for i in range(EXPECTED_BANDS):

        mean_value = float(mean_spectrum[i])
        std_value = float(std_spectrum[i])

        if abs(mean_value) > 1e-12:

            coefficient_variation = (
                std_value / abs(mean_value)
            )

        else:

            coefficient_variation = np.nan

        signature_rows.append(
            {
                "class": class_name,
                "band": int(BANDS[i]),
                "mean_reflectance": mean_value,
                "std_reflectance": std_value,
                "median_reflectance": float(
                    median_spectrum[i]
                ),
                "q25_reflectance": float(
                    q25[i]
                ),
                "q75_reflectance": float(
                    q75[i]
                ),
                "coefficient_of_variation":
                    float(coefficient_variation)
            }
        )


signature_df = pd.DataFrame(signature_rows)


# =====================================================================
# SAVE SIGNATURE DATA
# =====================================================================

signature_path = os.path.join(
    OUTPUT_DIR,
    "tomato_classwise_spectral_signatures.csv"
)

signature_df.to_csv(
    signature_path,
    index=False
)

print()
print("Saved:")
print(signature_path)


# =====================================================================
# BAND-WISE CLASS SEPARATION
# =====================================================================

print_header(
    "CALCULATING BETWEEN-CLASS SPECTRAL SEPARATION"
)

mean_table = signature_df.pivot(
    index="band",
    columns="class",
    values="mean_reflectance"
)

std_table = signature_df.pivot(
    index="band",
    columns="class",
    values="std_reflectance"
)


# ---------------------------------------------------------------------
# Between-class standard deviation
# ---------------------------------------------------------------------

between_class_std = mean_table.std(
    axis=1
)

# ---------------------------------------------------------------------
# Mean within-class standard deviation
# ---------------------------------------------------------------------

within_class_std = std_table.mean(
    axis=1
)

# ---------------------------------------------------------------------
# Separation ratio
# ---------------------------------------------------------------------

separation_ratio = (
    between_class_std /
    (within_class_std + 1e-12)
)

band_separation_df = pd.DataFrame(
    {
        "band": BANDS,
        "between_class_std":
            between_class_std.values,
        "mean_within_class_std":
            within_class_std.values,
        "between_within_ratio":
            separation_ratio.values
    }
)

band_separation_df = (
    band_separation_df
    .sort_values(
        "between_within_ratio",
        ascending=False
    )
)

separation_path = os.path.join(
    OUTPUT_DIR,
    "tomato_spectral_band_separation.csv"
)

band_separation_df.to_csv(
    separation_path,
    index=False
)

print("Saved:")
print(separation_path)


# =====================================================================
# TOP DISCRIMINATIVE BANDS
# =====================================================================

print_header("TOP DISCRIMINATIVE SPECTRAL BANDS")

top_bands = band_separation_df.head(30)

print(
    top_bands.to_string(
        index=False
    )
)

top_band_path = os.path.join(
    OUTPUT_DIR,
    "tomato_top_discriminative_bands.csv"
)

top_bands.to_csv(
    top_band_path,
    index=False
)

print()
print("Saved:")
print(top_band_path)


# =====================================================================
# CLASS SUMMARY
# =====================================================================

print_header("CLASS-LEVEL SPECTRAL SUMMARY")

class_summary_rows = []

for class_name in sorted(combined_class_spectra):

    spectra = combined_class_spectra[class_name]

    mean_signature = spectra.mean(
        axis=0
    )

    class_summary_rows.append(
        {
            "class": class_name,
            "spectra_count": spectra.shape[0],
            "bands": spectra.shape[1],
            "mean_over_all_bands":
                float(mean_signature.mean()),
            "std_over_all_bands":
                float(mean_signature.std()),
            "minimum_signature_value":
                float(mean_signature.min()),
            "maximum_signature_value":
                float(mean_signature.max())
        }
    )

class_summary_df = pd.DataFrame(
    class_summary_rows
)

class_summary_path = os.path.join(
    OUTPUT_DIR,
    "tomato_class_spectral_summary.csv"
)

class_summary_df.to_csv(
    class_summary_path,
    index=False
)

print(
    class_summary_df.to_string(
        index=False
    )
)

print()
print("Saved:")
print(class_summary_path)


# =====================================================================
# PLOT 1 — ALL CLASS MEAN SIGNATURES
# =====================================================================

print_header(
    "CREATING CLASS-WISE MEAN SPECTRAL SIGNATURE PLOT"
)

plt.figure(
    figsize=(14, 7)
)

for class_name in sorted(mean_table.columns):

    plt.plot(
        BANDS,
        mean_table[class_name].values,
        linewidth=2,
        label=class_name
    )

plt.xlabel("Spectral Band")
plt.ylabel("Normalized Reflectance")
plt.title(
    "Tomato Hyperspectral Class-Wise Spectral Signatures"
)

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plot_path = os.path.join(
    OUTPUT_DIR,
    "tomato_classwise_spectral_signatures.png"
)

plt.savefig(
    plot_path,
    dpi=300
)

plt.close()

print("Saved:")
print(plot_path)


# =====================================================================
# PLOT 2 — SIGNATURES WITH VARIABILITY
# =====================================================================

print_header(
    "CREATING SIGNATURE VARIABILITY PLOT"
)

plt.figure(
    figsize=(14, 7)
)

for class_name in sorted(mean_table.columns):

    mean_values = mean_table[class_name].values
    std_values = std_table[class_name].values

    upper = mean_values + std_values
    lower = mean_values - std_values

    plt.plot(
        BANDS,
        mean_values,
        linewidth=2,
        label=class_name
    )

    plt.fill_between(
        BANDS,
        lower,
        upper,
        alpha=0.12
    )

plt.xlabel("Spectral Band")
plt.ylabel("Normalized Reflectance")
plt.title(
    "Tomato Spectral Signatures with ±1 Standard Deviation"
)

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

variability_plot = os.path.join(
    OUTPUT_DIR,
    "tomato_classwise_spectral_variability.png"
)

plt.savefig(
    variability_plot,
    dpi=300
)

plt.close()

print("Saved:")
print(variability_plot)


# =====================================================================
# PLOT 3 — BAND SEPARATION
# =====================================================================

print_header(
    "CREATING SPECTRAL BAND SEPARATION PLOT"
)

# Restore band order for plotting

separation_plot_df = (
    band_separation_df
    .sort_values("band")
)

plt.figure(
    figsize=(14, 6)
)

plt.plot(
    separation_plot_df["band"],
    separation_plot_df[
        "between_within_ratio"
    ],
    linewidth=1.5
)

plt.xlabel("Spectral Band")
plt.ylabel(
    "Between-Class / Within-Class Ratio"
)

plt.title(
    "Tomato Spectral Band Discriminative Power"
)

plt.grid(True, alpha=0.3)
plt.tight_layout()

separation_plot_path = os.path.join(
    OUTPUT_DIR,
    "tomato_spectral_band_separation.png"
)

plt.savefig(
    separation_plot_path,
    dpi=300
)

plt.close()

print("Saved:")
print(separation_plot_path)


# =====================================================================
# PLOT 4 — TOP DISCRIMINATIVE REGION
# =====================================================================

print_header(
    "CREATING TOP DISCRIMINATIVE BAND PLOT"
)

top_band_numbers = set(
    top_bands["band"].astype(int)
)

plt.figure(
    figsize=(14, 7)
)

for class_name in sorted(mean_table.columns):

    plt.plot(
        BANDS,
        mean_table[class_name].values,
        linewidth=1.8,
        label=class_name
    )

# Mark top bands

for band in sorted(top_band_numbers):

    plt.axvline(
        band,
        alpha=0.08
    )

plt.xlabel("Spectral Band")
plt.ylabel("Normalized Reflectance")
plt.title(
    "Tomato Spectral Signatures with Top Discriminative Bands"
)

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

top_plot_path = os.path.join(
    OUTPUT_DIR,
    "tomato_spectral_signatures_top_bands.png"
)

plt.savefig(
    top_plot_path,
    dpi=300
)

plt.close()

print("Saved:")
print(top_plot_path)


# =====================================================================
# SAVE FILE STATISTICS
# =====================================================================

file_stats_df = pd.DataFrame(
    file_statistics
)

file_stats_path = os.path.join(
    OUTPUT_DIR,
    "tomato_spectral_file_statistics.csv"
)

file_stats_df.to_csv(
    file_stats_path,
    index=False
)

print()
print("Saved:")
print(file_stats_path)


# =====================================================================
# FINAL SUMMARY
# =====================================================================

print_header(
    "TOMATO FINAL SPECTRAL SIGNATURE ANALYSIS COMPLETED"
)

print()
print("MAT files analyzed       :", len(mat_files))
print("Tomato classes            :", len(found_classes))
print("Spectral bands            :", EXPECTED_BANDS)
print(
    "Samples per file          :",
    SAMPLES_PER_FILE
)
print(
    "Total sampled spectra     :",
    len(mat_files) * SAMPLES_PER_FILE
)

print()
print("Normalization:")
print(
    "Per-cube min-max normalization"
)

print()
print("Wavelength calibration:")
print(
    "NOT AVAILABLE"
)

print(
    "Spectral axis             : Band 1-550"
)

print()
print("No PCA performed.")
print("No classifier trained.")
print("No train/test split created.")
print("No raw MAT file modified.")

print()
print("OUTPUT DIRECTORY:")
print(OUTPUT_DIR)

print()
print("MAIN OUTPUTS:")

print(
    signature_path
)

print(
    separation_path
)

print(
    top_band_path
)

print(
    class_summary_path
)

print(
    plot_path
)

print(
    variability_plot
)

print(
    separation_plot_path
)

print(
    top_plot_path
)

print(
    file_stats_path
)

print()
print("=" * 70)
print("NEXT STEP")
print("=" * 70)

print("""
Review the generated Tomato spectral signatures.

Check:
    1. Class-wise spectral curves
    2. Spectral variability
    3. Discriminative bands
    4. Between-class / within-class separation
    5. Whether the signatures are suitable for the final
       Hyperspectral Imaging module report.

IMPORTANT:
    The band numbers are NOT wavelengths.
    Do not describe Band 263 as 263 nm, etc.

    A wavelength axis can only be added if the original
    sensor wavelength calibration is obtained.
""")

print("=" * 70)