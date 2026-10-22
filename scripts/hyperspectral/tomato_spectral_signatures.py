# ============================================================
# TOMATO HSI – CORRECTED SPECTRAL SIGNATURE ANALYSIS
# ============================================================
#
# Purpose:
#   1. Read raw Tomato HSI MATLAB (.mat) cubes
#   2. Validate cube dimensions and numerical values
#   3. Diagnose possible scale differences between files
#   4. Extract class-wise spectral signatures
#   5. Perform per-spectrum normalization for shape comparison
#   6. Calculate spectral variability
#   7. Calculate normalized class separation
#   8. Identify potentially discriminative bands
#   9. Save report-ready CSVs and plots
#
# IMPORTANT:
#   - No wavelengths are invented.
#   - Raw values are preserved.
#   - No arbitrary class-specific multiplication/division is used.
#   - Normalization is performed at the individual-spectrum level.
#
# ============================================================

import os
import glob
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.io import loadmat
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = r"D:\Projects\AI-Food-Quality-Platform"

INPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "raw",
    "hyperspectral",
    "tomato_hsi"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "tomato_spectral_signatures_corrected"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. SETTINGS
# ============================================================

RANDOM_STATE = 42

# Number of spectra sampled from each cube
SAMPLES_PER_CUBE = 5000

EXPECTED_BANDS = 550

# Savitzky-Golay smoothing
SMOOTH_WINDOW = 21
SMOOTH_POLYORDER = 3

rng = np.random.default_rng(RANDOM_STATE)


# ============================================================
# 3. CLASS IDENTIFICATION
# ============================================================

CLASS_NAMES = [
    "Alternaria_alternata",
    "Alternaria_solani",
    "Botrytis_cinerea",
    "Fusarium_oxysporum",
    "Healthy"
]


def identify_class(filename):

    name = os.path.basename(filename).lower()

    if "alternaria_alternata" in name:
        return "Alternaria_alternata"

    elif "alternaria_solani" in name:
        return "Alternaria_solani"

    elif "botrytis_cinerea" in name:
        return "Botrytis_cinerea"

    elif "fusarium_oxysporum" in name:
        return "Fusarium_oxysporum"

    elif "healthy" in name:
        return "Healthy"

    return None


# ============================================================
# 4. FIND MAT FILES
# ============================================================

mat_files = sorted(
    glob.glob(os.path.join(INPUT_DIR, "*.mat"))
)

print("=" * 80)
print("TOMATO HSI – CORRECTED SPECTRAL SIGNATURE ANALYSIS")
print("=" * 80)

print("\nInput directory:")
print(INPUT_DIR)

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nMAT files found:", len(mat_files))

if len(mat_files) == 0:
    raise FileNotFoundError(
        "No .mat files were found in the Tomato HSI directory."
    )


# ============================================================
# 5. DATA STRUCTURES
# ============================================================

raw_spectra_by_class = {
    cls: [] for cls in CLASS_NAMES
}

normalized_spectra_by_class = {
    cls: [] for cls in CLASS_NAMES
}

cube_summary = []


# ============================================================
# 6. NORMALIZATION FUNCTION
# ============================================================
#
# We normalize EACH PIXEL SPECTRUM independently.
#
# This converts:
#
#       x = [x1, x2, ..., x550]
#
# into a unit-length spectral vector:
#
#       x_norm = x / ||x||
#
# This is useful for comparing spectral SHAPE rather than
# absolute magnitude.
#
# It does NOT claim to convert raw sensor values into physical
# reflectance.
#
# ============================================================

def l2_normalize_spectra(X):

    norms = np.linalg.norm(X, axis=1, keepdims=True)

    norms[norms == 0] = 1.0

    return X / norms


# ============================================================
# 7. PROCESS EACH MAT FILE
# ============================================================

print("\n" + "=" * 80)
print("PROCESSING RAW HSI CUBES")
print("=" * 80)

for file_path in mat_files:

    filename = os.path.basename(file_path)

    class_name = identify_class(filename)

    if class_name is None:
        print("\nWARNING: Could not identify class:", filename)
        continue

    print("\n" + "-" * 80)
    print("File:", filename)
    print("Class:", class_name)

    # --------------------------------------------------------
    # Load MATLAB file
    # --------------------------------------------------------

    mat = loadmat(file_path)

    # Find 3D numeric variables
    candidate_variables = []

    for key, value in mat.items():

        if key.startswith("__"):
            continue

        if isinstance(value, np.ndarray) and value.ndim == 3:
            candidate_variables.append((key, value))

    if len(candidate_variables) == 0:

        raise ValueError(
            f"No 3D HSI cube found in {filename}"
        )

    variable_name, cube = candidate_variables[0]

    print("MAT variable:", variable_name)
    print("Cube shape:", cube.shape)
    print("Data type:", cube.dtype)

    # --------------------------------------------------------
    # Validate shape
    # --------------------------------------------------------

    if EXPECTED_BANDS not in cube.shape:

        raise ValueError(
            f"Expected one dimension of {EXPECTED_BANDS} bands, "
            f"but got {cube.shape} in {filename}"
        )

    spectral_axis = cube.shape.index(EXPECTED_BANDS)

    cube = np.moveaxis(
        cube,
        spectral_axis,
        -1
    )

    height, width, bands = cube.shape

    print(
        "Reordered cube shape:",
        cube.shape
    )

    # --------------------------------------------------------
    # Numerical validation
    # --------------------------------------------------------

    nan_count = np.isnan(cube).sum()
    inf_count = np.isinf(cube).sum()

    finite_cube = cube[np.isfinite(cube)]

    raw_min = float(np.min(finite_cube))
    raw_max = float(np.max(finite_cube))
    raw_mean = float(np.mean(finite_cube))
    raw_std = float(np.std(finite_cube))

    raw_median = float(np.median(finite_cube))

    q01 = float(np.percentile(finite_cube, 1))
    q05 = float(np.percentile(finite_cube, 5))
    q25 = float(np.percentile(finite_cube, 25))
    q50 = float(np.percentile(finite_cube, 50))
    q75 = float(np.percentile(finite_cube, 75))
    q95 = float(np.percentile(finite_cube, 95))
    q99 = float(np.percentile(finite_cube, 99))

    print("\nRaw numerical statistics:")

    print("Min:", raw_min)
    print("Max:", raw_max)
    print("Mean:", raw_mean)
    print("Std:", raw_std)
    print("Median:", raw_median)

    print("1%:", q01)
    print("5%:", q05)
    print("25%:", q25)
    print("50%:", q50)
    print("75%:", q75)
    print("95%:", q95)
    print("99%:", q99)

    print("NaN:", nan_count)
    print("Inf:", inf_count)

    # --------------------------------------------------------
    # Reshape:
    #
    # height × width × bands
    #
    # becomes:
    #
    # pixels × bands
    # --------------------------------------------------------

    spectra = cube.reshape(
        -1,
        bands
    ).astype(np.float64)

    # --------------------------------------------------------
    # Remove spectra containing invalid values
    # --------------------------------------------------------

    valid_mask = np.isfinite(spectra).all(axis=1)

    spectra = spectra[valid_mask]

    print(
        "Valid spectra:",
        spectra.shape[0]
    )

    # --------------------------------------------------------
    # Sample spectra
    # --------------------------------------------------------

    n_available = spectra.shape[0]

    n_sample = min(
        SAMPLES_PER_CUBE,
        n_available
    )

    sample_indices = rng.choice(
        n_available,
        size=n_sample,
        replace=False
    )

    sampled = spectra[sample_indices]

    print(
        "Sampled spectra:",
        sampled.shape
    )

    # --------------------------------------------------------
    # Store RAW spectra
    # --------------------------------------------------------

    raw_spectra_by_class[class_name].append(
        sampled
    )

    # --------------------------------------------------------
    # L2 NORMALIZATION
    # --------------------------------------------------------

    normalized = l2_normalize_spectra(
        sampled
    )

    normalized_spectra_by_class[class_name].append(
        normalized
    )

    # --------------------------------------------------------
    # Additional diagnostic:
    # spectral norm distribution
    # --------------------------------------------------------

    norms = np.linalg.norm(
        sampled,
        axis=1
    )

    cube_summary.append({

        "File": filename,
        "Class": class_name,
        "Variable": variable_name,

        "Height": height,
        "Width": width,
        "Bands": bands,

        "Valid_Spectra": n_available,
        "Sampled_Spectra": n_sample,

        "Raw_Min": raw_min,
        "Raw_Max": raw_max,
        "Raw_Mean": raw_mean,
        "Raw_Std": raw_std,
        "Raw_Median": raw_median,

        "Raw_Q01": q01,
        "Raw_Q05": q05,
        "Raw_Q25": q25,
        "Raw_Q50": q50,
        "Raw_Q75": q75,
        "Raw_Q95": q95,
        "Raw_Q99": q99,

        "NaN_Count": int(nan_count),
        "Inf_Count": int(inf_count),

        "Spectral_Norm_Mean": float(np.mean(norms)),
        "Spectral_Norm_Std": float(np.std(norms)),
        "Spectral_Norm_Min": float(np.min(norms)),
        "Spectral_Norm_Max": float(np.max(norms))
    })


# ============================================================
# 8. COMBINE CUBES BY CLASS
# ============================================================

print("\n" + "=" * 80)
print("COMBINING SPECTRA BY CLASS")
print("=" * 80)

raw_class_data = {}
normalized_class_data = {}

for cls in CLASS_NAMES:

    if len(raw_spectra_by_class[cls]) == 0:

        print(
            f"WARNING: No spectra found for {cls}"
        )

        continue

    raw_class_data[cls] = np.vstack(
        raw_spectra_by_class[cls]
    )

    normalized_class_data[cls] = np.vstack(
        normalized_spectra_by_class[cls]
    )

    print(
        cls,
        "RAW:",
        raw_class_data[cls].shape,
        "NORMALIZED:",
        normalized_class_data[cls].shape
    )


# ============================================================
# 9. SAVE RAW SCALE DIAGNOSTICS
# ============================================================

cube_summary_df = pd.DataFrame(
    cube_summary
)

cube_summary_path = os.path.join(
    OUTPUT_DIR,
    "tomato_hsi_raw_scale_diagnostics.csv"
)

cube_summary_df.to_csv(
    cube_summary_path,
    index=False
)

print("\nSaved:")
print(cube_summary_path)


# ============================================================
# 10. RAW CLASS-WISE SPECTRAL SIGNATURES
# ============================================================

raw_signature_rows = []

for cls in CLASS_NAMES:

    if cls not in raw_class_data:
        continue

    X = raw_class_data[cls]

    mean_spectrum = np.mean(
        X,
        axis=0
    )

    std_spectrum = np.std(
        X,
        axis=0
    )

    median_spectrum = np.median(
        X,
        axis=0
    )

    q25_spectrum = np.percentile(
        X,
        25,
        axis=0
    )

    q75_spectrum = np.percentile(
        X,
        75,
        axis=0
    )

    for band_idx in range(EXPECTED_BANDS):

        raw_signature_rows.append({

            "Class": cls,

            "Band": band_idx + 1,

            "Mean_Raw": mean_spectrum[band_idx],
            "Std_Raw": std_spectrum[band_idx],
            "Median_Raw": median_spectrum[band_idx],

            "Q25_Raw": q25_spectrum[band_idx],
            "Q75_Raw": q75_spectrum[band_idx]
        })


raw_signature_df = pd.DataFrame(
    raw_signature_rows
)

raw_signature_path = os.path.join(
    OUTPUT_DIR,
    "tomato_classwise_raw_spectral_signatures.csv"
)

raw_signature_df.to_csv(
    raw_signature_path,
    index=False
)

print("\nSaved:")
print(raw_signature_path)


# ============================================================
# 11. NORMALIZED CLASS-WISE SPECTRAL SIGNATURES
# ============================================================

normalized_signature_rows = []

for cls in CLASS_NAMES:

    if cls not in normalized_class_data:
        continue

    X = normalized_class_data[cls]

    mean_spectrum = np.mean(
        X,
        axis=0
    )

    std_spectrum = np.std(
        X,
        axis=0
    )

    median_spectrum = np.median(
        X,
        axis=0
    )

    q25_spectrum = np.percentile(
        X,
        25,
        axis=0
    )

    q75_spectrum = np.percentile(
        X,
        75,
        axis=0
    )

    for band_idx in range(EXPECTED_BANDS):

        normalized_signature_rows.append({

            "Class": cls,

            "Band": band_idx + 1,

            "Mean_L2_Normalized":
                mean_spectrum[band_idx],

            "Std_L2_Normalized":
                std_spectrum[band_idx],

            "Median_L2_Normalized":
                median_spectrum[band_idx],

            "Q25_L2_Normalized":
                q25_spectrum[band_idx],

            "Q75_L2_Normalized":
                q75_spectrum[band_idx]
        })


normalized_signature_df = pd.DataFrame(
    normalized_signature_rows
)

normalized_signature_path = os.path.join(
    OUTPUT_DIR,
    "tomato_classwise_normalized_spectral_signatures.csv"
)

normalized_signature_df.to_csv(
    normalized_signature_path,
    index=False
)

print("\nSaved:")
print(normalized_signature_path)


# ============================================================
# 12. RAW CLASS MEAN SUMMARY
# ============================================================

raw_summary_rows = []

for cls in CLASS_NAMES:

    if cls not in raw_class_data:
        continue

    X = raw_class_data[cls]

    mean_spectrum = np.mean(
        X,
        axis=0
    )

    raw_summary_rows.append({

        "Class": cls,

        "Overall_Mean":
            float(np.mean(mean_spectrum)),

        "Overall_Std":
            float(np.std(mean_spectrum)),

        "Min_Band_Mean":
            float(np.min(mean_spectrum)),

        "Max_Band_Mean":
            float(np.max(mean_spectrum))
    })


raw_summary_df = pd.DataFrame(
    raw_summary_rows
)

raw_summary_path = os.path.join(
    OUTPUT_DIR,
    "tomato_raw_class_scale_summary.csv"
)

raw_summary_df.to_csv(
    raw_summary_path,
    index=False
)


# ============================================================
# 13. NORMALIZED CLASS SUMMARY
# ============================================================

normalized_summary_rows = []

for cls in CLASS_NAMES:

    if cls not in normalized_class_data:
        continue

    X = normalized_class_data[cls]

    mean_spectrum = np.mean(
        X,
        axis=0
    )

    normalized_summary_rows.append({

        "Class": cls,

        "Overall_Mean":
            float(np.mean(mean_spectrum)),

        "Overall_Std":
            float(np.std(mean_spectrum)),

        "Min_Band_Mean":
            float(np.min(mean_spectrum)),

        "Max_Band_Mean":
            float(np.max(mean_spectrum))
    })


normalized_summary_df = pd.DataFrame(
    normalized_summary_rows
)

normalized_summary_path = os.path.join(
    OUTPUT_DIR,
    "tomato_normalized_class_summary.csv"
)

normalized_summary_df.to_csv(
    normalized_summary_path,
    index=False
)


# ============================================================
# 14. PLOT RAW CLASS SIGNATURES
# ============================================================

plt.figure(
    figsize=(14, 8)
)

for cls in CLASS_NAMES:

    subset = raw_signature_df[
        raw_signature_df["Class"] == cls
    ]

    plt.plot(
        subset["Band"],
        subset["Mean_Raw"],
        label=cls,
        linewidth=1.5
    )

plt.xlabel("Spectral Band")
plt.ylabel("Mean Raw Signal")

plt.title(
    "Tomato HSI – Raw Class-wise Spectral Signatures"
)

plt.legend()

plt.grid(
    alpha=0.25
)

plt.tight_layout()

raw_plot_path = os.path.join(
    OUTPUT_DIR,
    "tomato_raw_classwise_spectral_signatures.png"
)

plt.savefig(
    raw_plot_path,
    dpi=300
)

plt.close()


# ============================================================
# 15. PLOT NORMALIZED SIGNATURES
# ============================================================

plt.figure(
    figsize=(14, 8)
)

for cls in CLASS_NAMES:

    subset = normalized_signature_df[
        normalized_signature_df["Class"] == cls
    ]

    plt.plot(
        subset["Band"],
        subset["Mean_L2_Normalized"],
        label=cls,
        linewidth=1.5
    )

plt.xlabel("Spectral Band")

plt.ylabel(
    "Mean L2-normalized Spectral Response"
)

plt.title(
    "Tomato HSI – Normalized Class-wise Spectral Signatures"
)

plt.legend()

plt.grid(
    alpha=0.25
)

plt.tight_layout()

normalized_plot_path = os.path.join(
    OUTPUT_DIR,
    "tomato_normalized_classwise_spectral_signatures.png"
)

plt.savefig(
    normalized_plot_path,
    dpi=300
)

plt.close()


# ============================================================
# 16. SMOOTHED NORMALIZED SIGNATURES
# ============================================================

plt.figure(
    figsize=(14, 8)
)

for cls in CLASS_NAMES:

    subset = normalized_signature_df[
        normalized_signature_df["Class"] == cls
    ]

    y = subset[
        "Mean_L2_Normalized"
    ].values

    # Safety check
    if len(y) >= SMOOTH_WINDOW:

        smoothed = savgol_filter(
            y,
            SMOOTH_WINDOW,
            SMOOTH_POLYORDER
        )

    else:

        smoothed = y

    plt.plot(
        subset["Band"],
        smoothed,
        label=cls,
        linewidth=2
    )

plt.xlabel("Spectral Band")

plt.ylabel(
    "Smoothed L2-normalized Response"
)

plt.title(
    "Tomato HSI – Smoothed Normalized Spectral Signatures"
)

plt.legend()

plt.grid(
    alpha=0.25
)

plt.tight_layout()

smoothed_plot_path = os.path.join(
    OUTPUT_DIR,
    "tomato_smoothed_normalized_spectral_signatures.png"
)

plt.savefig(
    smoothed_plot_path,
    dpi=300
)

plt.close()


# ============================================================
# 17. NORMALIZED SPECTRAL VARIABILITY
# ============================================================

plt.figure(
    figsize=(14, 8)
)

for cls in CLASS_NAMES:

    subset = normalized_signature_df[
        normalized_signature_df["Class"] == cls
    ]

    plt.plot(
        subset["Band"],
        subset["Std_L2_Normalized"],
        label=cls,
        linewidth=1.3
    )

plt.xlabel("Spectral Band")

plt.ylabel(
    "Spectral Standard Deviation"
)

plt.title(
    "Tomato HSI – Normalized Spectral Variability"
)

plt.legend()

plt.grid(
    alpha=0.25
)

plt.tight_layout()

variability_plot_path = os.path.join(
    OUTPUT_DIR,
    "tomato_normalized_spectral_variability.png"
)

plt.savefig(
    variability_plot_path,
    dpi=300
)

plt.close()


# ============================================================
# 18. BETWEEN-CLASS VARIANCE
# ============================================================

class_mean_matrix = []

valid_classes = []

for cls in CLASS_NAMES:

    if cls not in normalized_class_data:
        continue

    X = normalized_class_data[cls]

    mean_spectrum = np.mean(
        X,
        axis=0
    )

    class_mean_matrix.append(
        mean_spectrum
    )

    valid_classes.append(cls)


class_mean_matrix = np.array(
    class_mean_matrix
)

between_class_variance = np.var(
    class_mean_matrix,
    axis=0
)

variance_df = pd.DataFrame({

    "Band":
        np.arange(1, EXPECTED_BANDS + 1),

    "Between_Class_Variance":
        between_class_variance
})

variance_path = os.path.join(
    OUTPUT_DIR,
    "tomato_normalized_between_class_variance.csv"
)

variance_df.to_csv(
    variance_path,
    index=False
)


# ============================================================
# 19. BETWEEN-CLASS VARIANCE PLOT
# ============================================================

plt.figure(
    figsize=(14, 7)
)

plt.plot(
    variance_df["Band"],
    variance_df["Between_Class_Variance"],
    linewidth=1.5
)

plt.xlabel("Spectral Band")

plt.ylabel(
    "Between-class Variance"
)

plt.title(
    "Tomato HSI – Normalized Between-class Spectral Variance"
)

plt.grid(
    alpha=0.25
)

plt.tight_layout()

variance_plot_path = os.path.join(
    OUTPUT_DIR,
    "tomato_normalized_between_class_variance.png"
)

plt.savefig(
    variance_plot_path,
    dpi=300
)

plt.close()


# ============================================================
# 20. TOP DISCRIMINATIVE BANDS
# ============================================================

TOP_N = 30

top_indices = np.argsort(
    between_class_variance
)[::-1][:TOP_N]

top_bands_df = pd.DataFrame({

    "Rank":
        np.arange(1, TOP_N + 1),

    "Band":
        top_indices + 1,

    "Between_Class_Variance":
        between_class_variance[top_indices]
})

top_bands_path = os.path.join(
    OUTPUT_DIR,
    "tomato_normalized_most_discriminative_bands.csv"
)

top_bands_df.to_csv(
    top_bands_path,
    index=False
)


# ============================================================
# 21. CLASS CENTROID SIMILARITY
# ============================================================

centroids = {}

for cls in valid_classes:

    X = normalized_class_data[cls]

    centroids[cls] = np.mean(
        X,
        axis=0
    )


centroid_matrix = np.vstack(
    [centroids[cls] for cls in valid_classes]
)

similarity_matrix = cosine_similarity(
    centroid_matrix
)

similarity_df = pd.DataFrame(
    similarity_matrix,
    index=valid_classes,
    columns=valid_classes
)

similarity_path = os.path.join(
    OUTPUT_DIR,
    "tomato_class_centroid_cosine_similarity.csv"
)

similarity_df.to_csv(
    similarity_path
)


# ============================================================
# 22. PAIRWISE NORMALIZED EUCLIDEAN DISTANCE
# ============================================================

distance_rows = []

for i in range(len(valid_classes)):

    for j in range(i + 1, len(valid_classes)):

        cls1 = valid_classes[i]
        cls2 = valid_classes[j]

        centroid1 = centroids[cls1]
        centroid2 = centroids[cls2]

        distance = np.linalg.norm(
            centroid1 - centroid2
        )

        cosine_sim = similarity_matrix[i, j]

        distance_rows.append({

            "Class_1": cls1,
            "Class_2": cls2,

            "Normalized_Euclidean_Distance":
                float(distance),

            "Cosine_Similarity":
                float(cosine_sim)
        })


distance_df = pd.DataFrame(
    distance_rows
)

distance_df = distance_df.sort_values(
    "Normalized_Euclidean_Distance",
    ascending=False
)

distance_path = os.path.join(
    OUTPUT_DIR,
    "tomato_normalized_class_separation.csv"
)

distance_df.to_csv(
    distance_path,
    index=False
)


# ============================================================
# 23. PCA OF NORMALIZED SPECTRA
# ============================================================
#
# This is NOT the classifier PCA.
#
# This PCA is only for visualizing whether normalized spectra
# from the five classes occupy distinguishable regions.
#
# ============================================================

pca_sample_per_class = 2000

pca_X = []
pca_y = []

for cls in valid_classes:

    X = normalized_class_data[cls]

    n = min(
        pca_sample_per_class,
        len(X)
    )

    idx = rng.choice(
        len(X),
        n,
        replace=False
    )

    pca_X.append(
        X[idx]
    )

    pca_y.extend(
        [cls] * n
    )


pca_X = np.vstack(
    pca_X
)

pca_model = PCA(
    n_components=2,
    random_state=RANDOM_STATE
)

pca_result = pca_model.fit_transform(
    pca_X
)

pca_df = pd.DataFrame({

    "PC1": pca_result[:, 0],
    "PC2": pca_result[:, 1],
    "Class": pca_y
})

pca_path = os.path.join(
    OUTPUT_DIR,
    "tomato_normalized_spectral_pca.csv"
)

pca_df.to_csv(
    pca_path,
    index=False
)


# ============================================================
# 24. PCA PLOT
# ============================================================

plt.figure(
    figsize=(11, 8)
)

for cls in valid_classes:

    subset = pca_df[
        pca_df["Class"] == cls
    ]

    plt.scatter(
        subset["PC1"],
        subset["PC2"],
        s=8,
        alpha=0.35,
        label=cls
    )

plt.xlabel(
    f"PC1 ({pca_model.explained_variance_ratio_[0] * 100:.2f}%)"
)

plt.ylabel(
    f"PC2 ({pca_model.explained_variance_ratio_[1] * 100:.2f}%)"
)

plt.title(
    "Tomato HSI – PCA of Normalized Spectral Signatures"
)

plt.legend()

plt.grid(
    alpha=0.2
)

plt.tight_layout()

pca_plot_path = os.path.join(
    OUTPUT_DIR,
    "tomato_normalized_spectral_pca.png"
)

plt.savefig(
    pca_plot_path,
    dpi=300
)

plt.close()


# ============================================================
# 25. FINAL VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("FINAL VALIDATION")
print("=" * 80)

print(
    "\nNumber of MAT files:",
    len(mat_files)
)

print(
    "Number of classes:",
    len(valid_classes)
)

print(
    "Expected spectral bands:",
    EXPECTED_BANDS
)

print(
    "Actual spectral bands:",
    class_mean_matrix.shape[1]
)

print(
    "Samples per cube:",
    SAMPLES_PER_CUBE
)

print(
    "\nClass sample counts:"
)

for cls in valid_classes:

    print(
        f"  {cls}: "
        f"{raw_class_data[cls].shape[0]} spectra"
    )


# Check NaN / Inf in final normalized data

all_normalized = np.vstack(
    [
        normalized_class_data[cls]
        for cls in valid_classes
    ]
)

print(
    "\nNormalized matrix shape:",
    all_normalized.shape
)

print(
    "Normalized NaN count:",
    np.isnan(all_normalized).sum()
)

print(
    "Normalized Inf count:",
    np.isinf(all_normalized).sum()
)


# Check normalization

norm_check = np.linalg.norm(
    all_normalized,
    axis=1
)

print(
    "Mean normalized vector norm:",
    np.mean(norm_check)
)

print(
    "Min normalized vector norm:",
    np.min(norm_check)
)

print(
    "Max normalized vector norm:",
    np.max(norm_check)
)


# ============================================================
# 26. OUTPUT LIST
# ============================================================

print("\n" + "=" * 80)
print("OUTPUT FILES")
print("=" * 80)

outputs = [

    cube_summary_path,
    raw_signature_path,
    normalized_signature_path,
    raw_summary_path,
    normalized_summary_path,

    raw_plot_path,
    normalized_plot_path,
    smoothed_plot_path,
    variability_plot_path,

    variance_path,
    variance_plot_path,
    top_bands_path,

    similarity_path,
    distance_path,

    pca_path,
    pca_plot_path
]

for path in outputs:

    print(
        "[OK]" if os.path.exists(path)
        else "[MISSING]",
        os.path.basename(path)
    )


print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

print(
    "\nIMPORTANT INTERPRETATION:"
)

print(
    "The raw spectral signatures preserve the original "
    "numerical scale of each HSI cube."
)

print(
    "The normalized signatures are intended for comparison "
    "of spectral shape while reducing the effect of absolute "
    "signal magnitude."
)

print(
    "No class-specific scaling factor or wavelength calibration "
    "has been assumed."
)

print(
    "The normalized results should therefore be interpreted "
    "as relative spectral-shape differences, not calibrated "
    "reflectance values."
)