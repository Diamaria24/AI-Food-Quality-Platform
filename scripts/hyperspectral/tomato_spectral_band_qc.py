"""
======================================================================
TOMATO HSI SPECTRAL BAND QUALITY CONTROL
======================================================================

Purpose
-------
Final quality-control analysis for the Tomato hyperspectral dataset.

Checks:
    1. Raw spectral-band statistics
    2. Per-band mean and standard deviation
    3. Coefficient of variation
    4. Saturation / near-saturation percentage
    5. Zero / near-zero percentage
    6. Between-class variance
    7. Within-class variance
    8. Between/within variance ratio
    9. Edge-band analysis
   10. Stable-band identification
   11. Discriminative-band identification
   12. QC plots

IMPORTANT
---------
This script does NOT:
    - modify raw MAT files
    - perform PCA
    - train a classifier
    - create a train/test split
    - assign artificial labels

The existing Tomato class labels are the dataset classes:

    Alternaria_alternata
    Alternaria_solani
    Botrytis_cinerea
    Fusarium_oxysporum
    Healthy

======================================================================
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

warnings.filterwarnings("ignore")


# =====================================================================
# CONFIGURATION
# =====================================================================

PROJECT_ROOT = Path(
    r"D:\Projects\AI-Food-Quality-Platform"
)

INPUT_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "raw"
    / "hyperspectral"
    / "tomato_hsi"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "tomato_spectral_band_qc"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

EXPECTED_BANDS = 550
SAMPLES_PER_CLASS = 1000

# Number of pixels sampled from each spatial cube
# for QC. This keeps memory usage reasonable.
RANDOM_STATE = 42

# Number of bands considered at each spectral edge.
EDGE_BANDS = 25

# Saturation threshold for 12-bit style data.
# 4095 is the maximum commonly used in this dataset.
SATURATION_THRESHOLD_12BIT = 4095

# Relative threshold for values considered close to saturation.
NEAR_SATURATION_FRACTION = 0.98

# Very small values
NEAR_ZERO_THRESHOLD = 1e-8


# =====================================================================
# CLASS DETECTION
# =====================================================================

CLASS_NAMES = [
    "Alternaria_alternata",
    "Alternaria_solani",
    "Botrytis_cinerea",
    "Fusarium_oxysporum",
    "Healthy",
]


# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================

def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)
    print()


def load_tomato_cube(mat_path):
    """
    Load the only 3-D MATLAB variable from a Tomato MAT file.
    """

    data = loadmat(mat_path)

    candidates = []

    for name, value in data.items():

        if name.startswith("__"):
            continue

        if isinstance(value, np.ndarray) and value.ndim == 3:
            candidates.append((name, value))

    if len(candidates) == 0:
        raise ValueError(
            f"No 3-D hyperspectral cube found in {mat_path.name}"
        )

    if len(candidates) > 1:
        raise ValueError(
            f"Multiple 3-D variables found in {mat_path.name}: "
            f"{[x[0] for x in candidates]}"
        )

    variable_name, cube = candidates[0]

    if cube.shape != (140, 280, EXPECTED_BANDS):
        raise ValueError(
            f"Unexpected cube shape in {mat_path.name}: "
            f"{cube.shape}. Expected "
            f"(140, 280, {EXPECTED_BANDS})."
        )

    return variable_name, cube.astype(np.float64)


def detect_class(filename):
    """
    Determine Tomato class from filename.
    """

    stem = filename.stem

    for class_name in CLASS_NAMES:

        if stem.startswith(class_name):
            return class_name

    raise ValueError(
        f"Could not determine class from filename: {filename.name}"
    )


def sample_spectra(cube, n_samples, rng):
    """
    Randomly sample spatial pixels while retaining
    the complete 550-band spectrum.
    """

    height, width, bands = cube.shape

    total_pixels = height * width

    n_samples = min(
        n_samples,
        total_pixels
    )

    indices = rng.choice(
        total_pixels,
        size=n_samples,
        replace=False
    )

    spectra = cube.reshape(
        total_pixels,
        bands
    )[indices]

    return spectra


# =====================================================================
# MAIN
# =====================================================================

def main():

    print_header(
        "TOMATO HSI SPECTRAL BAND QUALITY CONTROL"
    )

    print("Input directory:")
    print(INPUT_DIR)

    print()
    print("Output directory:")
    print(OUTPUT_DIR)

    # ---------------------------------------------------------------
    # FILE DISCOVERY
    # ---------------------------------------------------------------

    print_header("MAT FILE DISCOVERY")

    mat_files = sorted(
        INPUT_DIR.glob("*.mat")
    )

    if len(mat_files) != 10:

        raise ValueError(
            f"Expected 10 MAT files, found {len(mat_files)}."
        )

    for file in mat_files:
        print(file.name)

    print()
    print(f"Total MAT files: {len(mat_files)}")

    # ---------------------------------------------------------------
    # INITIALIZATION
    # ---------------------------------------------------------------

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    class_spectra = {
        class_name: []
        for class_name in CLASS_NAMES
    }

    file_statistics = []

    # ---------------------------------------------------------------
    # LOAD AND SAMPLE
    # ---------------------------------------------------------------

    print_header(
        "LOADING TOMATO HSI CUBES"
    )

    for mat_file in mat_files:

        class_name = detect_class(
            mat_file
        )

        print(
            f"Loading: {mat_file.name}"
        )

        variable_name, cube = load_tomato_cube(
            mat_file
        )

        print(
            f"  Variable : {variable_name}"
        )

        print(
            f"  Shape    : {cube.shape}"
        )

        print(
            f"  Min      : {np.nanmin(cube):.8f}"
        )

        print(
            f"  Max      : {np.nanmax(cube):.8f}"
        )

        print(
            f"  Mean     : {np.nanmean(cube):.8f}"
        )

        print(
            f"  NaN      : {np.isnan(cube).sum()}"
        )

        print(
            f"  Inf      : {np.isinf(cube).sum()}"
        )

        spectra = sample_spectra(
            cube,
            SAMPLES_PER_CLASS,
            rng
        )

        class_spectra[class_name].append(
            spectra
        )

        file_statistics.append(
            {
                "file": mat_file.name,
                "class": class_name,
                "height": cube.shape[0],
                "width": cube.shape[1],
                "bands": cube.shape[2],
                "min": np.nanmin(cube),
                "max": np.nanmax(cube),
                "mean": np.nanmean(cube),
                "std": np.nanstd(cube),
                "nan_count": np.isnan(cube).sum(),
                "inf_count": np.isinf(cube).sum(),
            }
        )

        del cube

    # ---------------------------------------------------------------
    # COMBINE CLASS DATA
    # ---------------------------------------------------------------

    print_header(
        "COMBINING SAMPLED SPECTRA"
    )

    for class_name in CLASS_NAMES:

        class_spectra[class_name] = np.vstack(
            class_spectra[class_name]
        )

        print(
            f"{class_name:25s}: "
            f"{class_spectra[class_name].shape}"
        )

    # ---------------------------------------------------------------
    # CLASS-WISE BAND STATISTICS
    # ---------------------------------------------------------------

    print_header(
        "CALCULATING CLASS-WISE BAND STATISTICS"
    )

    class_band_stats = []

    for class_name in CLASS_NAMES:

        spectra = class_spectra[class_name]

        means = np.mean(
            spectra,
            axis=0
        )

        stds = np.std(
            spectra,
            axis=0
        )

        mins = np.min(
            spectra,
            axis=0
        )

        maxs = np.max(
            spectra,
            axis=0
        )

        class_size = spectra.shape[0]

        saturation_pct = (
            np.sum(
                spectra >= SATURATION_THRESHOLD_12BIT,
                axis=0
            )
            / class_size
            * 100.0
        )

        near_saturation_threshold = (
            SATURATION_THRESHOLD_12BIT
            * NEAR_SATURATION_FRACTION
        )

        near_saturation_pct = (
            np.sum(
                spectra >= near_saturation_threshold,
                axis=0
            )
            / class_size
            * 100.0
        )

        near_zero_pct = (
            np.sum(
                np.abs(spectra)
                <= NEAR_ZERO_THRESHOLD,
                axis=0
            )
            / class_size
            * 100.0
        )

        for band_idx in range(EXPECTED_BANDS):

            mean_value = means[band_idx]

            std_value = stds[band_idx]

            if abs(mean_value) > 1e-12:

                coefficient_variation = (
                    std_value
                    / abs(mean_value)
                )

            else:

                coefficient_variation = np.nan

            class_band_stats.append(
                {
                    "class": class_name,
                    "band": band_idx + 1,
                    "mean": mean_value,
                    "std": std_value,
                    "min": mins[band_idx],
                    "max": maxs[band_idx],
                    "coefficient_variation":
                        coefficient_variation,
                    "saturation_percentage":
                        saturation_pct[band_idx],
                    "near_saturation_percentage":
                        near_saturation_pct[band_idx],
                    "near_zero_percentage":
                        near_zero_pct[band_idx],
                }
            )

    class_band_df = pd.DataFrame(
        class_band_stats
    )

    # ---------------------------------------------------------------
    # GLOBAL BAND STATISTICS
    # ---------------------------------------------------------------

    print_header(
        "CALCULATING GLOBAL BAND STATISTICS"
    )

    global_rows = []

    for band_idx in range(
        EXPECTED_BANDS
    ):

        values = np.concatenate(
            [
                class_spectra[class_name][
                    :, band_idx
                ]
                for class_name in CLASS_NAMES
            ]
        )

        mean_value = np.mean(values)
        std_value = np.std(values)

        if abs(mean_value) > 1e-12:

            cv = (
                std_value
                / abs(mean_value)
            )

        else:

            cv = np.nan

        saturation_pct = (
            np.mean(
                values >=
                SATURATION_THRESHOLD_12BIT
            )
            * 100.0
        )

        near_saturation_pct = (
            np.mean(
                values >=
                SATURATION_THRESHOLD_12BIT
                * NEAR_SATURATION_FRACTION
            )
            * 100.0
        )

        near_zero_pct = (
            np.mean(
                np.abs(values)
                <= NEAR_ZERO_THRESHOLD
            )
            * 100.0
        )

        global_rows.append(
            {
                "band": band_idx + 1,
                "mean": mean_value,
                "std": std_value,
                "coefficient_variation": cv,
                "min": np.min(values),
                "max": np.max(values),
                "saturation_percentage":
                    saturation_pct,
                "near_saturation_percentage":
                    near_saturation_pct,
                "near_zero_percentage":
                    near_zero_pct,
            }
        )

    global_df = pd.DataFrame(
        global_rows
    )

    # ---------------------------------------------------------------
    # BETWEEN-CLASS VARIANCE
    # ---------------------------------------------------------------

    print_header(
        "BETWEEN-CLASS / WITHIN-CLASS VARIANCE ANALYSIS"
    )

    between_class = []
    within_class = []

    for band_idx in range(
        EXPECTED_BANDS
    ):

        class_means = []

        class_variances = []

        class_counts = []

        for class_name in CLASS_NAMES:

            values = class_spectra[
                class_name
            ][:, band_idx]

            class_means.append(
                np.mean(values)
            )

            class_variances.append(
                np.var(values)
            )

            class_counts.append(
                len(values)
            )

        class_means = np.array(
            class_means
        )

        class_variances = np.array(
            class_variances
        )

        class_counts = np.array(
            class_counts
        )

        overall_mean = np.average(
            class_means,
            weights=class_counts
        )

        between = np.average(
            (
                class_means
                - overall_mean
            ) ** 2,
            weights=class_counts
        )

        within = np.average(
            class_variances,
            weights=class_counts
        )

        between_class.append(
            between
        )

        within_class.append(
            within
        )

    between_class = np.array(
        between_class
    )

    within_class = np.array(
        within_class
    )

    variance_ratio = (
        between_class
        /
        np.maximum(
            within_class,
            1e-12
        )
    )

    variance_df = pd.DataFrame(
        {
            "band":
                np.arange(
                    1,
                    EXPECTED_BANDS + 1
                ),
            "between_class_variance":
                between_class,
            "within_class_variance":
                within_class,
            "between_within_ratio":
                variance_ratio,
        }
    )

    # ---------------------------------------------------------------
    # EDGE BAND ANALYSIS
    # ---------------------------------------------------------------

    print_header(
        "EDGE BAND ANALYSIS"
    )

    first_edge = global_df[
        global_df["band"]
        <= EDGE_BANDS
    ].copy()

    last_edge = global_df[
        global_df["band"]
        > EXPECTED_BANDS - EDGE_BANDS
    ].copy()

    edge_df = pd.concat(
        [
            first_edge.assign(
                edge="Lower"
            ),
            last_edge.assign(
                edge="Upper"
            ),
        ],
        ignore_index=True
    )

    print(
        f"Lower edge bands: "
        f"1–{EDGE_BANDS}"
    )

    print(
        f"Upper edge bands: "
        f"{EXPECTED_BANDS - EDGE_BANDS + 1}"
        f"–{EXPECTED_BANDS}"
    )

    # ---------------------------------------------------------------
    # BAND STABILITY SCORE
    # ---------------------------------------------------------------

    print_header(
        "CALCULATING BAND STABILITY"
    )

    # Normalize quantities for comparison.
    #
    # High saturation / near-zero percentages reduce stability.
    # High within-class variation also reduces stability.
    #

    max_within = max(
        variance_df[
            "within_class_variance"
        ].max(),
        1e-12
    )

    within_norm = (
        variance_df[
            "within_class_variance"
        ]
        / max_within
    )

    saturation_penalty = (
        global_df[
            "near_saturation_percentage"
        ]
        / 100.0
    )

    zero_penalty = (
        global_df[
            "near_zero_percentage"
        ]
        / 100.0
    )

    stability_score = (
        100
        * (
            1
            - 0.5 * within_norm
            - 0.25 * saturation_penalty
            - 0.25 * zero_penalty
        )
    )

    stability_score = np.clip(
        stability_score,
        0,
        100
    )

    stability_df = pd.DataFrame(
        {
            "band":
                np.arange(
                    1,
                    EXPECTED_BANDS + 1
                ),
            "stability_score":
                stability_score,
        }
    )

    # ---------------------------------------------------------------
    # FINAL QC TABLE
    # ---------------------------------------------------------------

    qc_df = (
        global_df
        .merge(
            variance_df,
            on="band"
        )
        .merge(
            stability_df,
            on="band"
        )
    )

    # ---------------------------------------------------------------
    # DISCRIMINATIVE BANDS
    # ---------------------------------------------------------------

    print_header(
        "TOP DISCRIMINATIVE BANDS"
    )

    discriminative_df = (
        qc_df
        .sort_values(
            "between_within_ratio",
            ascending=False
        )
        .reset_index(drop=True)
    )

    discriminative_df[
        [
            "band",
            "between_class_variance",
            "within_class_variance",
            "between_within_ratio",
            "stability_score",
        ]
    ].head(30).to_string(
        index=False
    )

    print()

    # ---------------------------------------------------------------
    # STABLE + DISCRIMINATIVE BANDS
    # ---------------------------------------------------------------

    print_header(
        "STABLE DISCRIMINATIVE BANDS"
    )

    stable_discriminative = (
        qc_df[
            qc_df[
                "stability_score"
            ] >= 70
        ]
        .sort_values(
            "between_within_ratio",
            ascending=False
        )
        .reset_index(drop=True)
    )

    print(
        f"Stable bands (score >= 70): "
        f"{len(stable_discriminative)}"
    )

    print()

    print(
        stable_discriminative[
            [
                "band",
                "between_within_ratio",
                "stability_score",
                "near_saturation_percentage",
                "near_zero_percentage",
            ]
        ].head(30).to_string(
            index=False
        )
    )

    # ---------------------------------------------------------------
    # EDGE COMPARISON
    # ---------------------------------------------------------------

    print_header(
        "EDGE VS CENTRAL BAND COMPARISON"
    )

    lower_edge_scores = qc_df[
        qc_df["band"] <= EDGE_BANDS
    ]

    upper_edge_scores = qc_df[
        qc_df["band"]
        > EXPECTED_BANDS - EDGE_BANDS
    ]

    central_scores = qc_df[
        (qc_df["band"] > EDGE_BANDS)
        &
        (
            qc_df["band"]
            <= EXPECTED_BANDS - EDGE_BANDS
        )
    ]

    print(
        "Lower edge mean stability:",
        f"{lower_edge_scores['stability_score'].mean():.4f}"
    )

    print(
        "Upper edge mean stability:",
        f"{upper_edge_scores['stability_score'].mean():.4f}"
    )

    print(
        "Central region mean stability:",
        f"{central_scores['stability_score'].mean():.4f}"
    )

    print()

    print(
        "Lower edge mean saturation:",
        f"{lower_edge_scores['near_saturation_percentage'].mean():.4f}%"
    )

    print(
        "Upper edge mean saturation:",
        f"{upper_edge_scores['near_saturation_percentage'].mean():.4f}%"
    )

    print(
        "Central mean saturation:",
        f"{central_scores['near_saturation_percentage'].mean():.4f}%"
    )

    # ---------------------------------------------------------------
    # SAVE CSV FILES
    # ---------------------------------------------------------------

    print_header(
        "SAVING QC RESULTS"
    )

    outputs = {

        "tomato_qc_file_statistics.csv":
            pd.DataFrame(
                file_statistics
            ),

        "tomato_qc_class_band_statistics.csv":
            class_band_df,

        "tomato_qc_global_band_statistics.csv":
            global_df,

        "tomato_qc_band_variance_analysis.csv":
            variance_df,

        "tomato_qc_edge_band_analysis.csv":
            edge_df,

        "tomato_qc_band_stability.csv":
            stability_df,

        "tomato_qc_complete_band_analysis.csv":
            qc_df,

        "tomato_qc_discriminative_bands.csv":
            discriminative_df,

        "tomato_qc_stable_discriminative_bands.csv":
            stable_discriminative,
    }

    for filename, dataframe in outputs.items():

        path = (
            OUTPUT_DIR
            / filename
        )

        dataframe.to_csv(
            path,
            index=False
        )

        print(
            f"Saved: {path}"
        )

    # ---------------------------------------------------------------
    # PLOT 1 — GLOBAL MEAN
    # ---------------------------------------------------------------

    print_header(
        "CREATING GLOBAL MEAN SPECTRAL PLOT"
    )

    plt.figure(
        figsize=(14, 6)
    )

    plt.plot(
        global_df["band"],
        global_df["mean"]
    )

    plt.xlabel(
        "Spectral Band"
    )

    plt.ylabel(
        "Mean Raw Intensity"
    )

    plt.title(
        "Tomato HSI Global Mean Spectral Profile"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    path = (
        OUTPUT_DIR
        / "tomato_qc_global_mean_spectrum.png"
    )

    plt.savefig(
        path,
        dpi=200
    )

    plt.close()

    print(
        f"Saved: {path}"
    )

    # ---------------------------------------------------------------
    # PLOT 2 — WITHIN / BETWEEN VARIANCE
    # ---------------------------------------------------------------

    print_header(
        "CREATING VARIANCE PLOT"
    )

    plt.figure(
        figsize=(14, 6)
    )

    plt.plot(
        variance_df["band"],
        variance_df[
            "between_class_variance"
        ],
        label="Between-class variance"
    )

    plt.plot(
        variance_df["band"],
        variance_df[
            "within_class_variance"
        ],
        label="Within-class variance"
    )

    plt.xlabel(
        "Spectral Band"
    )

    plt.ylabel(
        "Variance"
    )

    plt.title(
        "Tomato HSI Between-Class vs Within-Class Variance"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    path = (
        OUTPUT_DIR
        / "tomato_qc_between_within_variance.png"
    )

    plt.savefig(
        path,
        dpi=200
    )

    plt.close()

    print(
        f"Saved: {path}"
    )

    # ---------------------------------------------------------------
    # PLOT 3 — DISCRIMINATIVE RATIO
    # ---------------------------------------------------------------

    print_header(
        "CREATING DISCRIMINATIVE BAND PLOT"
    )

    plt.figure(
        figsize=(14, 6)
    )

    plt.plot(
        qc_df["band"],
        qc_df[
            "between_within_ratio"
        ]
    )

    plt.xlabel(
        "Spectral Band"
    )

    plt.ylabel(
        "Between / Within Variance Ratio"
    )

    plt.title(
        "Tomato HSI Band Discriminative Strength"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    path = (
        OUTPUT_DIR
        / "tomato_qc_discriminative_band_ratio.png"
    )

    plt.savefig(
        path,
        dpi=200
    )

    plt.close()

    print(
        f"Saved: {path}"
    )

    # ---------------------------------------------------------------
    # PLOT 4 — STABILITY
    # ---------------------------------------------------------------

    print_header(
        "CREATING BAND STABILITY PLOT"
    )

    plt.figure(
        figsize=(14, 6)
    )

    plt.plot(
        stability_df["band"],
        stability_df[
            "stability_score"
        ]
    )

    plt.axhline(
        70,
        linestyle="--",
        label="Stability threshold"
    )

    plt.xlabel(
        "Spectral Band"
    )

    plt.ylabel(
        "Stability Score"
    )

    plt.title(
        "Tomato HSI Spectral Band Stability"
    )

    plt.ylim(
        0,
        100
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    path = (
        OUTPUT_DIR
        / "tomato_qc_band_stability.png"
    )

    plt.savefig(
        path,
        dpi=200
    )

    plt.close()

    print(
        f"Saved: {path}"
    )

    # ---------------------------------------------------------------
    # PLOT 5 — CLASS MEAN SIGNATURES
    # ---------------------------------------------------------------

    print_header(
        "CREATING CLASS-WISE RAW SIGNATURE PLOT"
    )

    plt.figure(
        figsize=(14, 7)
    )

    for class_name in CLASS_NAMES:

        subset = class_band_df[
            class_band_df["class"]
            == class_name
        ]

        plt.plot(
            subset["band"],
            subset["mean"],
            label=class_name
        )

    plt.xlabel(
        "Spectral Band"
    )

    plt.ylabel(
        "Mean Raw Intensity"
    )

    plt.title(
        "Tomato HSI Class-Wise Mean Raw Spectral Signatures"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    path = (
        OUTPUT_DIR
        / "tomato_qc_classwise_raw_signatures.png"
    )

    plt.savefig(
        path,
        dpi=200
    )

    plt.close()

    print(
        f"Saved: {path}"
    )

    # ---------------------------------------------------------------
    # FINAL SUMMARY
    # ---------------------------------------------------------------

    print_header(
        "TOMATO SPECTRAL BAND QC COMPLETED"
    )

    print(
        f"MAT files analyzed        : {len(mat_files)}"
    )

    print(
        f"Classes                   : {len(CLASS_NAMES)}"
    )

    print(
        f"Spectral bands            : {EXPECTED_BANDS}"
    )

    print(
        f"Sampled spectra/class     : {SAMPLES_PER_CLASS * 2}"
    )

    print(
        f"Total sampled spectra     : "
        f"{sum(len(v) for v in class_spectra.values())}"
    )

    print()

    print(
        "QC outputs:"
    )

    print(
        OUTPUT_DIR
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "Band numbers are reported as band indices."
    )

    print(
        "They are NOT converted to wavelengths because"
    )

    print(
        "the MATLAB files do not provide wavelength calibration."
    )

    print()

    print(
        "No raw MAT file was modified."
    )

    print(
        "No PCA was performed."
    )

    print(
        "No classifier was trained."
    )

    print(
        "No train/test split was created."
    )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()