"""
test_tomato_data_inspection.py

Detailed inspection of the Tomato HSI dataset.

Purpose
-------
This script investigates the raw MATLAB hyperspectral files before
preprocessing, PCA, and classification.

It specifically checks:

1. MATLAB variables present in each file
2. Hyperspectral cube shape
3. Data type
4. Minimum / maximum values
5. Mean / median
6. Percentiles
7. Zero percentage
8. Unique-value information
9. NaN / infinite values
10. Per-band statistics
11. Whether files appear to use different numerical scales
12. Spectral signatures before any normalization

IMPORTANT
---------
This script DOES NOT modify the raw data.

It does NOT:
    - normalize
    - divide by 4095
    - apply PCA
    - smooth spectra
    - mean-center data

The purpose is to understand the raw acquisition first.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "raw"
    / "hyperspectral"
    / "tomato_hsi"
)

OUTPUT_GRAPH_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "graphs"
    / "tomato_spectral"
)

OUTPUT_DATA_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "hyperspectral"
    / "tomato_spectral"
)


OUTPUT_GRAPH_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DATASET CONFIGURATION
# ============================================================

EXPECTED_HEIGHT = 140
EXPECTED_WIDTH = 280
EXPECTED_BANDS = 550

WAVELENGTH_START = 400
WAVELENGTH_END = 950


# ============================================================
# CLASS IDENTIFICATION
# ============================================================

def identify_class(filename):
    """
    Identify the tomato disease/health class from filename.
    """

    name = filename.lower()

    if "alternaria_alternata" in name:
        return "Alternaria alternata"

    if "alternaria_solani" in name:
        return "Alternaria solani"

    if "botrytis_cinerea" in name:
        return "Botrytis cinerea"

    if "fusarium_oxysporum" in name:
        return "Fusarium oxysporum"

    if "healthy" in name:
        return "Healthy"

    return "Unknown"


# ============================================================
# SEPARATOR
# ============================================================

def separator(title=None):

    print()
    print("=" * 75)

    if title:
        print(title)
        print("=" * 75)


# ============================================================
# WAVELENGTH AXIS
# ============================================================

def get_wavelengths():

    return np.linspace(
        WAVELENGTH_START,
        WAVELENGTH_END,
        EXPECTED_BANDS
    )


# ============================================================
# FIND HYPERSPECTRAL VARIABLE
# ============================================================

def find_hsi_variables(mat_data):
    """
    Find all 3D numerical arrays inside a MATLAB file.
    """

    candidates = []

    for name, value in mat_data.items():

        if name.startswith("__"):
            continue

        if not isinstance(value, np.ndarray):
            continue

        if value.ndim != 3:
            continue

        if not np.issubdtype(
            value.dtype,
            np.number
        ):
            continue

        candidates.append(
            (name, value)
        )

    return candidates


# ============================================================
# LOAD FILE
# ============================================================

def load_file(file_path):

    print()
    print(
        f"FILE: {file_path.name}"
    )

    mat_data = loadmat(
        file_path,
        squeeze_me=False
    )

    print()
    print("MATLAB VARIABLES")
    print("-" * 75)

    variable_rows = []

    for name, value in mat_data.items():

        if name.startswith("__"):
            continue

        if isinstance(value, np.ndarray):

            print(
                f"{name:30s}"
                f" shape={str(value.shape):20s}"
                f" dtype={value.dtype}"
            )

            variable_rows.append({
                "file": file_path.name,
                "variable": name,
                "shape": str(value.shape),
                "dimensions": value.ndim,
                "dtype": str(value.dtype),
                "size": value.size,
            })

        else:

            print(
                f"{name:30s}"
                f" type={type(value).__name__}"
            )

    hsi_variables = find_hsi_variables(
        mat_data
    )

    if not hsi_variables:

        raise ValueError(
            "No 3D numerical hyperspectral "
            "cube found."
        )

    print()
    print("3D HYPERSPECTRAL VARIABLES")
    print("-" * 75)

    for name, value in hsi_variables:

        print(
            f"{name:30s}"
            f" Shape: {value.shape}"
            f" Dtype: {value.dtype}"
        )

    # Prefer a cube with 550 bands
    selected_name = None
    selected_cube = None

    for name, value in hsi_variables:

        if value.shape[-1] == EXPECTED_BANDS:

            selected_name = name
            selected_cube = value
            break

    if selected_cube is None:

        selected_name, selected_cube = (
            hsi_variables[0]
        )

    cube = np.asarray(
        selected_cube,
        dtype=np.float32
    )

    print()
    print("SELECTED HYPERSPECTRAL CUBE")
    print("-" * 75)

    print(
        f"Variable : {selected_name}"
    )

    print(
        f"Shape    : {cube.shape}"
    )

    print(
        f"Dtype    : {cube.dtype}"
    )

    return cube, variable_rows, selected_name


# ============================================================
# RAW STATISTICS
# ============================================================

def calculate_statistics(
    cube,
    file_name,
    class_name
):
    """
    Calculate detailed statistics without modifying
    the cube.
    """

    flat = cube.reshape(-1)

    finite_mask = np.isfinite(flat)

    finite_values = flat[
        finite_mask
    ]

    if finite_values.size == 0:

        raise ValueError(
            "No finite values found."
        )

    nan_count = np.isnan(cube).sum()

    inf_count = np.isinf(cube).sum()

    zero_count = np.count_nonzero(
        cube == 0
    )

    total_count = cube.size

    zero_percentage = (
        zero_count
        / total_count
        * 100
    )

    percentiles = np.percentile(
        finite_values,
        [
            0,
            1,
            5,
            25,
            50,
            75,
            95,
            99,
            99.9,
            100,
        ]
    )

    # Unique count can be expensive on huge arrays,
    # but these tomato cubes are only 140*280*550.
    unique_count = np.unique(
        finite_values
    ).size

    statistics = {

        "file": file_name,

        "class": class_name,

        "height": cube.shape[0],

        "width": cube.shape[1],

        "bands": cube.shape[2],

        "dtype": str(cube.dtype),

        "minimum": float(
            percentiles[0]
        ),

        "percentile_1": float(
            percentiles[1]
        ),

        "percentile_5": float(
            percentiles[2]
        ),

        "percentile_25": float(
            percentiles[3]
        ),

        "median": float(
            percentiles[4]
        ),

        "percentile_75": float(
            percentiles[5]
        ),

        "percentile_95": float(
            percentiles[6]
        ),

        "percentile_99": float(
            percentiles[7]
        ),

        "percentile_99_9": float(
            percentiles[8]
        ),

        "maximum": float(
            percentiles[9]
        ),

        "mean": float(
            np.mean(finite_values)
        ),

        "std": float(
            np.std(finite_values)
        ),

        "nan_count": int(
            nan_count
        ),

        "infinite_count": int(
            inf_count
        ),

        "zero_count": int(
            zero_count
        ),

        "zero_percentage": float(
            zero_percentage
        ),

        "unique_values": int(
            unique_count
        ),
    }

    return statistics


# ============================================================
# PER-BAND STATISTICS
# ============================================================

def calculate_band_statistics(
    cube,
    wavelengths
):
    """
    Calculate mean, standard deviation,
    minimum and maximum for every wavelength.
    """

    rows = []

    for band in range(
        cube.shape[2]
    ):

        values = cube[
            :, :, band
        ].reshape(-1)

        valid = values[
            np.isfinite(values)
        ]

        if valid.size == 0:

            rows.append({
                "band": band + 1,
                "wavelength_nm": wavelengths[band],
                "mean": np.nan,
                "std": np.nan,
                "minimum": np.nan,
                "maximum": np.nan,
            })

            continue

        rows.append({
            "band": band + 1,

            "wavelength_nm":
                wavelengths[band],

            "mean":
                np.mean(valid),

            "std":
                np.std(valid),

            "minimum":
                np.min(valid),

            "maximum":
                np.max(valid),
        })

    return pd.DataFrame(rows)


# ============================================================
# CLASSIFY SCALE
# ============================================================

def determine_scale_category(
    minimum,
    maximum,
    mean
):
    """
    Give a descriptive scale category.

    IMPORTANT:
    This is NOT a scientific conversion.
    It is only an inspection label.
    """

    if (
        maximum <= 1.5
        and minimum >= -0.1
    ):

        return "0-1-like"

    if (
        maximum > 1000
        and maximum <= 4095
    ):

        return "4095-like"

    if maximum > 4095:

        return "Above-4095"

    return "Other"


# ============================================================
# PLOT RAW CLASS SIGNATURE
# ============================================================

def calculate_mean_signature(
    cube
):
    """
    Calculate mean spectrum across all spatial pixels.
    """

    signatures = []

    for band in range(
        cube.shape[2]
    ):

        values = cube[
            :, :, band
        ].reshape(-1)

        valid = values[
            np.isfinite(values)
        ]

        if valid.size == 0:

            signatures.append(
                np.nan
            )

        else:

            signatures.append(
                np.mean(valid)
            )

    return np.asarray(
        signatures,
        dtype=np.float64
    )


# ============================================================
# SAVE INDIVIDUAL SIGNATURE
# ============================================================

def save_signature(
    file_name,
    wavelengths,
    signature
):

    output_name = (
        Path(file_name).stem
        + "_raw_signature.csv"
    )

    output_path = (
        OUTPUT_DATA_DIR
        / output_name
    )

    dataframe = pd.DataFrame({

        "band":
            np.arange(
                1,
                len(wavelengths) + 1
            ),

        "wavelength_nm":
            wavelengths,

        "mean_raw_response":
            signature,
    })

    dataframe.to_csv(
        output_path,
        index=False
    )

    print(
        f"Saved signature CSV: "
        f"{output_path}"
    )


# ============================================================
# PLOT RAW SIGNATURES
# ============================================================

def plot_raw_signatures(
    signature_results,
    wavelengths
):
    """
    Plot all raw signatures.

    No normalization is applied.
    """

    plt.figure(
        figsize=(14, 8)
    )

    for result in signature_results:

        plt.plot(
            wavelengths,
            result["signature"],
            linewidth=1.5,
            label=(
                result["class"]
                + " - "
                + result["file"]
            )
        )

    plt.xlabel(
        "Wavelength (nm)"
    )

    plt.ylabel(
        "Raw Spectral Response"
    )

    plt.title(
        "Tomato HSI - Raw Spectral Signatures"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend(
        fontsize=7
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_GRAPH_DIR
        / "tomato_inspection_raw_signatures.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(
        f"Saved graph: {output_path}"
    )


# ============================================================
# PLOT SCALE DISTRIBUTION
# ============================================================

def plot_file_scale_comparison(
    statistics
):
    """
    Compare the mean and maximum numerical
    values across files.

    This helps visualize the scale discrepancy.
    """

    dataframe = pd.DataFrame(
        statistics
    )

    x = np.arange(
        len(dataframe)
    )

    plt.figure(
        figsize=(14, 7)
    )

    plt.bar(
        x,
        dataframe["maximum"]
    )

    plt.xticks(
        x,
        dataframe["file"],
        rotation=75,
        ha="right",
        fontsize=8
    )

    plt.ylabel(
        "Maximum Raw Value"
    )

    plt.title(
        "Tomato HSI - Raw Numerical Scale by File"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_GRAPH_DIR
        / "tomato_raw_scale_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(
        f"Saved graph: {output_path}"
    )


# ============================================================
# PLOT BAND MEAN FOR EACH FILE
# ============================================================

def plot_band_means(
    band_results,
    wavelengths
):
    """
    Plot mean response across wavelength
    for each individual file.
    """

    plt.figure(
        figsize=(14, 8)
    )

    for result in band_results:

        plt.plot(
            wavelengths,
            result["mean"],
            linewidth=1.5,
            label=(
                result["class"]
                + " - "
                + result["file"]
            )
        )

    plt.xlabel(
        "Wavelength (nm)"
    )

    plt.ylabel(
        "Mean Raw Response"
    )

    plt.title(
        "Tomato HSI - Per-Band Mean Response"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend(
        fontsize=7
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_GRAPH_DIR
        / "tomato_per_band_mean.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(
        f"Saved graph: {output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    separator(
        "TOMATO HSI RAW DATA INSPECTION"
    )

    print(
        f"Dataset directory:\n"
        f"{DATASET_DIR}"
    )

    print(
        f"Expected cube shape: "
        f"({EXPECTED_HEIGHT}, "
        f"{EXPECTED_WIDTH}, "
        f"{EXPECTED_BANDS})"
    )

    print(
        f"Expected spectral range: "
        f"{WAVELENGTH_START}-"
        f"{WAVELENGTH_END} nm"
    )

    # --------------------------------------------------------
    # Check directory
    # --------------------------------------------------------

    if not DATASET_DIR.exists():

        raise FileNotFoundError(
            f"Dataset directory does not exist:\n"
            f"{DATASET_DIR}"
        )

    # --------------------------------------------------------
    # Find MAT files
    # --------------------------------------------------------

    mat_files = sorted(
        DATASET_DIR.glob("*.mat")
    )

    if not mat_files:

        raise FileNotFoundError(
            "No .mat files found in "
            "tomato_hsi."
        )

    print()
    print(
        f"MAT files found: "
        f"{len(mat_files)}"
    )

    # --------------------------------------------------------
    # Wavelengths
    # --------------------------------------------------------

    wavelengths = get_wavelengths()

    print(
        f"Wavelength points: "
        f"{len(wavelengths)}"
    )

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    all_statistics = []

    all_variables = []

    signature_results = []

    band_results = []

    # --------------------------------------------------------
    # Process every file
    # --------------------------------------------------------

    for file_path in mat_files:

        separator(
            f"INSPECTING {file_path.name}"
        )

        class_name = identify_class(
            file_path.name
        )

        print(
            f"Class: {class_name}"
        )

        try:

            cube, variable_rows, selected_variable = (
                load_file(file_path)
            )

            # --------------------------------------------
            # Shape validation
            # --------------------------------------------

            print()
            print(
                "DIMENSION CHECK"
            )
            print("-" * 75)

            print(
                f"Height : {cube.shape[0]}"
            )

            print(
                f"Width  : {cube.shape[1]}"
            )

            print(
                f"Bands  : {cube.shape[2]}"
            )

            if cube.shape == (
                EXPECTED_HEIGHT,
                EXPECTED_WIDTH,
                EXPECTED_BANDS
            ):

                print(
                    "Shape status: CORRECT"
                )

            else:

                print(
                    "Shape status: DIFFERENT "
                    "FROM EXPECTED"
                )

            # --------------------------------------------
            # Raw statistics
            # --------------------------------------------

            statistics = (
                calculate_statistics(
                    cube,
                    file_path.name,
                    class_name
                )
            )

            scale_category = (
                determine_scale_category(
                    statistics["minimum"],
                    statistics["maximum"],
                    statistics["mean"]
                )
            )

            statistics[
                "selected_variable"
            ] = selected_variable

            statistics[
                "scale_category"
            ] = scale_category

            all_statistics.append(
                statistics
            )

            all_variables.extend(
                variable_rows
            )

            # --------------------------------------------
            # Print statistics
            # --------------------------------------------

            print()
            print(
                "RAW VALUE STATISTICS"
            )
            print("-" * 75)

            print(
                f"Minimum       : "
                f"{statistics['minimum']:.8f}"
            )

            print(
                f"1st percentile: "
                f"{statistics['percentile_1']:.8f}"
            )

            print(
                f"5th percentile: "
                f"{statistics['percentile_5']:.8f}"
            )

            print(
                f"25th percentile: "
                f"{statistics['percentile_25']:.8f}"
            )

            print(
                f"Median        : "
                f"{statistics['median']:.8f}"
            )

            print(
                f"75th percentile: "
                f"{statistics['percentile_75']:.8f}"
            )

            print(
                f"95th percentile: "
                f"{statistics['percentile_95']:.8f}"
            )

            print(
                f"99th percentile: "
                f"{statistics['percentile_99']:.8f}"
            )

            print(
                f"99.9 percentile: "
                f"{statistics['percentile_99_9']:.8f}"
            )

            print(
                f"Maximum       : "
                f"{statistics['maximum']:.8f}"
            )

            print(
                f"Mean          : "
                f"{statistics['mean']:.8f}"
            )

            print(
                f"Standard dev. : "
                f"{statistics['std']:.8f}"
            )

            print(
                f"Zero count    : "
                f"{statistics['zero_count']}"
            )

            print(
                f"Zero %        : "
                f"{statistics['zero_percentage']:.4f}%"
            )

            print(
                f"NaN count     : "
                f"{statistics['nan_count']}"
            )

            print(
                f"Inf count     : "
                f"{statistics['infinite_count']}"
            )

            print(
                f"Unique values : "
                f"{statistics['unique_values']}"
            )

            print(
                f"Scale category: "
                f"{scale_category}"
            )

            # --------------------------------------------
            # Mean spectral signature
            # --------------------------------------------

            signature = calculate_mean_signature(
                cube
            )

            signature_results.append({

                "file":
                    file_path.name,

                "class":
                    class_name,

                "signature":
                    signature,
            })

            save_signature(
                file_path.name,
                wavelengths,
                signature
            )

            # --------------------------------------------
            # Per-band statistics
            # --------------------------------------------

            band_dataframe = (
                calculate_band_statistics(
                    cube,
                    wavelengths
                )
            )

            band_output_name = (
                Path(file_path.name).stem
                + "_band_statistics.csv"
            )

            band_output_path = (
                OUTPUT_DATA_DIR
                / band_output_name
            )

            band_dataframe.to_csv(
                band_output_path,
                index=False
            )

            band_results.append({

                "file":
                    file_path.name,

                "class":
                    class_name,

                "mean":
                    band_dataframe["mean"].values,
            })

            print(
                f"Saved band statistics: "
                f"{band_output_path}"
            )

        except Exception as error:

            print()
            print(
                "ERROR"
            )

            print(
                f"{type(error).__name__}: "
                f"{error}"
            )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    separator(
        "DATASET SUMMARY"
    )

    statistics_dataframe = pd.DataFrame(
        all_statistics
    )

    summary_path = (
        OUTPUT_DATA_DIR
        / "tomato_raw_data_inspection.csv"
    )

    statistics_dataframe.to_csv(
        summary_path,
        index=False
    )

    print(
        f"Saved statistics: {summary_path}"
    )

    # --------------------------------------------------------
    # Save MATLAB variable information
    # --------------------------------------------------------

    variables_dataframe = pd.DataFrame(
        all_variables
    )

    variables_path = (
        OUTPUT_DATA_DIR
        / "tomato_matlab_variables.csv"
    )

    variables_dataframe.to_csv(
        variables_path,
        index=False
    )

    print(
        f"Saved MATLAB variable information: "
        f"{variables_path}"
    )

    # ========================================================
    # PRINT SCALE SUMMARY
    # ========================================================

    separator(
        "RAW SCALE SUMMARY"
    )

    if not statistics_dataframe.empty:

        columns = [
            "file",
            "class",
            "selected_variable",
            "minimum",
            "median",
            "mean",
            "maximum",
            "scale_category",
        ]

        print(
            statistics_dataframe[
                columns
            ].to_string(
                index=False
            )
        )

    # ========================================================
    # CLASS SUMMARY
    # ========================================================

    separator(
        "CLASS FILE COUNTS"
    )

    class_counts = (
        statistics_dataframe[
            "class"
        ].value_counts()
    )

    for class_name, count in (
        class_counts.items()
    ):

        print(
            f"{class_name:25s}: "
            f"{count} file(s)"
        )

    # ========================================================
    # GRAPHS
    # ========================================================

    separator(
        "GENERATING INSPECTION GRAPHS"
    )

    if signature_results:

        plot_raw_signatures(
            signature_results,
            wavelengths
        )

    if not statistics_dataframe.empty:

        plot_file_scale_comparison(
            all_statistics
        )

    if band_results:

        plot_band_means(
            band_results,
            wavelengths
        )

    # ========================================================
    # FINAL
    # ========================================================

    separator(
        "TOMATO RAW DATA INSPECTION COMPLETED"
    )

    print(
        f"Files inspected successfully: "
        f"{len(all_statistics)}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "No normalization was applied."
    )

    print(
        "No division by 4095 was applied."
    )

    print(
        "No PCA was applied."
    )

    print(
        "No smoothing was applied."
    )

    print(
        "No mean-centering was applied."
    )

    print()
    print(
        "The numerical scale difference will "
        "be investigated before preprocessing."
    )

    print()
    print(
        f"Graphs saved to:\n"
        f"{OUTPUT_GRAPH_DIR}"
    )

    print()
    print(
        f"CSV files saved to:\n"
        f"{OUTPUT_DATA_DIR}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
    