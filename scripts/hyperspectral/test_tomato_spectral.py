"""
test_tomato_spectral.py

Tomato fungal hyperspectral spectral-signature analysis.

Dataset:
    Javidan Tomato Disease HSI Dataset

Classes:
    Healthy
    Alternaria alternata
    Alternaria solani
    Botrytis cinerea
    Fusarium oxysporum

Spectral range:
    400 - 950 nm

Expected HSI cube:
    (140, 280, 550)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "raw"
    / "hyperspectral"
    / "tomato_hsi"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "graphs"
    / "tomato_spectral"
)

CSV_OUTPUT_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "hyperspectral"
    / "tomato_spectral"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Dataset documentation states 400-950 nm.
WAVELENGTH_START = 400
WAVELENGTH_END = 950

EXPECTED_BANDS = 550
EXPECTED_HEIGHT = 140
EXPECTED_WIDTH = 280


# ============================================================
# CLASS MAPPING
# ============================================================

CLASS_MAPPING = {
    "Alternaria_alternata": "Alternaria alternata",
    "Alternaria_solani": "Alternaria solani",
    "Botrytis_cinerea": "Botrytis cinerea",
    "Fusarium_oxysporum": "Fusarium oxysporum",
    "Healthy": "Healthy",
}


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def print_separator(title=None):
    print()
    print("=" * 70)

    if title:
        print(title)
        print("=" * 70)


def get_wavelengths():
    """
    Generate wavelength axis for the documented
    400-950 nm spectral range.
    """

    return np.linspace(
        WAVELENGTH_START,
        WAVELENGTH_END,
        EXPECTED_BANDS
    )


def identify_class(filename):
    """
    Determine disease/health class from filename.
    """

    filename_lower = filename.lower()

    if "alternaria_alternata" in filename_lower:
        return "Alternaria alternata"

    if "alternaria_solani" in filename_lower:
        return "Alternaria solani"

    if "botrytis_cinerea" in filename_lower:
        return "Botrytis cinerea"

    if "fusarium_oxysporum" in filename_lower:
        return "Fusarium oxysporum"

    if "healthy" in filename_lower:
        return "Healthy"

    return "Unknown"


def find_hsi_variable(mat_data):
    """
    Find the hyperspectral cube inside a MATLAB file.

    The dataset files contain variables such as:

        Day_one_1
        Day_one_2

    The function searches for a 3D numerical array.
    """

    candidates = []

    for name, value in mat_data.items():

        # Ignore MATLAB metadata
        if name.startswith("__"):
            continue

        if isinstance(value, np.ndarray):

            if value.ndim == 3:

                candidates.append(
                    (name, value)
                )

    if not candidates:
        raise ValueError(
            "No 3D hyperspectral cube found in MAT file."
        )

    # Prefer a cube with 550 spectral bands
    for name, value in candidates:

        if value.shape[-1] == EXPECTED_BANDS:
            return name, value

    # Otherwise return first 3D array
    return candidates[0]


def load_hsi_mat(file_path):
    """
    Load one MATLAB hyperspectral file.
    """

    print()
    print(f"Loading: {file_path.name}")

    mat_data = loadmat(
        file_path,
        squeeze_me=False
    )

    variable_name, cube = find_hsi_variable(
        mat_data
    )

    cube = np.asarray(
        cube,
        dtype=np.float32
    )

    print(
        f"Variable     : {variable_name}"
    )

    print(
        f"Shape        : {cube.shape}"
    )

    print(
        f"Dtype        : {cube.dtype}"
    )

    return cube


def validate_cube(cube, filename):
    """
    Validate hyperspectral cube dimensions.
    """

    if cube.ndim != 3:

        raise ValueError(
            f"{filename}: Expected 3D cube, "
            f"got {cube.ndim}D."
        )

    if cube.shape[-1] != EXPECTED_BANDS:

        raise ValueError(
            f"{filename}: Expected "
            f"{EXPECTED_BANDS} bands, "
            f"got {cube.shape[-1]}."
        )

    if (
        cube.shape[0] != EXPECTED_HEIGHT
        or cube.shape[1] != EXPECTED_WIDTH
    ):

        print(
            "WARNING: Spatial dimensions differ "
            f"from expected "
            f"({EXPECTED_HEIGHT}, {EXPECTED_WIDTH})."
        )

    return True


def calculate_signature(cube):
    """
    Calculate mean and standard deviation
    spectral signatures.

    Pixels containing NaN or infinite values
    are removed independently for every band.
    """

    bands = cube.shape[-1]

    mean_signature = np.zeros(
        bands,
        dtype=np.float64
    )

    std_signature = np.zeros(
        bands,
        dtype=np.float64
    )

    valid_counts = np.zeros(
        bands,
        dtype=np.int64
    )

    for band in range(bands):

        values = cube[:, :, band].reshape(-1)

        values = values[
            np.isfinite(values)
        ]

        valid_counts[band] = len(values)

        if len(values) == 0:

            mean_signature[band] = np.nan
            std_signature[band] = np.nan

        else:

            mean_signature[band] = np.mean(values)

            std_signature[band] = np.std(values)

    return (
        mean_signature,
        std_signature,
        valid_counts
    )


def normalize_for_visualization(signature):
    """
    Per-signature min-max normalization.

    IMPORTANT:
        This is ONLY for visual comparison.

    It must NOT be used as the final scientific
    calibration for model training.
    """

    minimum = np.nanmin(signature)
    maximum = np.nanmax(signature)

    if maximum == minimum:

        return np.zeros_like(
            signature
        )

    return (
        signature - minimum
    ) / (
        maximum - minimum
    )


# ============================================================
# PROCESS ONE FILE
# ============================================================

def process_file(file_path):
    """
    Process one tomato hyperspectral file.
    """

    class_name = identify_class(
        file_path.name
    )

    if class_name == "Unknown":

        print(
            f"WARNING: Could not identify class "
            f"for {file_path.name}"
        )

        return None

    cube = load_hsi_mat(
        file_path
    )

    validate_cube(
        cube,
        file_path.name
    )

    # --------------------------------------------------------
    # Raw statistics
    # --------------------------------------------------------

    finite_values = cube[
        np.isfinite(cube)
    ]

    nan_count = np.isnan(cube).sum()

    inf_count = np.isinf(cube).sum()

    print(
        f"Class        : {class_name}"
    )

    print(
        f"Minimum      : {np.min(finite_values):.6f}"
    )

    print(
        f"Maximum      : {np.max(finite_values):.6f}"
    )

    print(
        f"Mean         : {np.mean(finite_values):.6f}"
    )

    print(
        f"NaN values   : {nan_count}"
    )

    print(
        f"Inf values   : {inf_count}"
    )

    # --------------------------------------------------------
    # Spectral signature
    # --------------------------------------------------------

    mean_signature, std_signature, valid_counts = (
        calculate_signature(cube)
    )

    return {
        "class_name": class_name,
        "filename": file_path.name,
        "cube_shape": cube.shape,
        "mean_signature": mean_signature,
        "std_signature": std_signature,
        "valid_counts": valid_counts,
        "raw_min": float(np.min(finite_values)),
        "raw_max": float(np.max(finite_values)),
        "raw_mean": float(np.mean(finite_values)),
        "nan_count": int(nan_count),
        "inf_count": int(inf_count),
    }


# ============================================================
# SAVE INDIVIDUAL SIGNATURE
# ============================================================

def save_signature_csv(result, wavelengths):
    """
    Save spectral signature for one file.
    """

    filename = result["filename"]

    output_name = (
        Path(filename).stem
        + "_spectral_signature.csv"
    )

    output_path = (
        CSV_OUTPUT_DIR
        / output_name
    )

    dataframe = pd.DataFrame({
        "wavelength_nm": wavelengths,
        "mean": result["mean_signature"],
        "std": result["std_signature"],
        "valid_pixels": result["valid_counts"],
    })

    dataframe.to_csv(
        output_path,
        index=False
    )

    print(
        f"Saved CSV: {output_path}"
    )


# ============================================================
# PLOT INDIVIDUAL SIGNATURE
# ============================================================

def plot_individual_signature(
    result,
    wavelengths
):
    """
    Plot raw spectral signature with
    ±1 standard deviation.
    """

    class_name = result["class_name"]

    filename = Path(
        result["filename"]
    ).stem

    mean_signature = (
        result["mean_signature"]
    )

    std_signature = (
        result["std_signature"]
    )

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        wavelengths,
        mean_signature,
        linewidth=2,
        label=class_name
    )

    plt.fill_between(
        wavelengths,
        mean_signature - std_signature,
        mean_signature + std_signature,
        alpha=0.2
    )

    plt.xlabel(
        "Wavelength (nm)"
    )

    plt.ylabel(
        "Spectral Response"
    )

    plt.title(
        f"Tomato HSI Spectral Signature - {class_name}"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / f"{filename}_spectral_signature.png"
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
# COMBINED RAW SIGNATURE PLOT
# ============================================================

def plot_combined_raw(
    results,
    wavelengths
):
    """
    Plot raw mean signatures from all files.

    Files are plotted separately so that differences
    in numerical scale are visible.
    """

    plt.figure(
        figsize=(14, 8)
    )

    for result in results:

        plt.plot(
            wavelengths,
            result["mean_signature"],
            alpha=0.8,
            linewidth=1.5,
            label=(
                f"{result['class_name']} - "
                f"{result['filename']}"
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
        fontsize=8
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "tomato_raw_spectral_signatures.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(
        f"Saved combined graph: {output_path}"
    )


# ============================================================
# NORMALIZED CLASS COMPARISON
# ============================================================

def calculate_class_average(
    results
):
    """
    Calculate average signature for each class.

    Raw signatures are averaged within each class.
    """

    grouped = {}

    for result in results:

        class_name = result["class_name"]

        if class_name not in grouped:

            grouped[class_name] = []

        grouped[class_name].append(
            result["mean_signature"]
        )

    class_signatures = {}

    for class_name, signatures in grouped.items():

        class_signatures[class_name] = np.mean(
            np.vstack(signatures),
            axis=0
        )

    return class_signatures


def plot_normalized_class_signatures(
    class_signatures,
    wavelengths
):
    """
    Plot per-class min-max normalized signatures.

    This is intended for shape comparison only.
    """

    plt.figure(
        figsize=(14, 8)
    )

    for class_name in sorted(
        class_signatures.keys()
    ):

        signature = class_signatures[
            class_name
        ]

        normalized = normalize_for_visualization(
            signature
        )

        plt.plot(
            wavelengths,
            normalized,
            linewidth=2,
            label=class_name
        )

    plt.xlabel(
        "Wavelength (nm)"
    )

    plt.ylabel(
        "Normalized Spectral Response"
    )

    plt.title(
        "Tomato Fungal Disease Spectral Signatures"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "tomato_normalized_class_signatures.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(
        f"Saved normalized graph: {output_path}"
    )


# ============================================================
# CLASS SUMMARY CSV
# ============================================================

def save_class_summary(
    results
):
    """
    Save dataset-level summary information.
    """

    rows = []

    for result in results:

        rows.append({
            "filename": result["filename"],
            "class": result["class_name"],
            "height": result["cube_shape"][0],
            "width": result["cube_shape"][1],
            "bands": result["cube_shape"][2],
            "raw_min": result["raw_min"],
            "raw_max": result["raw_max"],
            "raw_mean": result["raw_mean"],
            "nan_count": result["nan_count"],
            "inf_count": result["inf_count"],
        })

    dataframe = pd.DataFrame(rows)

    output_path = (
        CSV_OUTPUT_DIR
        / "tomato_dataset_summary.csv"
    )

    dataframe.to_csv(
        output_path,
        index=False
    )

    print(
        f"Saved summary: {output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print_separator(
        "TOMATO FUNGAL HYPERSPECTRAL "
        "SPECTRAL SIGNATURE ANALYSIS"
    )

    print(
        f"Dataset directory:\n{DATASET_DIR}"
    )

    print(
        f"Spectral range : "
        f"{WAVELENGTH_START}-{WAVELENGTH_END} nm"
    )

    print(
        f"Expected bands : {EXPECTED_BANDS}"
    )

    # --------------------------------------------------------
    # Check directory
    # --------------------------------------------------------

    if not DATASET_DIR.exists():

        raise FileNotFoundError(
            f"Dataset directory not found:\n"
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
            "No .mat files found."
        )

    print()
    print(
        f"MAT files found: {len(mat_files)}"
    )

    # --------------------------------------------------------
    # Wavelength axis
    # --------------------------------------------------------

    wavelengths = get_wavelengths()

    print(
        f"Wavelength axis shape: "
        f"{wavelengths.shape}"
    )

    print(
        f"First wavelength: "
        f"{wavelengths[0]:.2f} nm"
    )

    print(
        f"Last wavelength : "
        f"{wavelengths[-1]:.2f} nm"
    )

    # --------------------------------------------------------
    # Process files
    # --------------------------------------------------------

    results = []

    for file_path in mat_files:

        print_separator(
            f"PROCESSING: {file_path.name}"
        )

        try:

            result = process_file(
                file_path
            )

            if result is not None:

                results.append(
                    result
                )

                save_signature_csv(
                    result,
                    wavelengths
                )

                plot_individual_signature(
                    result,
                    wavelengths
                )

        except Exception as error:

            print(
                f"ERROR processing "
                f"{file_path.name}:"
            )

            print(
                type(error).__name__,
                error
            )

    # --------------------------------------------------------
    # Check results
    # --------------------------------------------------------

    if not results:

        raise RuntimeError(
            "No valid hyperspectral files "
            "were processed."
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_separator(
        "PROCESSING SUMMARY"
    )

    print(
        f"Files successfully processed: "
        f"{len(results)}"
    )

    class_counts = {}

    for result in results:

        class_name = result["class_name"]

        class_counts[class_name] = (
            class_counts.get(
                class_name,
                0
            ) + 1
        )

    for class_name in sorted(
        class_counts
    ):

        print(
            f"{class_name:25s}: "
            f"{class_counts[class_name]} file(s)"
        )

    # --------------------------------------------------------
    # Save dataset summary
    # --------------------------------------------------------

    save_class_summary(
        results
    )

    # --------------------------------------------------------
    # Raw combined signatures
    # --------------------------------------------------------

    print_separator(
        "RAW SPECTRAL SIGNATURE COMPARISON"
    )

    plot_combined_raw(
        results,
        wavelengths
    )

    # --------------------------------------------------------
    # Class average
    # --------------------------------------------------------

    print_separator(
        "CLASS-LEVEL SPECTRAL SIGNATURES"
    )

    class_signatures = (
        calculate_class_average(
            results
        )
    )

    for class_name in sorted(
        class_signatures
    ):

        signature = class_signatures[
            class_name
        ]

        print(
            f"{class_name:25s} "
            f"Min={np.min(signature):.6f} "
            f"Max={np.max(signature):.6f} "
            f"Mean={np.mean(signature):.6f}"
        )

    # --------------------------------------------------------
    # Normalized comparison
    # --------------------------------------------------------

    plot_normalized_class_signatures(
        class_signatures,
        wavelengths
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print_separator(
        "TOMATO SPECTRAL ANALYSIS COMPLETED"
    )

    print(
        "Outputs:"
    )

    print(
        f"Graphs : {OUTPUT_DIR}"
    )

    print(
        f"CSV    : {CSV_OUTPUT_DIR}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "The normalized signatures are for "
        "visual comparison only."
    )

    print(
        "Raw spectral values have NOT been "
        "silently rescaled."
    )

    print(
        "Calibration/scaling will be decided "
        "before PCA and classification."
    )


if __name__ == "__main__":
    main()