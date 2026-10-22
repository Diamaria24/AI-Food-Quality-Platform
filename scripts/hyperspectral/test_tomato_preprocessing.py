"""
test_tomato_preprocessing.py

Tests the tomato-specific hyperspectral preprocessing
pipeline on the currently downloaded tomato MAT files.
"""

import os
import sys

import numpy as np
from scipy.io import loadmat

# ------------------------------------------------------------
# Allow importing modules from this directory
# ------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from tomato_preprocessing import TomatoHSIPreprocessor


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        SCRIPT_DIR,
        "..",
        ".."
    )
)

TOMATO_DIR = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "raw",
    "hyperspectral",
    "tomato_hsi"
)


# ------------------------------------------------------------
# Expected classes
# ------------------------------------------------------------

CLASS_MAP = {

    "Alternaria_alternata":
        "Alternaria alternata",

    "Alternaria_solani":
        "Alternaria solani",

    "Botrytis_cinerea":
        "Botrytis cinerea",

    "Fusarium_oxysporum":
        "Fusarium oxysporum",

    "Healthy":
        "Healthy"
}


# ------------------------------------------------------------
# Find hyperspectral variable
# ------------------------------------------------------------

def find_hyperspectral_variable(mat_data):

    candidates = []

    for name, value in mat_data.items():

        if name.startswith("__"):
            continue

        if isinstance(value, np.ndarray):

            if value.ndim == 3:

                candidates.append(
                    (name, value)
                )

    if not candidates:

        raise ValueError(
            "No 3D hyperspectral variable "
            "was found in MAT file."
        )

    # Prefer the expected shape
    for name, value in candidates:

        if value.shape == (140, 280, 550):

            return name, value

    # Otherwise return first 3D variable
    return candidates[0]


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print()
    print("=" * 70)
    print("TOMATO HSI PREPROCESSING TEST")
    print("=" * 70)

    print()
    print("Dataset directory:")
    print(TOMATO_DIR)

    if not os.path.isdir(TOMATO_DIR):

        print()
        print("ERROR:")
        print("Tomato dataset directory not found.")

        return

    mat_files = sorted(
        [
            f
            for f in os.listdir(TOMATO_DIR)
            if f.lower().endswith(".mat")
        ]
    )

    print()
    print("MAT files found:", len(mat_files))

    if not mat_files:

        print("No MAT files found.")
        return

    print()
    print("=" * 70)
    print("PROCESSING FILES")
    print("=" * 70)

    success_count = 0

    for filename in mat_files:

        print()
        print("-" * 70)
        print("FILE:", filename)
        print("-" * 70)

        try:

            filepath = os.path.join(
                TOMATO_DIR,
                filename
            )

            # ------------------------------------------------
            # Load MAT file
            # ------------------------------------------------

            mat_data = loadmat(
                filepath
            )

            variable_name, cube = (
                find_hyperspectral_variable(
                    mat_data
                )
            )

            cube = np.asarray(
                cube,
                dtype=np.float32
            )

            print(
                "Variable       :",
                variable_name
            )

            print(
                "Original shape :",
                cube.shape
            )

            print(
                "Original dtype :",
                cube.dtype
            )

            # ------------------------------------------------
            # Raw statistics
            # ------------------------------------------------

            print()
            print("RAW DATA")

            print(
                "Minimum        :",
                float(np.min(cube))
            )

            print(
                "Maximum        :",
                float(np.max(cube))
            )

            print(
                "Mean           :",
                float(np.mean(cube))
            )

            print(
                "NaN values     :",
                int(np.isnan(cube).sum())
            )

            print(
                "Infinite values:",
                int(np.isinf(cube).sum())
            )

            # ------------------------------------------------
            # Scale detection
            # ------------------------------------------------

            scale = (
                TomatoHSIPreprocessor.detect_scale(
                    cube
                )
            )

            print()
            print(
                "Detected scale :",
                scale
            )

            # ------------------------------------------------
            # Wavelength axis
            # ------------------------------------------------

            wavelengths = (
                TomatoHSIPreprocessor.get_wavelengths()
            )

            print()
            print(
                "Wavelengths    :",
                wavelengths.shape
            )

            print(
                "First wavelength:",
                float(wavelengths[0]),
                "nm"
            )

            print(
                "Last wavelength :",
                float(wavelengths[-1]),
                "nm"
            )

            # ------------------------------------------------
            # Preprocessing
            # ------------------------------------------------

            print()
            print("PREPROCESSING...")

            processed, metadata = (
                TomatoHSIPreprocessor.process(
                    cube
                )
            )

            # ------------------------------------------------
            # Results
            # ------------------------------------------------

            print()
            print("PROCESSED DATA")

            print(
                "Shape          :",
                processed.shape
            )

            print(
                "Dtype          :",
                processed.dtype
            )

            print(
                "Detected scale :",
                metadata["detected_scale"]
            )

            print(
                "Used scale     :",
                metadata["used_scale"]
            )

            print(
                "Original min   :",
                metadata["original_min"]
            )

            print(
                "Original max   :",
                metadata["original_max"]
            )

            print(
                "Final min      :",
                metadata["final_min"]
            )

            print(
                "Final max      :",
                metadata["final_max"]
            )

            print(
                "Final mean     :",
                metadata["final_mean"]
            )

            print()
            print(
                "Savitzky-Golay :",
                metadata["smoothing"]
            )

            print(
                "Window length  :",
                metadata["savgol_window"]
            )

            print(
                "Polynomial order:",
                metadata["savgol_polyorder"]
            )

            print(
                "Mean centered  :",
                metadata["mean_centering"]
            )

            # ------------------------------------------------
            # Validation
            # ------------------------------------------------

            print()
            print("VALIDATION")

            if processed.shape == (140, 280, 550):

                print(
                    "Shape check    : PASS"
                )

            else:

                print(
                    "Shape check    : FAIL"
                )

            nan_count = int(
                np.isnan(processed).sum()
            )

            inf_count = int(
                np.isinf(processed).sum()
            )

            print(
                "NaN values     :",
                nan_count
            )

            print(
                "Infinite values:",
                inf_count
            )

            if nan_count == 0 and inf_count == 0:

                print(
                    "Numerical check: PASS"
                )

            else:

                print(
                    "Numerical check: FAIL"
                )

            success_count += 1

        except Exception as exc:

            print()
            print(
                "ERROR PROCESSING:",
                filename
            )

            print(
                type(exc).__name__,
                ":",
                exc
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TOMATO PREPROCESSING TEST COMPLETED")
    print("=" * 70)

    print(
        "Files processed successfully:",
        success_count,
        "/",
        len(mat_files)
    )

    print()

    if success_count == len(mat_files):

        print(
            "ALL TOMATO FILES PASSED PREPROCESSING."
        )

    else:

        print(
            "Some files require investigation."
        )

    print()


if __name__ == "__main__":
    main()