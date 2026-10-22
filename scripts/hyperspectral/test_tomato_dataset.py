"""
test_tomato_dataset.py

Inspect the Tomato fungal-disease hyperspectral dataset.

Dataset classes:
    - Healthy
    - Alternaria alternata
    - Alternaria solani
    - Botrytis cinerea
    - Fusarium oxysporum

The dataset is stored as MATLAB .mat files.

This script DOES NOT run preprocessing or PCA.
It only determines:
    - MATLAB variables
    - variable shapes
    - data types
    - wavelength information
    - spectral dimensions
    - minimum / maximum values
    - NaN / Inf counts
    - metadata
"""

from pathlib import Path
import gc

import numpy as np


# ================================================================
# PATH CONFIGURATION
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TOMATO_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "raw"
    / "hyperspectral"
    / "tomato_hsi"
)


# ================================================================
# DATASET CLASS MAPPING
# ================================================================

CLASS_MAPPING = {
    "Healthy": "Healthy",
    "Alternaria_alternata": "Alternaria alternata",
    "Alternaria_solani": "Alternaria solani",
    "Botrytis_cinerea": "Botrytis cinerea",
    "Fusarium_oxysporum": "Fusarium oxysporum",
}


# ================================================================
# PRINT HELPERS
# ================================================================

def separator(char="=", length=70):
    print(char * length)


def print_section(title):
    separator()
    print(title)
    separator()


# ================================================================
# IDENTIFY CLASS
# ================================================================

def identify_class(filename):
    """
    Identify the disease/condition from the filename.
    """

    name = filename.lower()

    if "healthy" in name:
        return "Healthy"

    if "alternaria_alternata" in name:
        return "Alternaria alternata"

    if "alternaria_solani" in name:
        return "Alternaria solani"

    if "botrytis_cinerea" in name:
        return "Botrytis cinerea"

    if "fusarium_oxysporum" in name:
        return "Fusarium oxysporum"

    return "Unknown"


# ================================================================
# MATLAB FILE STRUCTURE INSPECTION
# ================================================================

def inspect_mat_structure(mat_path):
    """
    Inspect MATLAB file structure without loading the complete data.

    First tries scipy.io.whosmat().

    If the file is MATLAB v7.3, falls back to h5py.
    """

    print_section(
        f"MATLAB STRUCTURE: {mat_path.name}"
    )

    # ------------------------------------------------------------
    # Try scipy
    # ------------------------------------------------------------

    try:

        from scipy.io import whosmat

        variables = whosmat(str(mat_path))

        print("MATLAB variables:")

        if not variables:
            print("No variables found.")
            return []

        for name, shape, dtype in variables:

            print(
                f" - {name}"
                f" | Shape: {shape}"
                f" | Type: {dtype}"
            )

        return variables

    except Exception as scipy_error:

        print(
            "scipy.io.whosmat could not inspect this file."
        )

        print(
            "Reason:",
            type(scipy_error).__name__,
            str(scipy_error)
        )

    # ------------------------------------------------------------
    # MATLAB v7.3 fallback
    # ------------------------------------------------------------

    try:

        import h5py

        print("Trying MATLAB v7.3 / HDF5 inspection...")

        with h5py.File(mat_path, "r") as file:

            print("HDF5 datasets/groups:")

            def visitor(name, obj):

                if isinstance(obj, h5py.Dataset):

                    print(
                        f" - {name}"
                        f" | Shape: {obj.shape}"
                        f" | Dtype: {obj.dtype}"
                    )

            file.visititems(visitor)

        return []

    except Exception as h5_error:

        print(
            "HDF5 inspection failed."
        )

        print(
            "Reason:",
            type(h5_error).__name__,
            str(h5_error)
        )

        return []


# ================================================================
# LOAD MATLAB FILE
# ================================================================

def load_mat_file(mat_path):
    """
    Load a MATLAB .mat file.

    Supports:
        - MATLAB v5/v7
        - MATLAB v7.3 HDF5
    """

    # ------------------------------------------------------------
    # Try scipy
    # ------------------------------------------------------------

    try:

        from scipy.io import loadmat

        data = loadmat(
            str(mat_path),
            squeeze_me=False,
            struct_as_record=False
        )

        # Remove MATLAB internal variables
        data = {
            key: value
            for key, value in data.items()
            if not key.startswith("__")
        }

        return data

    except Exception as scipy_error:

        print(
            "scipy.io.loadmat failed:"
        )

        print(
            type(scipy_error).__name__,
            str(scipy_error)
        )

    # ------------------------------------------------------------
    # Try HDF5
    # ------------------------------------------------------------

    try:

        import h5py

        data = {}

        with h5py.File(mat_path, "r") as file:

            def visitor(name, obj):

                if isinstance(obj, h5py.Dataset):

                    try:
                        data[name] = obj[()]
                    except Exception:
                        pass

            file.visititems(visitor)

        return data

    except Exception as h5_error:

        raise RuntimeError(
            f"Unable to read MATLAB file: {mat_path}\n"
            f"scipy error: {scipy_error}\n"
            f"HDF5 error: {h5_error}"
        )


# ================================================================
# NUMPY ARRAY INSPECTION
# ================================================================

def inspect_array(name, value):
    """
    Inspect a NumPy-compatible variable.
    """

    print()
    print(f"Variable: {name}")

    # ------------------------------------------------------------
    # Convert if possible
    # ------------------------------------------------------------

    try:

        array = np.asarray(value)

    except Exception as error:

        print(
            "Could not convert variable to NumPy array."
        )

        print(
            type(error).__name__,
            str(error)
        )

        return

    print("Shape :", array.shape)
    print("Dtype :", array.dtype)

    # ------------------------------------------------------------
    # Numeric data
    # ------------------------------------------------------------

    if np.issubdtype(array.dtype, np.number):

        if array.size == 0:

            print("Array is empty.")
            return

        # Avoid expensive calculations on unusual objects
        try:

            finite_mask = np.isfinite(array)

            nan_count = np.isnan(array).sum()
            inf_count = np.isinf(array).sum()

            print("NaN values       :", int(nan_count))
            print("Infinite values  :", int(inf_count))

            if np.any(finite_mask):

                finite_values = array[finite_mask]

                print(
                    "Minimum          :",
                    float(np.min(finite_values))
                )

                print(
                    "Maximum          :",
                    float(np.max(finite_values))
                )

                print(
                    "Mean             :",
                    float(np.mean(finite_values))
                )

        except Exception as error:

            print(
                "Could not calculate numeric statistics:"
            )

            print(
                type(error).__name__,
                str(error)
            )

    else:

        print(
            "Non-numeric variable."
        )

        # Print small textual values when possible
        try:

            if array.size <= 20:

                print(
                    "Value:",
                    array
                )

        except Exception:
            pass


# ================================================================
# FIND POSSIBLE WAVELENGTH VARIABLES
# ================================================================

def find_wavelength_variables(data):
    """
    Search MATLAB variables for possible wavelength information.
    """

    wavelength_keywords = [
        "wavelength",
        "wavelengths",
        "lambda",
        "band",
        "bands",
        "spectral"
    ]

    matches = []

    for name in data.keys():

        name_lower = name.lower()

        for keyword in wavelength_keywords:

            if keyword in name_lower:

                matches.append(name)
                break

    return matches


# ================================================================
# INSPECT ONE TOMATO FILE
# ================================================================

def inspect_file(mat_path):
    """
    Fully inspect one .mat file.
    """

    condition = identify_class(mat_path.name)

    print()
    separator("#")
    print(
        f"DATASET FILE: {mat_path.name}"
    )
    print(
        f"Condition    : {condition}"
    )
    print(
        f"File size    : "
        f"{mat_path.stat().st_size / (1024 ** 2):.2f} MB"
    )
    separator("#")

    # ------------------------------------------------------------
    # Structure inspection
    # ------------------------------------------------------------

    variables = inspect_mat_structure(mat_path)

    # ------------------------------------------------------------
    # Load actual data
    # ------------------------------------------------------------

    print_section(
        f"LOADING DATA: {mat_path.name}"
    )

    try:

        data = load_mat_file(mat_path)

    except Exception as error:

        print(
            "ERROR LOADING FILE"
        )

        print(
            type(error).__name__,
            ":",
            str(error)
        )

        return

    print(
        "Variables loaded:",
        len(data)
    )

    # ------------------------------------------------------------
    # Print variables
    # ------------------------------------------------------------

    print_section(
        "VARIABLE DETAILS"
    )

    for name, value in data.items():

        inspect_array(
            name,
            value
        )

    # ------------------------------------------------------------
    # Search for wavelength information
    # ------------------------------------------------------------

    print_section(
        "POSSIBLE WAVELENGTH INFORMATION"
    )

    wavelength_variables = find_wavelength_variables(
        data
    )

    if wavelength_variables:

        for name in wavelength_variables:

            print(
                "Possible wavelength variable:",
                name
            )

            try:

                value = np.asarray(
                    data[name]
                )

                print(
                    "Shape:",
                    value.shape
                )

                if value.size <= 100:

                    print(
                        "Values:",
                        value.flatten()
                    )

            except Exception:
                pass

    else:

        print(
            "No obvious wavelength variable found."
        )

    # ------------------------------------------------------------
    # Release memory
    # ------------------------------------------------------------

    del data

    gc.collect()


# ================================================================
# MAIN
# ================================================================

def main():

    print()
    separator("#")
    print(
        "TOMATO FUNGAL HYPERSPECTRAL DATASET INSPECTION"
    )
    separator("#")

    print()
    print(
        "Project directory:"
    )

    print(
        TOMATO_DIR
    )

    # ------------------------------------------------------------
    # Check directory
    # ------------------------------------------------------------

    if not TOMATO_DIR.exists():

        print()
        print(
            "ERROR: Tomato dataset directory not found."
        )

        print(
            "Expected:"
        )

        print(
            TOMATO_DIR
        )

        return

    # ------------------------------------------------------------
    # Find MATLAB files
    # ------------------------------------------------------------

    mat_files = sorted(
        TOMATO_DIR.glob("*.mat")
    )

    if not mat_files:

        print()
        print(
            "ERROR: No .mat files found."
        )

        return

    print()
    print(
        f"MATLAB files found: {len(mat_files)}"
    )

    # ------------------------------------------------------------
    # Dataset summary
    # ------------------------------------------------------------

    print_section(
        "DATASET FILE SUMMARY"
    )

    class_counts = {}

    for mat_file in mat_files:

        condition = identify_class(
            mat_file.name
        )

        class_counts[condition] = (
            class_counts.get(condition, 0) + 1
        )

        size_mb = (
            mat_file.stat().st_size
            / (1024 ** 2)
        )

        print(
            f"{mat_file.name:<40}"
            f"{size_mb:>10.2f} MB"
            f"    {condition}"
        )

    # ------------------------------------------------------------
    # Class summary
    # ------------------------------------------------------------

    print_section(
        "CLASS SUMMARY"
    )

    for condition, count in class_counts.items():

        print(
            f"{condition:<25} : {count} file(s)"
        )

    # ------------------------------------------------------------
    # Inspect each file
    # ------------------------------------------------------------

    for mat_file in mat_files:

        inspect_file(
            mat_file
        )

    # ------------------------------------------------------------
    # Completion
    # ------------------------------------------------------------

    print()
    separator("#")
    print(
        "TOMATO DATASET INSPECTION COMPLETED"
    )
    separator("#")


if __name__ == "__main__":
    main()