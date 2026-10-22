"""
dataset_info.py

Inspect the structure and available metadata/labels of the
hyperspectral datasets used in the project.

This script does NOT create or assume labels.

It is used to determine:
    - dataset dimensions
    - file structure
    - available metadata
    - possible class/condition information
    - whether supervised classification is possible
    - Papaya Label_6Class distribution
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ==========================================================
# IMPORT PATH
# ==========================================================

CURRENT_DIR = Path(__file__).resolve().parent

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))


from loaders import HyperspectralLoader


# ==========================================================
# PROJECT PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

HYPERSPECTRAL_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "raw"
    / "hyperspectral"
)


# ==========================================================
# DATASET DIRECTORIES
# ==========================================================

DATASETS = {

    "Papaya": (
        HYPERSPECTRAL_ROOT
        / "papaya_hsi"
    ),

    "Eggplant": (
        HYPERSPECTRAL_ROOT
        / "eggplant_hsi"
    ),

    "Cabbage": (
        HYPERSPECTRAL_ROOT
        / "cabbage_hsi"
    ),

    "Strawberry": (
        HYPERSPECTRAL_ROOT
        / "strawberry_hsi"
    )
}


# ==========================================================
# PAPAYA DOCUMENTATION CSV
# ==========================================================

PAPAYA_CSV = (
    HYPERSPECTRAL_ROOT
    / "papaya_hsi"
    / "Garillos-Manliguez_Hyperspectral Images of Papaya"
    / "hsi32_dataset_example files directory.csv"
)


# ==========================================================
# COMMON LABEL KEYWORDS
# ==========================================================

LABEL_KEYWORDS = [

    "label",
    "class",
    "condition",
    "quality",
    "quality_class",
    "disease",
    "diseased",
    "mold",
    "mould",
    "fresh",
    "rotten",
    "healthy",
    "unhealthy",
    "good",
    "bad",
    "defect",
    "defective"
]


# ==========================================================
# PRINT HEADER
# ==========================================================

def print_header(title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ==========================================================
# INSPECT FILE STRUCTURE
# ==========================================================

def inspect_files(
    dataset_name,
    dataset_path
):

    print_header(
        f"{dataset_name.upper()} FILE STRUCTURE"
    )

    if not dataset_path.exists():

        print(
            "Dataset directory does not exist:"
        )

        print(dataset_path)

        return []

    files = [
        p
        for p in dataset_path.rglob("*")
        if p.is_file()
    ]

    print(
        "Dataset directory:",
        dataset_path
    )

    print(
        "Total files:",
        len(files)
    )

    print()

    for file in files:

        relative = file.relative_to(
            dataset_path
        )

        size_mb = (
            file.stat().st_size
            / (1024 * 1024)
        )

        print(
            f"{relative}"
            f"  [{size_mb:.2f} MB]"
        )

    return files


# ==========================================================
# SEARCH FILENAMES FOR LABEL INFORMATION
# ==========================================================

def inspect_filename_labels(files):

    print_header(
        "POSSIBLE LABEL INFORMATION IN FILENAMES"
    )

    matches = []

    for file in files:

        name = file.name.lower()

        matched_keywords = [
            keyword
            for keyword in LABEL_KEYWORDS
            if keyword in name
        ]

        if matched_keywords:

            matches.append(
                (
                    file,
                    matched_keywords
                )
            )

    if not matches:

        print(
            "No obvious label/condition keywords "
            "were found in filenames."
        )

    else:

        for file, keywords in matches:

            print(
                file,
                "->",
                ", ".join(keywords)
            )

    return matches


# ==========================================================
# INSPECT DIRECTORY NAMES
# ==========================================================

def inspect_directory_labels(
    dataset_name,
    dataset_path
):

    print_header(
        f"{dataset_name.upper()} DIRECTORY ANALYSIS"
    )

    if not dataset_path.exists():

        print(
            "Directory does not exist."
        )

        return

    directories = [
        p
        for p in dataset_path.rglob("*")
        if p.is_dir()
    ]

    print(
        "Directories:"
    )

    if not directories:

        print(
            "No subdirectories found."
        )

    for directory in directories:

        relative = directory.relative_to(
            dataset_path
        )

        print(
            " -",
            relative
        )

    print()

    print(
        "Directories with possible "
        "class/condition names:"
    )

    found = False

    for directory in directories:

        name = directory.name.lower()

        matched = [
            keyword
            for keyword in LABEL_KEYWORDS
            if keyword in name
        ]

        if matched:

            print(
                f" - {directory.name}"
                f" -> {', '.join(matched)}"
            )

            found = True

    if not found:

        print(
            "No obvious class/condition "
            "directories found."
        )


# ==========================================================
# INSPECT CSV
# ==========================================================

def inspect_csv(file_path):

    print_header(
        f"CSV INSPECTION: {file_path.name}"
    )

    try:

        # Read a limited number of rows for the
        # general inspection.
        df = pd.read_csv(
            file_path,
            nrows=1000
        )

    except Exception as e:

        print(
            "Could not read CSV:",
            type(e).__name__,
            e
        )

        return

    print(
        "Columns:",
        len(df.columns)
    )

    print(
        "Rows inspected:",
        len(df)
    )

    print()

    print(
        "Column names:"
    )

    for column in df.columns:

        print(
            " -",
            column
        )

    print()

    print(
        "Possible label columns:"
    )

    found = False

    for column in df.columns:

        lower = str(column).lower()

        if any(
            keyword in lower
            for keyword in LABEL_KEYWORDS
        ):

            print(
                " -",
                column
            )

            found = True

    if not found:

        print(
            "None found from keyword search."
        )

    print()

    print(
        "Data types:"
    )

    print(
        df.dtypes.to_string()
    )

    print()

    print(
        "First 5 rows:"
    )

    print(
        df.head().to_string(
            index=False
        )
    )


# ==========================================================
# PAPAYA LABEL ANALYSIS
# ==========================================================

def analyze_papaya_labels(
    csv_path
):

    print_header(
        "PAPAYA LABEL_6CLASS ANALYSIS"
    )

    if not csv_path.exists():

        print(
            "Papaya documentation CSV not found:"
        )

        print(csv_path)

        return

    print(
        "CSV file:",
        csv_path
    )

    try:

        df = pd.read_csv(
            csv_path
        )

    except Exception as e:

        print(
            "Could not read Papaya CSV:",
            type(e).__name__,
            e
        )

        return

    # ------------------------------------------------------
    # Basic information
    # ------------------------------------------------------

    print()

    print(
        "Total rows:",
        len(df)
    )

    print(
        "Total columns:",
        len(df.columns)
    )

    # ------------------------------------------------------
    # Check required columns
    # ------------------------------------------------------

    required_columns = [
        "Subregions",
        "RGB_Filename",
        "Header_Cropped",
        "Sample_ID",
        "Label_6Class",
        "Subregions_Img"
    ]

    print()

    print(
        "Required documentation columns:"
    )

    for column in required_columns:

        if column in df.columns:

            print(
                f" - {column}: FOUND"
            )

        else:

            print(
                f" - {column}: NOT FOUND"
            )

    # ------------------------------------------------------
    # Label analysis
    # ------------------------------------------------------

    if "Label_6Class" not in df.columns:

        print()

        print(
            "Label_6Class column was not found."
        )

        return

    print()

    print(
        "=" * 70
    )

    print(
        "LABEL_6CLASS DISTRIBUTION"
    )

    print(
        "=" * 70
    )

    # Count every class
    counts = (
        df["Label_6Class"]
        .value_counts(
            dropna=False
        )
        .sort_index()
    )

    total = len(df)

    for label, count in counts.items():

        percentage = (
            count / total * 100
        )

        print(
            f"Class {label}: "
            f"{count} samples "
            f"({percentage:.2f}%)"
        )

    # ------------------------------------------------------
    # Unique labels
    # ------------------------------------------------------

    print()

    print(
        "=" * 70
    )

    print(
        "UNIQUE LABEL VALUES"
    )

    print(
        "=" * 70
    )

    unique_labels = (
        df["Label_6Class"]
        .dropna()
        .unique()
        .tolist()
    )

    unique_labels = sorted(
        unique_labels
    )

    print(
        unique_labels
    )

    print(
        "Number of unique classes:",
        len(unique_labels)
    )

    # ------------------------------------------------------
    # Missing labels
    # ------------------------------------------------------

    missing_labels = (
        df["Label_6Class"]
        .isna()
        .sum()
    )

    print()

    print(
        "Missing Label_6Class values:",
        missing_labels
    )

    # ------------------------------------------------------
    # Sample ID information
    # ------------------------------------------------------

    if "Sample_ID" in df.columns:

        print()

        print(
            "=" * 70
        )

        print(
            "SAMPLE ID INFORMATION"
        )

        print(
            "=" * 70
        )

        print(
            "Minimum Sample_ID:",
            df["Sample_ID"].min()
        )

        print(
            "Maximum Sample_ID:",
            df["Sample_ID"].max()
        )

        print(
            "Unique Sample_ID values:",
            df["Sample_ID"].nunique()
        )

    # ------------------------------------------------------
    # Class examples
    # ------------------------------------------------------

    print()

    print(
        "=" * 70
    )

    print(
        "EXAMPLES FROM EACH LABEL CLASS"
    )

    print(
        "=" * 70
    )

    display_columns = [
        column
        for column in [
            "Subregions",
            "RGB_Filename",
            "Header_Cropped",
            "Sample_ID",
            "Label_6Class",
            "Subregions_Img"
        ]
        if column in df.columns
    ]

    for label in unique_labels:

        print()

        print(
            f"CLASS {label}"
        )

        print(
            "-" * 70
        )

        subset = df[
            df["Label_6Class"] == label
        ]

        print(
            subset[
                display_columns
            ]
            .head(10)
            .to_string(
                index=False
            )
        )

    # ------------------------------------------------------
    # HSI file references
    # ------------------------------------------------------

    if "Subregions" in df.columns:

        print()

        print(
            "=" * 70
        )

        print(
            "PAPAYA HSI FILE REFERENCES"
        )

        print(
            "=" * 70
        )

        print(
            "Example Subregions entries:"
        )

        print(
            df[
                "Subregions"
            ]
            .head(10)
            .to_string(
                index=False
            )
        )

    if "Header_Cropped" in df.columns:

        print()

        print(
            "Example Header_Cropped entries:"
        )

        print(
            df[
                "Header_Cropped"
            ]
            .head(10)
            .to_string(
                index=False
            )
        )


# ==========================================================
# INSPECT ENVI HDR
# ==========================================================

def inspect_hdr(
    file_path
):

    print_header(
        f"ENVI HDR INSPECTION: {file_path.name}"
    )

    try:

        # Use the existing loader method.
        img, cube = (
            HyperspectralLoader.load_hdr(
                file_path
            )
        )

    except Exception as e:

        print(
            "Could not open HDR:",
            type(e).__name__,
            e
        )

        return

    # ------------------------------------------------------
    # Basic image information
    # ------------------------------------------------------

    print(
        "Shape:",
        cube.shape
    )

    print(
        "Dtype:",
        cube.dtype
    )

    # ------------------------------------------------------
    # Metadata
    # ------------------------------------------------------

    print()

    print(
        "Metadata:"
    )

    metadata = img.metadata

    if not metadata:

        print(
            "No metadata found."
        )

    else:

        for key, value in metadata.items():

            print(
                f"{key}: {value}"
            )

    # ------------------------------------------------------
    # Search metadata for possible labels
    # ------------------------------------------------------

    print()

    print(
        "Possible label/condition metadata:"
    )

    found = False

    for key, value in metadata.items():

        key_lower = str(
            key
        ).lower()

        value_lower = str(
            value
        ).lower()

        key_match = any(
            keyword in key_lower
            for keyword in LABEL_KEYWORDS
        )

        value_match = any(
            keyword in value_lower
            for keyword in LABEL_KEYWORDS
        )

        if key_match or value_match:

            print(
                f"{key}: {value}"
            )

            found = True

    if not found:

        print(
            "No obvious label information "
            "found in HDR metadata."
        )


# ==========================================================
# INSPECT DATASET
# ==========================================================

def inspect_dataset(
    dataset_name,
    dataset_path
):

    print()

    print(
        "#" * 70
    )

    print(
        f"DATASET INSPECTION: "
        f"{dataset_name.upper()}"
    )

    print(
        "#" * 70
    )

    # ------------------------------------------------------
    # File structure
    # ------------------------------------------------------

    files = inspect_files(
        dataset_name,
        dataset_path
    )

    # ------------------------------------------------------
    # Directory analysis
    # ------------------------------------------------------

    inspect_directory_labels(
        dataset_name,
        dataset_path
    )

    # ------------------------------------------------------
    # Filename analysis
    # ------------------------------------------------------

    inspect_filename_labels(
        files
    )

    # ------------------------------------------------------
    # File types
    # ------------------------------------------------------

    hdr_files = [
        file
        for file in files
        if file.suffix.lower() == ".hdr"
    ]

    csv_files = [
        file
        for file in files
        if file.suffix.lower() == ".csv"
    ]

    # ------------------------------------------------------
    # HDR files
    # ------------------------------------------------------

    if hdr_files:

        print_header(
            "HDR FILES FOUND"
        )

        for file in hdr_files:

            print(
                file.relative_to(
                    dataset_path
                )
            )

    # ------------------------------------------------------
    # CSV files
    # ------------------------------------------------------

    if csv_files:

        print_header(
            "CSV FILES FOUND"
        )

        for file in csv_files:

            print(
                file.relative_to(
                    dataset_path
                )
            )

    # ------------------------------------------------------
    # Inspect first HDR
    # ------------------------------------------------------

    if hdr_files:

        inspect_hdr(
            hdr_files[0]
        )

    # ------------------------------------------------------
    # Inspect CSV files
    # ------------------------------------------------------

    for csv_file in csv_files:

        inspect_csv(
            csv_file
        )


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    print_header(
        "HYPERSPECTRAL DATASET "
        "INFORMATION INSPECTION"
    )

    print(
        "Project root:",
        PROJECT_ROOT
    )

    print(
        "Dataset root:",
        HYPERSPECTRAL_ROOT
    )

    # ------------------------------------------------------
    # Papaya label analysis FIRST
    # ------------------------------------------------------

    print()

    print(
        "#" * 70
    )

    print(
        "PAPAYA LABEL INVESTIGATION"
    )

    print(
        "#" * 70
    )

    analyze_papaya_labels(
        PAPAYA_CSV
    )

    # ------------------------------------------------------
    # Inspect all available datasets
    # ------------------------------------------------------

    for dataset_name, dataset_path in DATASETS.items():

        if not dataset_path.exists():

            print()

            print(
                f"Skipping {dataset_name}: "
                "directory not found."
            )

            continue

        try:

            inspect_dataset(
                dataset_name,
                dataset_path
            )

        except Exception as e:

            print()

            print(
                "=" * 70
            )

            print(
                f"ERROR INSPECTING "
                f"{dataset_name.upper()}"
            )

            print(
                "=" * 70
            )

            print(
                type(e).__name__,
                ":",
                e
            )

    # ------------------------------------------------------
    # Complete
    # ------------------------------------------------------

    print()

    print(
        "=" * 70
    )

    print(
        "DATASET INFORMATION INSPECTION COMPLETED"
    )

    print(
        "=" * 70
    )