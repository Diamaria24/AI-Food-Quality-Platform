"""
test_tomato_classification_data.py

Prepare the real Tomato HSI PCA dataset for classification.

IMPORTANT:
The train/test split is performed at SOURCE FILE / CUBE level.

This prevents pixels from the same hyperspectral cube from
appearing in both training and testing.

Current dataset:
    5 classes
    2 MAT files per class

Split:
    1 MAT file per class -> TRAIN
    1 MAT file per class -> TEST

Each file:
    5,000 PCA samples

Expected:
    25,000 training samples
    25,000 testing samples
"""

import os
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

INPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "processed",
    "hyperspectral",
    "tomato_pca",
    "tomato_pca_features.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "processed",
    "hyperspectral",
    "tomato_classification"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# CLASS ORDER
# ============================================================

CLASS_ORDER = [
    "Healthy",
    "Alternaria alternata",
    "Alternaria solani",
    "Botrytis cinerea",
    "Fusarium oxysporum"
]


# ============================================================
# HEADER
# ============================================================

def header(title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    header(
        "TOMATO HSI CLASSIFICATION DATA PREPARATION"
    )

    print(
        "Input file:"
    )

    print(
        INPUT_FILE
    )

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not os.path.exists(INPUT_FILE):

        print()
        print(
            "ERROR: PCA feature file not found."
        )

        return

    # --------------------------------------------------------
    # Load PCA data
    # --------------------------------------------------------

    dataframe = pd.read_csv(
        INPUT_FILE
    )

    print()
    print(
        "INPUT DATA"
    )

    print(
        f"Shape       : {dataframe.shape}"
    )

    print(
        f"Columns     : {len(dataframe.columns)}"
    )

    print(
        f"Rows        : {len(dataframe)}"
    )

    # --------------------------------------------------------
    # Identify PCA columns
    # --------------------------------------------------------

    pca_columns = [
        column
        for column in dataframe.columns
        if column.startswith("PC")
    ]

    print()
    print(
        f"PCA features: {len(pca_columns)}"
    )

    print(
        "PCA columns:"
    )

    print(
        pca_columns
    )

    # --------------------------------------------------------
    # Required metadata
    # --------------------------------------------------------

    required_columns = [
        "class",
        "source_file"
    ]

    for column in required_columns:

        if column not in dataframe.columns:

            raise ValueError(
                f"Required column missing: {column}"
            )

    # --------------------------------------------------------
    # Numerical validation
    # --------------------------------------------------------

    print()
    print(
        "NUMERICAL VALIDATION"
    )

    X = dataframe[
        pca_columns
    ]

    print(
        f"NaN values       : "
        f"{X.isna().sum().sum()}"
    )

    print(
        f"Infinite values  : "
        f"{np.isinf(X.to_numpy()).sum()}"
    )

    # --------------------------------------------------------
    # File-level class information
    # --------------------------------------------------------

    header(
        "SOURCE FILE DISTRIBUTION"
    )

    file_table = (
        dataframe[
            ["source_file", "class"]
        ]
        .drop_duplicates()
        .sort_values(
            ["class", "source_file"]
        )
    )

    print(
        file_table.to_string(
            index=False
        )
    )

    print()
    print(
        f"Number of source cubes: "
        f"{len(file_table)}"
    )

    # --------------------------------------------------------
    # Verify exactly two files/class
    # --------------------------------------------------------

    files_per_class = (
        file_table
        .groupby("class")
        .size()
    )

    print()
    print(
        "FILES PER CLASS"
    )

    print(
        files_per_class
    )

    if not all(
        files_per_class == 2
    ):

        raise ValueError(
            "Expected exactly 2 source files "
            "per class for the current experiment."
        )

    # --------------------------------------------------------
    # Create deterministic cube-level split
    # --------------------------------------------------------

    header(
        "CREATING CUBE-LEVEL TRAIN / TEST SPLIT"
    )

    train_files = []
    test_files = []

    for class_name in CLASS_ORDER:

        class_files = sorted(
            file_table[
                file_table["class"] == class_name
            ]["source_file"]
            .tolist()
        )

        if len(class_files) != 2:

            raise ValueError(
                f"Expected 2 files for {class_name}, "
                f"found {len(class_files)}."
            )

        # Deterministic split
        train_file = class_files[0]
        test_file = class_files[1]

        train_files.append(
            train_file
        )

        test_files.append(
            test_file
        )

        print()
        print(
            f"{class_name}"
        )

        print(
            f"  TRAIN: {train_file}"
        )

        print(
            f"  TEST : {test_file}"
        )

    # --------------------------------------------------------
    # Assign split
    # --------------------------------------------------------

    dataframe["split"] = np.where(
        dataframe["source_file"].isin(
            train_files
        ),
        "train",
        "test"
    )

    # --------------------------------------------------------
    # Verify no source-file leakage
    # --------------------------------------------------------

    train_source_files = set(
        dataframe[
            dataframe["split"] == "train"
        ]["source_file"]
    )

    test_source_files = set(
        dataframe[
            dataframe["split"] == "test"
        ]["source_file"]
    )

    overlap = (
        train_source_files
        .intersection(
            test_source_files
        )
    )

    print()
    print(
        "SOURCE FILE LEAKAGE CHECK"
    )

    print(
        f"Train source files: "
        f"{len(train_source_files)}"
    )

    print(
        f"Test source files : "
        f"{len(test_source_files)}"
    )

    print(
        f"Overlapping files : "
        f"{len(overlap)}"
    )

    if overlap:

        raise ValueError(
            "DATA LEAKAGE DETECTED: "
            "source file occurs in both train and test."
        )

    print(
        "Leakage check: PASS"
    )

    # --------------------------------------------------------
    # Separate datasets
    # --------------------------------------------------------

    train_df = dataframe[
        dataframe["split"] == "train"
    ].copy()

    test_df = dataframe[
        dataframe["split"] == "test"
    ].copy()

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    header(
        "TRAINING CLASS DISTRIBUTION"
    )

    print(
        train_df["class"].value_counts()
    )

    header(
        "TEST CLASS DISTRIBUTION"
    )

    print(
        test_df["class"].value_counts()
    )

    # --------------------------------------------------------
    # Expected sample counts
    # --------------------------------------------------------

    print()
    print(
        "SAMPLE COUNTS"
    )

    print(
        f"Training samples: "
        f"{len(train_df)}"
    )

    print(
        f"Testing samples : "
        f"{len(test_df)}"
    )

    # --------------------------------------------------------
    # Save complete split dataset
    # --------------------------------------------------------

    complete_output = os.path.join(
        OUTPUT_DIR,
        "tomato_classification_dataset.csv"
    )

    dataframe.to_csv(
        complete_output,
        index=False
    )

    print()
    print(
        f"Saved complete dataset:"
    )

    print(
        complete_output
    )

    # --------------------------------------------------------
    # Save training dataset
    # --------------------------------------------------------

    train_output = os.path.join(
        OUTPUT_DIR,
        "tomato_train.csv"
    )

    train_df.to_csv(
        train_output,
        index=False
    )

    print(
        f"Saved training dataset:"
    )

    print(
        train_output
    )

    # --------------------------------------------------------
    # Save testing dataset
    # --------------------------------------------------------

    test_output = os.path.join(
        OUTPUT_DIR,
        "tomato_test.csv"
    )

    test_df.to_csv(
        test_output,
        index=False
    )

    print(
        f"Saved testing dataset:"
    )

    print(
        test_output
    )

    # --------------------------------------------------------
    # Save split information
    # --------------------------------------------------------

    split_info = file_table.copy()

    split_info["split"] = np.where(
        split_info["source_file"].isin(
            train_files
        ),
        "train",
        "test"
    )

    split_info_output = os.path.join(
        OUTPUT_DIR,
        "tomato_source_file_split.csv"
    )

    split_info.to_csv(
        split_info_output,
        index=False
    )

    print(
        f"Saved source-file split:"
    )

    print(
        split_info_output
    )

    # --------------------------------------------------------
    # Final feature matrix information
    # --------------------------------------------------------

    header(
        "CLASSIFICATION-READY DATA"
    )

    print(
        f"PCA features        : "
        f"{len(pca_columns)}"
    )

    print(
        f"Training matrix     : "
        f"{train_df[pca_columns].shape}"
    )

    print(
        f"Testing matrix      : "
        f"{test_df[pca_columns].shape}"
    )

    print(
        f"Training labels     : "
        f"{train_df['class'].shape}"
    )

    print(
        f"Testing labels      : "
        f"{test_df['class'].shape}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Train/test separation was performed "
        "at source-cube level."
    )

    print(
        "Pixels from the same MAT file were "
        "never divided between train and test."
    )

    print()
    print(
        "This dataset is now ready for "
        "classifier development."
    )

    header(
        "TOMATO CLASSIFICATION DATA PREPARATION COMPLETED"
    )


if __name__ == "__main__":
    main()