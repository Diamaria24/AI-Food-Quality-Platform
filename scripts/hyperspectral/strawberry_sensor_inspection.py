import os
import pandas as pd
import numpy as np


# ======================================================================
# STRAWBERRY SENSOR DATASET INSPECTION
# ======================================================================

print("=" * 70)
print("STRAWBERRY SENSOR DATASET INSPECTION")
print("=" * 70)


# ======================================================================
# PATHS
# ======================================================================

PROJECT_ROOT = r"D:\Projects\AI-Food-Quality-Platform"

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "raw",
    "hyperspectral",
    "strawberry_sensor"
)


FILES = [
    "exp-1-plant.csv",
    "exp-2-plant.csv"
]


# ======================================================================
# PROCESS EACH FILE
# ======================================================================

for filename in FILES:

    filepath = os.path.join(
        DATA_DIR,
        filename
    )

    print("\n")
    print("=" * 70)
    print("FILE:", filename)
    print("=" * 70)

    print("\nPath:")
    print(filepath)

    if not os.path.exists(filepath):

        print("\nERROR: File not found.")
        continue

    # --------------------------------------------------------------
    # Load CSV
    # --------------------------------------------------------------

    print("\nLoading CSV...")

    df = pd.read_csv(
        filepath,
        low_memory=False
    )

    print("\nLoading completed.")

    # --------------------------------------------------------------
    # Shape
    # --------------------------------------------------------------

    print("\n" + "-" * 70)
    print("DATASET SHAPE")
    print("-" * 70)

    print(
        "\nRows:",
        df.shape[0]
    )

    print(
        "Columns:",
        df.shape[1]
    )

    # --------------------------------------------------------------
    # Column names
    # --------------------------------------------------------------

    print("\n" + "-" * 70)
    print("ALL COLUMNS")
    print("-" * 70)

    for i, column in enumerate(
        df.columns,
        start=1
    ):

        print(
            f"{i:3d}. {column}"
        )

    # --------------------------------------------------------------
    # Data types
    # --------------------------------------------------------------

    print("\n" + "-" * 70)
    print("DATA TYPES")
    print("-" * 70)

    print(
        df.dtypes.to_string()
    )

    # --------------------------------------------------------------
    # Missing values
    # --------------------------------------------------------------

    print("\n" + "-" * 70)
    print("MISSING VALUES")
    print("-" * 70)

    missing = df.isna().sum()

    missing = missing[
        missing > 0
    ].sort_values(
        ascending=False
    )

    if len(missing) == 0:

        print(
            "\nNo missing values found."
        )

    else:

        print(
            "\nColumns containing missing values:"
        )

        print(
            missing.to_string()
        )

    # --------------------------------------------------------------
    # Duplicate rows
    # --------------------------------------------------------------

    print("\n" + "-" * 70)
    print("DUPLICATE ROWS")
    print("-" * 70)

    duplicates = df.duplicated().sum()

    print(
        "\nDuplicate rows:",
        duplicates
    )

    # --------------------------------------------------------------
    # Numeric columns
    # --------------------------------------------------------------

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    print("\n" + "-" * 70)
    print("NUMERIC COLUMNS")
    print("-" * 70)

    print(
        "\nNumber of numeric columns:",
        len(numeric_columns)
    )

    for column in numeric_columns:

        print(
            column
        )

    # --------------------------------------------------------------
    # Non-numeric columns
    # --------------------------------------------------------------

    non_numeric_columns = df.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    print("\n" + "-" * 70)
    print("NON-NUMERIC COLUMNS")
    print("-" * 70)

    print(
        "\nNumber of non-numeric columns:",
        len(non_numeric_columns)
    )

    for column in non_numeric_columns:

        print(
            column
        )

    # --------------------------------------------------------------
    # Numeric summary
    # --------------------------------------------------------------

    print("\n" + "-" * 70)
    print("NUMERIC SUMMARY")
    print("-" * 70)

    if len(numeric_columns) > 0:

        summary = df[
            numeric_columns
        ].describe().T

        print(
            summary.to_string()
        )

    # --------------------------------------------------------------
    # Unique values in non-numeric columns
    # --------------------------------------------------------------

    print("\n" + "-" * 70)
    print("NON-NUMERIC COLUMN VALUES")
    print("-" * 70)

    for column in non_numeric_columns:

        unique_values = (
            df[column]
            .dropna()
            .unique()
        )

        print(
            f"\n{column}"
        )

        print(
            "Number of unique values:",
            len(unique_values)
        )

        if len(unique_values) <= 30:

            print(
                "Values:",
                unique_values
            )

        else:

            print(
                "First 30 values:",
                unique_values[:30]
            )

    # --------------------------------------------------------------
    # First rows
    # --------------------------------------------------------------

    print("\n" + "-" * 70)
    print("FIRST 10 ROWS")
    print("-" * 70)

    print(
        df.head(10).to_string()
    )

    # --------------------------------------------------------------
    # Last rows
    # --------------------------------------------------------------

    print("\n" + "-" * 70)
    print("LAST 5 ROWS")
    print("-" * 70)

    print(
        df.tail(5).to_string()
    )


# ======================================================================
# FINAL MESSAGE
# ======================================================================

print("\n")
print("=" * 70)
print("STRAWBERRY SENSOR INSPECTION COMPLETED")
print("=" * 70)

print("""
NEXT STEP:

Review the complete output before deciding:

    1. Target variable
    2. Sensor/input features
    3. Experiment combination strategy
    4. Train/test splitting strategy
    5. Leakage controls
    6. Classification or regression task
    7. Appropriate preprocessing
    8. Random Forest / SVM / XGBoost comparison

Do NOT train a model yet.
""")

print("=" * 70)