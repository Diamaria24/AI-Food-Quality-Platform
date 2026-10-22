"""
test_dataset_labels.py

Papaya dataset label inspection and mapping.

Purpose:
    1. Locate the Papaya dataset.
    2. Inspect CSV label information.
    3. Determine the unique Label_6Class values.
    4. Show class distributions.
    5. Inspect relationships between labels, Sample_ID,
       HDR filenames and subregions.
    6. Check whether the labels can be safely mapped
       to the hyperspectral samples.

IMPORTANT:
    This script does NOT assume what classes 1-6 mean.
    It only reports what is actually present in the dataset.
"""

import os
import glob
import pandas as pd
import numpy as np


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = r"D:\Projects\AI-Food-Quality-Platform"

PAPAYA_DIR = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "raw",
    "hyperspectral",
    "papaya_hsi"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("PAPAYA DATASET LABEL INSPECTION")
print("=" * 70)

print()
print("Dataset directory:")
print(PAPAYA_DIR)


# ============================================================
# CHECK DIRECTORY
# ============================================================

if not os.path.exists(PAPAYA_DIR):

    print()
    print("ERROR:")
    print("Papaya dataset directory was not found.")

    raise FileNotFoundError(
        PAPAYA_DIR
    )


# ============================================================
# LIST FILES
# ============================================================

print()
print("=" * 70)
print("PAPAYA DATASET FILES")
print("=" * 70)

all_files = []

for root, dirs, files in os.walk(PAPAYA_DIR):

    for file in files:

        full_path = os.path.join(
            root,
            file
        )

        all_files.append(full_path)


print()
print(
    "Total files found:",
    len(all_files)
)


for file in all_files[:100]:

    relative = os.path.relpath(
        file,
        PAPAYA_DIR
    )

    size_mb = (
        os.path.getsize(file)
        /
        (1024 * 1024)
    )

    print(
        f"{relative}  [{size_mb:.2f} MB]"
    )


if len(all_files) > 100:

    print()
    print(
        f"... and {len(all_files) - 100} more files"
    )


# ============================================================
# FIND CSV FILES
# ============================================================

print()
print("=" * 70)
print("CSV FILES")
print("=" * 70)

csv_files = glob.glob(
    os.path.join(
        PAPAYA_DIR,
        "**",
        "*.csv"
    ),
    recursive=True
)

print()
print(
    "CSV files found:",
    len(csv_files)
)

for csv_file in csv_files:

    print(
        "-",
        os.path.relpath(
            csv_file,
            PAPAYA_DIR
        )
    )


if not csv_files:

    raise FileNotFoundError(
        "No CSV file was found in the Papaya dataset."
    )


# ============================================================
# SELECT DATASET CSV
# ============================================================

dataset_csv = None

for csv_file in csv_files:

    filename = os.path.basename(
        csv_file
    ).lower()

    if (
        "directory" in filename
        or
        "dataset" in filename
        or
        "label" in filename
    ):

        dataset_csv = csv_file
        break


if dataset_csv is None:

    dataset_csv = csv_files[0]


print()
print(
    "Selected CSV:"
)

print(
    dataset_csv
)


# ============================================================
# LOAD CSV
# ============================================================

print()
print("=" * 70)
print("LOADING PAPAYA LABEL CSV")
print("=" * 70)

df = pd.read_csv(
    dataset_csv
)

print()
print(
    "Rows:",
    len(df)
)

print(
    "Columns:",
    len(df.columns)
)


# ============================================================
# COLUMN NAMES
# ============================================================

print()
print("=" * 70)
print("COLUMN NAMES")
print("=" * 70)

for column in df.columns:

    print(
        "-",
        column
    )


# ============================================================
# DATA TYPES
# ============================================================

print()
print("=" * 70)
print("DATA TYPES")
print("=" * 70)

print(
    df.dtypes
)


# ============================================================
# FIND LABEL COLUMNS
# ============================================================

print()
print("=" * 70)
print("POSSIBLE LABEL COLUMNS")
print("=" * 70)

label_columns = []

for column in df.columns:

    column_lower = str(
        column
    ).lower()

    if any(
        keyword in column_lower
        for keyword in [
            "label",
            "class",
            "condition",
            "disease",
            "quality"
        ]
    ):

        label_columns.append(
            column
        )


if label_columns:

    for column in label_columns:

        print(
            "-",
            column
        )

else:

    print(
        "No obvious label columns found."
    )


# ============================================================
# CHECK LABEL_6CLASS
# ============================================================

print()
print("=" * 70)
print("LABEL_6CLASS INSPECTION")
print("=" * 70)

if "Label_6Class" not in df.columns:

    print()
    print(
        "Label_6Class column was NOT found."
    )

    print()
    print(
        "Available columns:"
    )

    print(
        list(df.columns)
    )

else:

    labels = df["Label_6Class"]

    print()
    print(
        "Data type:",
        labels.dtype
    )

    print(
        "Missing labels:",
        labels.isna().sum()
    )

    print(
        "Unique labels:",
        sorted(
            labels.dropna()
            .unique()
            .tolist()
        )
    )

    print()
    print(
        "Number of unique classes:",
        labels.nunique()
    )


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

if "Label_6Class" in df.columns:

    print()
    print("=" * 70)
    print("PAPAYA CLASS DISTRIBUTION")
    print("=" * 70)

    class_counts = (
        df["Label_6Class"]
        .value_counts()
        .sort_index()
    )

    print()

    print(
        class_counts
        .to_string()
    )

    print()

    total = len(df)

    for label, count in class_counts.items():

        percentage = (
            count / total
        ) * 100

        print(
            f"Class {label}: "
            f"{count} samples "
            f"({percentage:.2f}%)"
        )


# ============================================================
# SAMPLE ID INSPECTION
# ============================================================

print()
print("=" * 70)
print("SAMPLE_ID INSPECTION")
print("=" * 70)

if "Sample_ID" in df.columns:

    print()
    print(
        "Unique Sample_ID values:",
        df["Sample_ID"].nunique()
    )

    print(
        "Minimum Sample_ID:",
        df["Sample_ID"].min()
    )

    print(
        "Maximum Sample_ID:",
        df["Sample_ID"].max()
    )

    duplicate_sample_ids = (
        df["Sample_ID"]
        .duplicated()
        .sum()
    )

    print(
        "Duplicate Sample_ID rows:",
        duplicate_sample_ids
    )

else:

    print(
        "Sample_ID column not found."
    )


# ============================================================
# LABEL ↔ SAMPLE_ID RELATIONSHIP
# ============================================================

if (
    "Label_6Class" in df.columns
    and
    "Sample_ID" in df.columns
):

    print()
    print("=" * 70)
    print("LABEL ↔ SAMPLE_ID RELATIONSHIP")
    print("=" * 70)

    mapping = (
        df[
            [
                "Sample_ID",
                "Label_6Class"
            ]
        ]
        .drop_duplicates()
        .sort_values(
            "Sample_ID"
        )
    )

    print()

    print(
        mapping
        .head(50)
        .to_string(
            index=False
        )
    )

    if len(mapping) > 50:

        print()
        print(
            f"... {len(mapping) - 50} more mappings"
        )


# ============================================================
# CHECK WHETHER EACH SAMPLE ID HAS ONE LABEL
# ============================================================

if (
    "Label_6Class" in df.columns
    and
    "Sample_ID" in df.columns
):

    print()
    print("=" * 70)
    print("SAMPLE_ID LABEL CONSISTENCY")
    print("=" * 70)

    labels_per_sample = (
        df.groupby(
            "Sample_ID"
        )["Label_6Class"]
        .nunique()
    )

    inconsistent = (
        labels_per_sample[
            labels_per_sample > 1
        ]
    )

    print()
    print(
        "Sample IDs inspected:",
        len(labels_per_sample)
    )

    print(
        "Sample IDs with multiple labels:",
        len(inconsistent)
    )

    if len(inconsistent) > 0:

        print()
        print(
            "WARNING: Some Sample_ID values "
            "have multiple Label_6Class values."
        )

        print()

        print(
            inconsistent
            .head(30)
            .to_string()
        )

    else:

        print()
        print(
            "Each Sample_ID has exactly one label."
        )


# ============================================================
# INSPECT SUBREGIONS
# ============================================================

print()
print("=" * 70)
print("SUBREGION INSPECTION")
print("=" * 70)

if "Subregions" in df.columns:

    print()
    print(
        "Unique Subregions:",
        df["Subregions"].nunique()
    )

    print()

    print(
        df["Subregions"]
        .value_counts()
        .head(30)
        .to_string()
    )

else:

    print(
        "Subregions column not found."
    )


# ============================================================
# LABEL ↔ SUBREGION RELATIONSHIP
# ============================================================

if (
    "Label_6Class" in df.columns
    and
    "Subregions" in df.columns
):

    print()
    print("=" * 70)
    print("LABEL ↔ SUBREGION RELATIONSHIP")
    print("=" * 70)

    relationship = (
        pd.crosstab(
            df["Label_6Class"],
            df["Subregions"]
        )
    )

    print()

    print(
        relationship
        .to_string()
    )


# ============================================================
# INSPECT HDR FILES
# ============================================================

print()
print("=" * 70)
print("PAPAYA HDR FILES")
print("=" * 70)

hdr_files = glob.glob(
    os.path.join(
        PAPAYA_DIR,
        "**",
        "*.hdr"
    ),
    recursive=True
)

print()
print(
    "HDR files found:",
    len(hdr_files)
)

for hdr_file in hdr_files[:100]:

    print(
        "-",
        os.path.relpath(
            hdr_file,
            PAPAYA_DIR
        )
    )


# ============================================================
# MATCH CSV HDR NAMES
# ============================================================

if "Subregions" in df.columns:

    print()
    print("=" * 70)
    print("CSV ↔ HDR FILE MATCHING")
    print("=" * 70)

    hdr_names = {
        os.path.basename(
            path
        ).lower()
        for path in hdr_files
    }

    csv_names = (
        df["Subregions"]
        .dropna()
        .astype(str)
        .str.lower()
        .unique()
    )

    matched = [
        name
        for name in csv_names
        if name in hdr_names
    ]

    unmatched = [
        name
        for name in csv_names
        if name not in hdr_names
    ]

    print()
    print(
        "Unique HDR names in CSV:",
        len(csv_names)
    )

    print(
        "Matched HDR files:",
        len(matched)
    )

    print(
        "Unmatched HDR names:",
        len(unmatched)
    )

    if unmatched:

        print()
        print(
            "First unmatched names:"
        )

        for name in unmatched[:30]:

            print(
                "-",
                name
            )


# ============================================================
# LABEL + HDR SUMMARY
# ============================================================

if (
    "Label_6Class" in df.columns
    and
    "Subregions" in df.columns
):

    print()
    print("=" * 70)
    print("LABEL ↔ HDR SUMMARY")
    print("=" * 70)

    summary = (
        df.groupby(
            [
                "Label_6Class",
                "Subregions"
            ]
        )
        .size()
        .reset_index(
            name="records"
        )
    )

    print()

    print(
        summary
        .head(100)
        .to_string(
            index=False
        )
    )


# ============================================================
# FIRST RECORDS
# ============================================================

print()
print("=" * 70)
print("FIRST 10 RECORDS")
print("=" * 70)

print()

print(
    df.head(10)
    .to_string(
        index=False
    )
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("PAPAYA LABEL INSPECTION SUMMARY")
print("=" * 70)

print()

if "Label_6Class" in df.columns:

    print(
        "Label column: Label_6Class"
    )

    print(
        "Number of classes:",
        df["Label_6Class"].nunique()
    )

    print(
        "Classes:",
        sorted(
            df["Label_6Class"]
            .dropna()
            .unique()
            .tolist()
        )
    )

else:

    print(
        "Label_6Class was not found."
    )


if "Sample_ID" in df.columns:

    print(
        "Unique Sample_IDs:",
        df["Sample_ID"].nunique()
    )


if "Subregions" in df.columns:

    print(
        "Unique Subregions:",
        df["Subregions"].nunique()
    )


print()
print(
    "IMPORTANT:"
)

print(
    "No class meaning has been assumed."
)

print(
    "Use this output to determine the actual "
    "meaning of Label_6Class before classification."
)

print()
print("=" * 70)
print("PAPAYA LABEL INSPECTION COMPLETED")
print("=" * 70)