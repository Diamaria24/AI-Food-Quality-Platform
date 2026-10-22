import os
import re
import pandas as pd
from collections import Counter

# ================================================================
# PAPAYA HSI SAMPLE ↔ LABEL MAPPING INSPECTION
# ================================================================

print("=" * 70)
print("PAPAYA HSI SAMPLE ↔ LABEL MAPPING INSPECTION")
print("=" * 70)

# ----------------------------------------------------------------
# PATH
# ----------------------------------------------------------------

CSV_PATH = (
    r"D:\Projects\AI-Food-Quality-Platform\datasets\raw"
    r"\hyperspectral\papaya_hsi"
    r"\Garillos-Manliguez_Hyperspectral Images of Papaya"
    r"\hsi32_dataset_example files directory.csv"
)

print("\nCSV:")
print(CSV_PATH)

if not os.path.exists(CSV_PATH):
    print("\nERROR: CSV file not found.")
    print("Check the CSV path above.")
    raise FileNotFoundError(CSV_PATH)

# ----------------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------------

df = pd.read_csv(CSV_PATH)

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print("Rows :", len(df))
print("Columns:", list(df.columns))

required_columns = [
    "Subregions",
    "RGB_Filename",
    "Header_Cropped",
    "Sample_ID",
    "Label_6Class",
    "Subregions_Img"
]

missing = [c for c in required_columns if c not in df.columns]

if missing:
    print("\nERROR: Missing columns:")
    print(missing)
    raise ValueError("Required columns are missing.")

print("\nAll required columns found.")

# ----------------------------------------------------------------
# BASIC LABEL INFORMATION
# ----------------------------------------------------------------

print("\n" + "=" * 70)
print("LABEL DISTRIBUTION")
print("=" * 70)

label_counts = df["Label_6Class"].value_counts().sort_index()

print(label_counts)

print("\nNumber of classes:", df["Label_6Class"].nunique())
print("Classes:", sorted(df["Label_6Class"].unique()))

# ----------------------------------------------------------------
# SAMPLE INFORMATION
# ----------------------------------------------------------------

print("\n" + "=" * 70)
print("SAMPLE INFORMATION")
print("=" * 70)

print("Unique Sample_IDs:", df["Sample_ID"].nunique())
print("Unique Subregions:", df["Subregions"].nunique())

sample_counts = (
    df.groupby("Sample_ID")
      .size()
      .sort_index()
)

print("\nSubregions per Sample_ID:")
print(sample_counts.to_string())

# ----------------------------------------------------------------
# CHECK WHETHER EACH SAMPLE HAS ONLY ONE LABEL
# ----------------------------------------------------------------

print("\n" + "=" * 70)
print("SAMPLE_ID ↔ LABEL CONSISTENCY")
print("=" * 70)

sample_label_counts = (
    df.groupby("Sample_ID")["Label_6Class"]
      .nunique()
)

inconsistent_samples = sample_label_counts[
    sample_label_counts > 1
]

if len(inconsistent_samples) == 0:
    print("PASS: Every Sample_ID has exactly one Label_6Class.")
else:
    print("WARNING: Some Sample_IDs contain multiple labels.")

    print("\nInconsistent Sample_IDs:")
    print(inconsistent_samples.to_string())

    for sample_id in inconsistent_samples.index:
        subset = df[df["Sample_ID"] == sample_id]

        print("\nSample_ID:", sample_id)
        print(
            subset[
                ["Subregions", "Header_Cropped", "Label_6Class"]
            ].to_string(index=False)
        )

# ----------------------------------------------------------------
# SAMPLE_ID → LABEL TABLE
# ----------------------------------------------------------------

print("\n" + "=" * 70)
print("SAMPLE_ID → LABEL MAPPING")
print("=" * 70)

sample_mapping = (
    df.groupby("Sample_ID")["Label_6Class"]
      .agg(["nunique", "first", "count"])
      .reset_index()
)

sample_mapping.columns = [
    "Sample_ID",
    "Number_of_Labels",
    "Label",
    "Subregion_Count"
]

print(sample_mapping.to_string(index=False))

# ----------------------------------------------------------------
# LABEL → SAMPLE_ID GROUPS
# ----------------------------------------------------------------

print("\n" + "=" * 70)
print("LABEL → SAMPLE_ID GROUPS")
print("=" * 70)

for label in sorted(df["Label_6Class"].unique()):

    sample_ids = sorted(
        df.loc[
            df["Label_6Class"] == label,
            "Sample_ID"
        ].unique()
    )

    print(f"\nLabel {label}")
    print("Number of samples:", len(sample_ids))
    print("Sample IDs:", sample_ids)

# ----------------------------------------------------------------
# CHECK FIRST NUMBER IN SUBREGION NAME
#
# Example:
# 3-29-a-s4.hdr → prefix = 3
# 6-62-b-s2.hdr → prefix = 6
# ----------------------------------------------------------------

print("\n" + "=" * 70)
print("SUBREGION PREFIX ↔ LABEL CHECK")
print("=" * 70)

def extract_prefix(filename):

    if pd.isna(filename):
        return None

    filename = str(filename)

    match = re.match(r"^(\d+)-", filename)

    if match:
        return int(match.group(1))

    return None


df["Subregion_Prefix"] = df["Subregions"].apply(
    extract_prefix
)

prefix_missing = df["Subregion_Prefix"].isna().sum()

print("Rows with missing prefix:", prefix_missing)

prefix_label_check = (
    df.groupby(["Subregion_Prefix", "Label_6Class"])
      .size()
      .reset_index(name="Count")
)

print("\nPrefix ↔ Label counts:")
print(prefix_label_check.to_string(index=False))

# ----------------------------------------------------------------
# CHECK WHETHER PREFIX == LABEL
# ----------------------------------------------------------------

valid_prefix_rows = df["Subregion_Prefix"].notna()

prefix_matches = (
    df.loc[valid_prefix_rows, "Subregion_Prefix"]
    ==
    df.loc[valid_prefix_rows, "Label_6Class"]
)

print("\nPrefix == Label matches:",
      prefix_matches.sum())

print("Prefix != Label mismatches:",
      (~prefix_matches).sum())

if prefix_matches.all():
    print("\nPASS:")
    print("The first number in the Subregions filename")
    print("matches Label_6Class for every inspected row.")
else:
    print("\nWARNING:")
    print("The first number in the Subregions filename")
    print("does NOT always match Label_6Class.")

    mismatches = df.loc[
        valid_prefix_rows & (~prefix_matches),
        [
            "Subregions",
            "Sample_ID",
            "Label_6Class",
            "Subregion_Prefix"
        ]
    ]

    print("\nMismatched rows:")
    print(mismatches.head(100).to_string(index=False))

# ----------------------------------------------------------------
# LABEL × SAMPLE COUNT
# ----------------------------------------------------------------

print("\n" + "=" * 70)
print("LABEL × SAMPLE COUNT")
print("=" * 70)

label_sample_table = (
    df.groupby("Label_6Class")["Sample_ID"]
      .nunique()
      .sort_index()
)

print(label_sample_table)

# ----------------------------------------------------------------
# SAVE SAMPLE MAPPING
# ----------------------------------------------------------------

OUTPUT_DIR = (
    r"D:\Projects\AI-Food-Quality-Platform"
    r"\datasets\processed\hyperspectral"
    r"\papaya_classification"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

mapping_path = os.path.join(
    OUTPUT_DIR,
    "papaya_sample_label_mapping.csv"
)

sample_mapping.to_csv(
    mapping_path,
    index=False
)

print("\nSaved sample mapping:")
print(mapping_path)

# ----------------------------------------------------------------
# SAVE PREFIX CHECK
# ----------------------------------------------------------------

prefix_path = os.path.join(
    OUTPUT_DIR,
    "papaya_prefix_label_check.csv"
)

prefix_label_check.to_csv(
    prefix_path,
    index=False
)

print("\nSaved prefix-label check:")
print(prefix_path)

# ----------------------------------------------------------------
# FINAL SUMMARY
# ----------------------------------------------------------------

print("\n" + "=" * 70)
print("PAPAYA SAMPLE ↔ LABEL INSPECTION COMPLETED")
print("=" * 70)

print("\nRows:", len(df))
print("Classes:", sorted(df["Label_6Class"].unique()))
print("Unique Sample_IDs:", df["Sample_ID"].nunique())
print("Unique Subregions:", df["Subregions"].nunique())

if len(inconsistent_samples) == 0:
    print("Sample_ID label consistency: PASS")
else:
    print("Sample_ID label consistency: FAIL")

if prefix_matches.all():
    print("Filename prefix ↔ label consistency: PASS")
else:
    print("Filename prefix ↔ label consistency: CHECK REQUIRED")

print("\nOutputs:")
print(mapping_path)
print(prefix_path)

print("\n" + "=" * 70)