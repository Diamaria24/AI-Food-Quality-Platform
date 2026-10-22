import os
import glob
import re

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


# ================================================================
# CONFIGURATION
# ================================================================

PROJECT_ROOT = r"D:\Projects\AI-Food-Quality-Platform"

PISTACHIO_DIR = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "raw",
    "hyperspectral",
    "pistachio_hsi"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "processed",
    "hyperspectral",
    "pistachio_classification"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# Project classification threshold
THRESHOLD_UG_KG = 8.0

RANDOM_STATE = 42

N_SPLITS = 5


# ================================================================
# GROUND-TRUTH MAPPING
# ================================================================
#
# Values supplied by the HyperPistachio dataset.
#
# Level 01 is the healthy control.
#
# Binary classification:
#
#   AFB1 < 8.0       -> 0 = Below_Threshold
#   AFB1 >= 8.0      -> 1 = At_or_Above_Threshold
#
# ================================================================

AFB1_LEVELS = {

    1: 0.00,
    2: 0.40,
    3: 0.67,
    4: 0.88,
    5: 1.13,
    6: 1.66,
    7: 2.15,
    8: 2.30,
    9: 2.48,
    10: 2.82,
    11: 3.01,
    12: 3.05,
    13: 3.85,
    14: 4.43,
    15: 5.12,
    16: 5.30,
    17: 6.37,
    18: 8.93,
    19: 12.16,
    20: 17.12,
    21: 24.03,
    22: 26.14,
    23: 33.17,
    24: 56.06,
    25: 57.29,
    26: 114.67
}


# ================================================================
# HELPER
# ================================================================

def get_level_number(folder_name):

    match = re.search(
        r"(\d+)",
        folder_name
    )

    if not match:
        raise ValueError(
            f"Unable to determine level from: {folder_name}"
        )

    return int(
        match.group(1)
    )


# ================================================================
# HEADER
# ================================================================

print("=" * 70)
print("PISTACHIO HSI GROUND-TRUTH + GROUP SPLIT")
print("=" * 70)

print(
    "\nDataset:"
)

print(
    PISTACHIO_DIR
)

print(
    "\nAFB1 threshold:",
    THRESHOLD_UG_KG,
    "µg/kg"
)


# ================================================================
# FIND LEVEL FOLDERS
# ================================================================

level_folders = []

for name in os.listdir(
    PISTACHIO_DIR
):

    path = os.path.join(
        PISTACHIO_DIR,
        name
    )

    if os.path.isdir(path):

        match = re.search(
            r"(\d+)",
            name
        )

        if match:

            level_number = int(
                match.group(1)
            )

            level_folders.append(
                (
                    level_number,
                    name
                )
            )


level_folders.sort()


print("\n" + "=" * 70)
print("LEVEL VALIDATION")
print("=" * 70)

print(
    "\nLevels found:",
    len(level_folders)
)

expected_levels = set(
    range(1, 27)
)

found_levels = {
    number
    for number, name
    in level_folders
}

missing_levels = (
    expected_levels
    - found_levels
)

extra_levels = (
    found_levels
    - expected_levels
)

if missing_levels:

    print(
        "\nMissing levels:",
        sorted(missing_levels)
    )

else:

    print(
        "\nAll Level 01–Level 26 folders found."
    )

if extra_levels:

    print(
        "Unexpected levels:",
        sorted(extra_levels)
    )


# ================================================================
# CREATE SAMPLE MANIFEST
# ================================================================

print("\n" + "=" * 70)
print("CREATING SAMPLE MANIFEST")
print("=" * 70)


records = []


for level_number, level_folder in level_folders:

    if level_number not in AFB1_LEVELS:

        raise ValueError(
            f"No AFB1 ground truth for Level {level_number}"
        )

    afb1_value = AFB1_LEVELS[
        level_number
    ]

    if afb1_value < THRESHOLD_UG_KG:

        binary_label = 0

        binary_class = (
            "Below_Threshold"
        )

    else:

        binary_label = 1

        binary_class = (
            "At_or_Above_Threshold"
        )


    level_path = os.path.join(
        PISTACHIO_DIR,
        level_folder
    )


    bil_files = sorted(
        glob.glob(
            os.path.join(
                level_path,
                "*.bil"
            )
        )
    )


    hdr_files = sorted(
        glob.glob(
            os.path.join(
                level_path,
                "*.hdr"
            )
        )
    )


    png_files = sorted(
        glob.glob(
            os.path.join(
                level_path,
                "*.png"
            )
        )
    )


    # ------------------------------------------------------------
    # Match BIL / HDR / PNG by sample name
    # ------------------------------------------------------------

    bil_dict = {
        os.path.splitext(
            os.path.basename(path)
        )[0]: path

        for path in bil_files
    }

    hdr_dict = {
        os.path.splitext(
            os.path.basename(path)
        )[0]: path

        for path in hdr_files
    }

    png_dict = {
        os.path.splitext(
            os.path.basename(path)
        )[0]: path

        for path in png_files
    }


    sample_names = sorted(
        set(bil_dict)
        |
        set(hdr_dict)
        |
        set(png_dict)
    )


    for sample_name in sample_names:

        if sample_name not in bil_dict:

            raise ValueError(
                f"Missing BIL for {sample_name}"
            )

        if sample_name not in hdr_dict:

            raise ValueError(
                f"Missing HDR for {sample_name}"
            )

        if sample_name not in png_dict:

            raise ValueError(
                f"Missing PNG for {sample_name}"
            )


        records.append({

            "Sample_ID":
                sample_name,

            "Level":
                level_number,

            "Level_Name":
                level_folder,

            "AFB1_ug_kg":
                afb1_value,

            "Binary_Label":
                binary_label,

            "Binary_Class":
                binary_class,

            "BIL_Path":
                bil_dict[sample_name],

            "HDR_Path":
                hdr_dict[sample_name],

            "PNG_Path":
                png_dict[sample_name],

            # IMPORTANT:
            # Both samples from the same contamination
            # level remain grouped together.
            "Group_ID":
                level_number
        })


manifest = pd.DataFrame(
    records
)


# ================================================================
# MANIFEST VALIDATION
# ================================================================

print("\n" + "=" * 70)
print("MANIFEST VALIDATION")
print("=" * 70)

print(
    "\nTotal HSI samples:",
    len(manifest)
)

print(
    "Unique levels:",
    manifest["Level"].nunique()
)

print(
    "Unique Sample_IDs:",
    manifest["Sample_ID"].nunique()
)

print(
    "Unique Group_IDs:",
    manifest["Group_ID"].nunique()
)


# ------------------------------------------------------------
# Check duplicate sample IDs
# ------------------------------------------------------------

duplicate_ids = manifest[
    manifest["Sample_ID"].duplicated(
        keep=False
    )
]

if len(duplicate_ids) > 0:

    raise ValueError(
        "Duplicate Sample_IDs detected."
    )

else:

    print(
        "\nSample_ID uniqueness: PASS"
    )


# ------------------------------------------------------------
# Check exactly 2 samples per level
# ------------------------------------------------------------

samples_per_level = (
    manifest
    .groupby("Level")
    .size()
)


print(
    "\nSamples per level:"
)

print(
    samples_per_level
)


if not all(
    samples_per_level == 2
):

    raise ValueError(
        "Not every level contains exactly 2 samples."
    )

else:

    print(
        "\nExactly 2 samples per level: PASS"
    )


# ================================================================
# GROUND-TRUTH TABLE
# ================================================================

print("\n" + "=" * 70)
print("GROUND-TRUTH MAPPING")
print("=" * 70)

ground_truth = (
    manifest[
        [
            "Level",
            "AFB1_ug_kg",
            "Binary_Label",
            "Binary_Class"
        ]
    ]
    .drop_duplicates()
    .sort_values("Level")
)


print(
    ground_truth.to_string(
        index=False
    )
)


# ================================================================
# BINARY CLASS DISTRIBUTION
# ================================================================

print("\n" + "=" * 70)
print("BINARY CLASS DISTRIBUTION")
print("=" * 70)


class_distribution = (
    manifest[
        [
            "Binary_Label",
            "Binary_Class"
        ]
    ]
    .drop_duplicates(
        subset=[
            "Binary_Label",
            "Binary_Class"
        ]
    )
)


print(
    "\nLevel distribution:"
)

print(
    manifest
    .groupby(
        [
            "Binary_Label",
            "Binary_Class"
        ]
    )["Level"]
    .nunique()
)


print(
    "\nCube distribution:"
)

print(
    manifest[
        [
            "Binary_Label",
            "Binary_Class"
        ]
    ]
    .value_counts()
    .sort_index()
)


# ================================================================
# SAVE GROUND-TRUTH MANIFEST
# ================================================================

manifest_path = os.path.join(
    OUTPUT_DIR,
    "pistachio_ground_truth_manifest.csv"
)

manifest.to_csv(
    manifest_path,
    index=False
)


print(
    "\nSaved ground-truth manifest:"
)

print(
    manifest_path
)


# ================================================================
# STRATIFIED GROUP K-FOLD ANALYSIS
# ================================================================

print("\n" + "=" * 70)
print("5-FOLD STRATIFIED GROUP SPLIT ANALYSIS")
print("=" * 70)

print(
"""
Important:

Group_ID = contamination Level.

Both HSI cubes from the same level must remain
in the same fold.

This prevents the same contamination level from
being represented in both training and validation
during the group-level split analysis.
"""
)


X_dummy = manifest[
    [
        "Sample_ID"
    ]
]

y = manifest[
    "Binary_Label"
]

groups = manifest[
    "Group_ID"
]


cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)


fold_records = []


for fold_number, (
    train_index,
    test_index
) in enumerate(
    cv.split(
        X_dummy,
        y,
        groups
    ),
    start=1
):

    train_df = manifest.iloc[
        train_index
    ]

    test_df = manifest.iloc[
        test_index
    ]


    train_groups = set(
        train_df["Group_ID"]
    )

    test_groups = set(
        test_df["Group_ID"]
    )


    overlap = (
        train_groups
        &
        test_groups
    )


    print(
        "\n" + "-" * 70
    )

    print(
        f"FOLD {fold_number}"
    )

    print(
        "-" * 70
    )

    print(
        "\nTraining cubes:",
        len(train_df)
    )

    print(
        "Testing cubes :",
        len(test_df)
    )

    print(
        "Training levels:",
        len(train_groups)
    )

    print(
        "Testing levels :",
        len(test_groups)
    )

    print(
        "Group overlap:",
        len(overlap)
    )


    if overlap:

        raise ValueError(
            f"Group leakage detected in fold {fold_number}: "
            f"{sorted(overlap)}"
        )


    train_class_counts = (
        train_df[
            "Binary_Class"
        ]
        .value_counts()
        .sort_index()
    )

    test_class_counts = (
        test_df[
            "Binary_Class"
        ]
        .value_counts()
        .sort_index()
    )


    print(
        "\nTraining class distribution:"
    )

    print(
        train_class_counts
    )


    print(
        "\nTesting class distribution:"
    )

    print(
        test_class_counts
    )


    fold_records.append({

        "fold":
            fold_number,

        "training_samples":
            len(train_df),

        "testing_samples":
            len(test_df),

        "training_levels":
            len(train_groups),

        "testing_levels":
            len(test_groups),

        "group_overlap":
            len(overlap),

        "training_below_threshold":
            int(
                (
                    train_df[
                        "Binary_Label"
                    ] == 0
                ).sum()
            ),

        "training_at_or_above_threshold":
            int(
                (
                    train_df[
                        "Binary_Label"
                    ] == 1
                ).sum()
            ),

        "testing_below_threshold":
            int(
                (
                    test_df[
                        "Binary_Label"
                    ] == 0
                ).sum()
            ),

        "testing_at_or_above_threshold":
            int(
                (
                    test_df[
                        "Binary_Label"
                    ] == 1
                ).sum()
            )
    })


# ================================================================
# SELECT FINAL FOLD
# ================================================================
#
# We choose the fold with the closest overall test size to 20%
# and balanced representation of the two binary groups.
#
# ================================================================

fold_df = pd.DataFrame(
    fold_records
)


fold_df["test_fraction"] = (
    fold_df["testing_samples"]
    /
    len(manifest)
)


fold_df["class_difference"] = (
    abs(
        fold_df[
            "testing_below_threshold"
        ]
        -
        fold_df[
            "testing_at_or_above_threshold"
        ]
    )
)


# We prioritize:
# 1. test fraction close to 20%
# 2. smaller class imbalance

fold_df["selection_score"] = (
    abs(
        fold_df[
            "test_fraction"
        ]
        - 0.20
    )
    +
    (
        fold_df[
            "class_difference"
        ]
        * 0.01
    )
)


best_fold = (
    fold_df
    .sort_values(
        [
            "selection_score",
            "class_difference"
        ]
    )
    .iloc[0]
)


selected_fold = int(
    best_fold["fold"]
)


print("\n" + "=" * 70)
print("FINAL FOLD SELECTION")
print("=" * 70)

print(
    "\nSelected fold:",
    selected_fold
)


# ================================================================
# RECREATE SELECTED FOLD
# ================================================================

selected_train_index = None
selected_test_index = None


for fold_number, (
    train_index,
    test_index
) in enumerate(
    cv.split(
        X_dummy,
        y,
        groups
    ),
    start=1
):

    if fold_number == selected_fold:

        selected_train_index = (
            train_index
        )

        selected_test_index = (
            test_index
        )

        break


train_df = manifest.iloc[
    selected_train_index
].copy()


test_df = manifest.iloc[
    selected_test_index
].copy()


# ================================================================
# FINAL LEAKAGE CHECK
# ================================================================

train_groups = set(
    train_df["Group_ID"]
)

test_groups = set(
    test_df["Group_ID"]
)

group_overlap = (
    train_groups
    &
    test_groups
)


print("\n" + "=" * 70)
print("FINAL LEAKAGE CHECK")
print("=" * 70)

print(
    "\nTraining samples:",
    len(train_df)
)

print(
    "Testing samples :",
    len(test_df)
)

print(
    "Training levels:",
    len(train_groups)
)

print(
    "Testing levels :",
    len(test_groups)
)

print(
    "Overlapping levels:",
    len(group_overlap)
)


if group_overlap:

    raise ValueError(
        "FINAL SPLIT FAILED: contamination levels overlap."
    )

else:

    print(
        "\nPASS: No contamination-level overlap."
    )


# ================================================================
# FINAL CLASS DISTRIBUTION
# ================================================================

print("\n" + "=" * 70)
print("FINAL TRAIN / TEST CLASS DISTRIBUTION")
print("=" * 70)


print(
    "\nTraining:"
)

print(
    train_df[
        "Binary_Class"
    ].value_counts()
)


print(
    "\nTesting:"
)

print(
    test_df[
        "Binary_Class"
    ].value_counts()
)


print(
    "\nTraining levels:"
)

print(
    sorted(
        train_df[
            "Level"
        ].unique()
    )
)


print(
    "\nTesting levels:"
)

print(
    sorted(
        test_df[
            "Level"
        ].unique()
    )
)


# ================================================================
# SAVE FINAL SPLITS
# ================================================================

train_path = os.path.join(
    OUTPUT_DIR,
    "pistachio_train_group_split.csv"
)

test_path = os.path.join(
    OUTPUT_DIR,
    "pistachio_test_group_split.csv"
)

folds_path = os.path.join(
    OUTPUT_DIR,
    "pistachio_group_split_folds.csv"
)


train_df.to_csv(
    train_path,
    index=False
)

test_df.to_csv(
    test_path,
    index=False
)

fold_df.to_csv(
    folds_path,
    index=False
)


# ================================================================
# FINAL SUMMARY
# ================================================================

print("\n" + "=" * 70)
print("PISTACHIO GROUP SPLIT COMPLETED")
print("=" * 70)

print(
f"""
AFB1 threshold             : {THRESHOLD_UG_KG} µg/kg

Total HSI cubes            : {len(manifest)}

Total contamination levels : {manifest['Level'].nunique()}

Training cubes             : {len(train_df)}
Testing cubes              : {len(test_df)}

Training levels            : {len(train_groups)}
Testing levels             : {len(test_groups)}

Overlapping levels         : {len(group_overlap)}

Selected fold              : {selected_fold}

Ground truth:
    < 8.0 µg/kg
        -> Below_Threshold

    >= 8.0 µg/kg
        -> At_or_Above_Threshold

Saved:

{manifest_path}

{train_path}

{test_path}

{folds_path}
"""
)

print("=" * 70)

print(
"""
NEXT STEP:

HSI preprocessing
    ->
reflectance conversion
    ->
spectral normalization
    ->
PCA

No classifier has been trained yet.
"""
)