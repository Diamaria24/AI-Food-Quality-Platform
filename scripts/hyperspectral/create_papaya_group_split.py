import os
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold


# ================================================================
# PAPAYA LEAKAGE-SAFE GROUP SPLIT
# ================================================================

print("=" * 70)
print("PAPAYA HSI LEAKAGE-SAFE GROUP SPLIT")
print("=" * 70)


# ----------------------------------------------------------------
# PATHS
# ----------------------------------------------------------------

CSV_PATH = (
    r"D:\Projects\AI-Food-Quality-Platform\datasets\raw"
    r"\hyperspectral\papaya_hsi"
    r"\Garillos-Manliguez_Hyperspectral Images of Papaya"
    r"\hsi32_dataset_example files directory.csv"
)

OUTPUT_DIR = (
    r"D:\Projects\AI-Food-Quality-Platform"
    r"\datasets\processed\hyperspectral"
    r"\papaya_classification"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ----------------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------------

df = pd.read_csv(CSV_PATH)

print("\nRows:", len(df))
print("Unique Sample_IDs:", df["Sample_ID"].nunique())
print("Classes:", sorted(df["Label_6Class"].unique()))


# ----------------------------------------------------------------
# STRATIFIED GROUP K-FOLD
#
# Group = Sample_ID
# Target = Label_6Class
#
# This prevents the same Sample_ID from appearing in both
# training and testing.
# ----------------------------------------------------------------

X_dummy = df.drop(columns=["Label_6Class"])

y = df["Label_6Class"]

groups = df["Sample_ID"]


cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ----------------------------------------------------------------
# EXAMINE ALL 5 FOLDS
# ----------------------------------------------------------------

print("\n" + "=" * 70)
print("5-FOLD GROUP SPLIT ANALYSIS")
print("=" * 70)


fold_results = []

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X_dummy, y, groups),
    start=1
):

    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    train_groups = set(train_df["Sample_ID"])
    test_groups = set(test_df["Sample_ID"])

    overlap = train_groups.intersection(test_groups)

    print(f"\n{'-' * 70}")
    print(f"FOLD {fold}")
    print(f"{'-' * 70}")

    print("Training rows:", len(train_df))
    print("Testing rows :", len(test_df))

    print(
        "Training Sample_IDs:",
        len(train_groups)
    )

    print(
        "Testing Sample_IDs :",
        len(test_groups)
    )

    print(
        "Group overlap:",
        len(overlap)
    )

    print("\nTraining class distribution:")
    print(
        train_df["Label_6Class"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nTesting class distribution:")
    print(
        test_df["Label_6Class"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    fold_results.append(
        {
            "fold": fold,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "train_groups": len(train_groups),
            "test_groups": len(test_groups),
            "group_overlap": len(overlap)
        }
    )


# ----------------------------------------------------------------
# USE FOLD 1 AS FINAL HOLDOUT
# ----------------------------------------------------------------

print("\n" + "=" * 70)
print("CREATING FINAL TRAIN / TEST SPLIT")
print("=" * 70)


splits = list(
    cv.split(X_dummy, y, groups)
)

train_idx, test_idx = splits[0]


train_df = df.iloc[train_idx].copy()
test_df = df.iloc[test_idx].copy()


# ----------------------------------------------------------------
# VERIFY NO GROUP LEAKAGE
# ----------------------------------------------------------------

train_groups = set(train_df["Sample_ID"])
test_groups = set(test_df["Sample_ID"])

overlap = train_groups.intersection(test_groups)


print("\nFinal split:")

print("Training rows:", len(train_df))
print("Testing rows :", len(test_df))

print(
    "Training Sample_IDs:",
    len(train_groups)
)

print(
    "Testing Sample_IDs:",
    len(test_groups)
)

print(
    "Overlapping Sample_IDs:",
    overlap
)


if len(overlap) == 0:
    print("\nPASS: No Sample_ID leakage.")
else:
    print("\nERROR: Sample_ID leakage detected!")


# ----------------------------------------------------------------
# CLASS DISTRIBUTION
# ----------------------------------------------------------------

print("\nTraining class distribution:")
print(
    train_df["Label_6Class"]
    .value_counts()
    .sort_index()
)

print("\nTesting class distribution:")
print(
    test_df["Label_6Class"]
    .value_counts()
    .sort_index()
)


# ----------------------------------------------------------------
# SAVE SPLITS
# ----------------------------------------------------------------

train_path = os.path.join(
    OUTPUT_DIR,
    "papaya_train_split.csv"
)

test_path = os.path.join(
    OUTPUT_DIR,
    "papaya_test_split.csv"
)


train_df.to_csv(
    train_path,
    index=False
)

test_df.to_csv(
    test_path,
    index=False
)


# ----------------------------------------------------------------
# SAVE FOLD SUMMARY
# ----------------------------------------------------------------

fold_summary_path = os.path.join(
    OUTPUT_DIR,
    "papaya_group_split_folds.csv"
)

pd.DataFrame(
    fold_results
).to_csv(
    fold_summary_path,
    index=False
)


# ----------------------------------------------------------------
# FINAL
# ----------------------------------------------------------------

print("\n" + "=" * 70)
print("PAPAYA GROUP SPLIT COMPLETED")
print("=" * 70)

print("\nSaved:")
print(train_path)
print(test_path)
print(fold_summary_path)

print("\nNext step:")
print("HSI preprocessing → spectral normalization → PCA")