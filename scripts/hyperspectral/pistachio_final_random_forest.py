import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt


# ======================================================================
# PISTACHIO HSI FINAL RANDOM FOREST CLASSIFICATION
# ======================================================================

print("=" * 70)
print("PISTACHIO HSI FINAL RANDOM FOREST CLASSIFICATION")
print("=" * 70)


# ======================================================================
# PROJECT PATHS
# ======================================================================

PROJECT_ROOT = r"D:\Projects\AI-Food-Quality-Platform"

PCA_DIR = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "processed",
    "hyperspectral",
    "pistachio_classification",
    "pca",
    "pca_100pc"
)


TRAIN_FILE = os.path.join(
    PCA_DIR,
    "pistachio_train_100pc.csv"
)


TEST_FILE = os.path.join(
    PCA_DIR,
    "pistachio_test_100pc.csv"
)


OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "processed",
    "hyperspectral",
    "pistachio_classification",
    "pca",
    "pca_100pc"
)


GRAPH_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "graphs",
    "pistachio_classification"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    GRAPH_DIR,
    exist_ok=True
)


# ======================================================================
# RANDOM FOREST CONFIGURATION
# ======================================================================

RANDOM_STATE = 42

N_ESTIMATORS = 300

N_JOBS = -1


# ======================================================================
# LOAD DATA
# ======================================================================

print("\n" + "=" * 70)
print("LOADING PISTACHIO 100-PC CLASSIFICATION DATA")
print("=" * 70)


print(
    "\nTraining file:"
)

print(
    TRAIN_FILE
)


print(
    "\nTesting file:"
)

print(
    TEST_FILE
)


if not os.path.exists(
    TRAIN_FILE
):

    raise FileNotFoundError(
        f"Training file not found:\n{TRAIN_FILE}"
    )


if not os.path.exists(
    TEST_FILE
):

    raise FileNotFoundError(
        f"Testing file not found:\n{TEST_FILE}"
    )


train_df = pd.read_csv(
    TRAIN_FILE
)


test_df = pd.read_csv(
    TEST_FILE
)


print(
    "\nTraining shape :",
    train_df.shape
)


print(
    "Testing shape  :",
    test_df.shape
)


# ======================================================================
# PCA FEATURES
# ======================================================================

print("\n" + "=" * 70)
print("PCA FEATURES")
print("=" * 70)


pc_columns = [

    column

    for column in train_df.columns

    if column.startswith("PC")
]


# Sort PC columns numerically.
pc_columns = sorted(
    pc_columns,
    key=lambda x: int(
        x.replace("PC", "")
    )
)


print(
    "\nNumber of PCA components:",
    len(pc_columns)
)


print(
    "\nPCA features:"
)

print(
    pc_columns
)


if len(pc_columns) != 100:

    raise ValueError(
        f"Expected 100 PCA components, "
        f"found {len(pc_columns)}."
    )


# ======================================================================
# COLUMN VALIDATION
# ======================================================================

print("\n" + "=" * 70)
print("COLUMN VALIDATION")
print("=" * 70)


required_columns = [

    "Sample_ID",
    "Level",
    "AFB1_ug_kg",
    "Binary_Label",
    "Binary_Class",
    "Group_ID"
]


for column in required_columns:

    if column not in train_df.columns:

        raise ValueError(
            f"Missing column in training data: "
            f"{column}"
        )


    if column not in test_df.columns:

        raise ValueError(
            f"Missing column in testing data: "
            f"{column}"
        )


for column in pc_columns:

    if column not in test_df.columns:

        raise ValueError(
            f"Missing PCA column in testing data: "
            f"{column}"
        )


print(
    "\nAll required columns found."
)


# ======================================================================
# SAMPLE_ID LEAKAGE CHECK
# ======================================================================

print("\n" + "=" * 70)
print("SAMPLE_ID LEAKAGE CHECK")
print("=" * 70)


train_sample_ids = set(
    train_df[
        "Sample_ID"
    ]
)


test_sample_ids = set(
    test_df[
        "Sample_ID"
    ]
)


sample_overlap = (
    train_sample_ids
    &
    test_sample_ids
)


print(
    "\nTraining Sample_IDs:",
    len(train_sample_ids)
)


print(
    "Testing Sample_IDs :",
    len(test_sample_ids)
)


print(
    "Overlapping Sample_IDs:",
    len(sample_overlap)
)


if sample_overlap:

    raise ValueError(
        "Sample_ID leakage detected:\n"
        f"{sorted(sample_overlap)}"
    )


print(
    "\nSample_ID leakage check: PASS"
)


# ======================================================================
# GROUP_ID / CONTAMINATION LEVEL LEAKAGE CHECK
# ======================================================================

print("\n" + "=" * 70)
print("CONTAMINATION-LEVEL LEAKAGE CHECK")
print("=" * 70)


train_groups = set(
    train_df[
        "Group_ID"
    ]
)


test_groups = set(
    test_df[
        "Group_ID"
    ]
)


group_overlap = (
    train_groups
    &
    test_groups
)


print(
    "\nTraining Group_IDs:",
    len(train_groups)
)


print(
    "Testing Group_IDs :",
    len(test_groups)
)


print(
    "Overlapping Group_IDs:",
    len(group_overlap)
)


if group_overlap:

    raise ValueError(
        "Contamination-level leakage detected:\n"
        f"{sorted(group_overlap)}"
    )


print(
    "\nSource-level leakage check: PASS"
)


# ======================================================================
# ORIGINAL CLASS DISTRIBUTION
# ======================================================================

print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)


print(
    "\nTraining class distribution:"
)


print(
    train_df[
        "Binary_Class"
    ].value_counts()
)


print(
    "\nTesting class distribution:"
)


print(
    test_df[
        "Binary_Class"
    ].value_counts()
)


# ======================================================================
# TARGET LABEL VALIDATION
# ======================================================================

print("\n" + "=" * 70)
print("TARGET LABEL VALIDATION")
print("=" * 70)


allowed_labels = {
    0,
    1
}


train_labels = set(
    train_df[
        "Binary_Label"
    ].unique()
)


test_labels = set(
    test_df[
        "Binary_Label"
    ].unique()
)


print(
    "\nTraining labels:",
    sorted(train_labels)
)


print(
    "Testing labels :",
    sorted(test_labels)
)


if not train_labels.issubset(
    allowed_labels
):

    raise ValueError(
        "Unexpected training labels."
    )


if not test_labels.issubset(
    allowed_labels
):

    raise ValueError(
        "Unexpected testing labels."
    )


print(
    "\nBinary label validation: PASS"
)


# ======================================================================
# DATA MATRICES
# ======================================================================

print("\n" + "=" * 70)
print("DATA MATRICES")
print("=" * 70)


X_train = train_df[
    pc_columns
].to_numpy(
    dtype=np.float32
)


X_test = test_df[
    pc_columns
].to_numpy(
    dtype=np.float32
)


y_train = train_df[
    "Binary_Label"
].to_numpy(
    dtype=np.int32
)


y_test = test_df[
    "Binary_Label"
].to_numpy(
    dtype=np.int32
)


print(
    "\nTraining matrix:",
    X_train.shape
)


print(
    "Testing matrix :",
    X_test.shape
)


print(
    "Training labels:",
    y_train.shape
)


print(
    "Testing labels :",
    y_test.shape
)


# ======================================================================
# NUMERICAL VALIDATION
# ======================================================================

print("\n" + "=" * 70)
print("NUMERICAL VALIDATION")
print("=" * 70)


train_nan = np.isnan(
    X_train
).sum()


test_nan = np.isnan(
    X_test
).sum()


train_inf = np.isinf(
    X_train
).sum()


test_inf = np.isinf(
    X_test
).sum()


print(
    "\nTraining NaN:",
    train_nan
)


print(
    "Testing NaN :",
    test_nan
)


print(
    "Training Inf:",
    train_inf
)


print(
    "Testing Inf :",
    test_inf
)


if (
    train_nan > 0
    or
    test_nan > 0
    or
    train_inf > 0
    or
    test_inf > 0
):

    raise ValueError(
        "NaN or Inf detected in PCA features."
    )


print(
    "\nNumerical validation: PASS"
)


# ======================================================================
# RANDOM FOREST TRAINING
# ======================================================================

print("\n" + "=" * 70)
print("TRAINING FINAL RANDOM FOREST")
print("=" * 70)


print(
    "\nRandom Forest parameters:"
)


print(
    "Number of trees:",
    N_ESTIMATORS
)


print(
    "Random state:",
    RANDOM_STATE
)


print(
    "Number of CPU jobs:",
    N_JOBS
)


print(
    "\nTraining samples:",
    len(X_train)
)


print(
    "Testing samples:",
    len(X_test)
)


print(
    "Input features:",
    len(pc_columns)
)


print(
    "\nFitting Random Forest..."
)


rf = RandomForestClassifier(

    n_estimators=N_ESTIMATORS,

    random_state=RANDOM_STATE,

    n_jobs=N_JOBS,

    class_weight=None
)


rf.fit(
    X_train,
    y_train
)


print(
    "\nRandom Forest training completed."
)


# ======================================================================
# GENERATE PREDICTIONS
# ======================================================================

print("\n" + "=" * 70)
print("GENERATING TEST PREDICTIONS")
print("=" * 70)


y_pred = rf.predict(
    X_test
)


print(
    "\nPredictions generated:",
    len(y_pred)
)


# ======================================================================
# PERFORMANCE METRICS
# ======================================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


macro_precision = precision_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)


macro_recall = recall_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)


macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)


# ======================================================================
# RESULTS
# ======================================================================

print("\n" + "=" * 70)
print("RANDOM FOREST RESULTS")
print("=" * 70)


print(
    f"\nTEST ACCURACY : {accuracy * 100:.2f}%"
)


print(
    f"MACRO PRECISION: {macro_precision * 100:.2f}%"
)


print(
    f"MACRO RECALL   : {macro_recall * 100:.2f}%"
)


print(
    f"MACRO F1       : {macro_f1 * 100:.2f}%"
)


# ======================================================================
# CLASSIFICATION REPORT
# ======================================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)


report = classification_report(

    y_test,

    y_pred,

    labels=[0, 1],

    target_names=[
        "Below_Threshold",
        "At_or_Above_Threshold"
    ],

    digits=4,

    zero_division=0
)


print(
    report
)


# ======================================================================
# PER-CLASS PERFORMANCE
# ======================================================================

report_dict = classification_report(

    y_test,

    y_pred,

    labels=[0, 1],

    target_names=[
        "Below_Threshold",
        "At_or_Above_Threshold"
    ],

    output_dict=True,

    zero_division=0
)


per_class_rows = []


for class_name in [

    "Below_Threshold",
    "At_or_Above_Threshold"

]:

    values = report_dict[
        class_name
    ]


    per_class_rows.append({

        "class":
            class_name,

        "precision":
            values["precision"],

        "recall":
            values["recall"],

        "f1_score":
            values["f1-score"],

        "support":
            int(values["support"])
    })


per_class_df = pd.DataFrame(
    per_class_rows
)


print("\n" + "=" * 70)
print("PER-CLASS PERFORMANCE")
print("=" * 70)


print(
    per_class_df.to_string(
        index=False
    )
)


# ======================================================================
# CONFUSION MATRIX
# ======================================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)


cm = confusion_matrix(

    y_test,

    y_pred,

    labels=[0, 1]
)


print(
    "\n"
)


print(
    cm
)


# ======================================================================
# SAVE CONFUSION MATRIX
# ======================================================================

confusion_matrix_path = os.path.join(

    GRAPH_DIR,

    "pistachio_final_random_forest_confusion_matrix.png"
)


plt.figure(
    figsize=(7, 6)
)


plt.imshow(
    cm,
    interpolation="nearest"
)


plt.title(
    "Pistachio Random Forest Confusion Matrix"
)


plt.colorbar()


tick_marks = np.arange(
    2
)


plt.xticks(
    tick_marks,
    [
        "Below Threshold",
        "At/Above Threshold"
    ],
    rotation=25,
    ha="right"
)


plt.yticks(
    tick_marks,
    [
        "Below Threshold",
        "At/Above Threshold"
    ]
)


threshold = (
    cm.max()
    /
    2.0
)


for i in range(
    cm.shape[0]
):

    for j in range(
        cm.shape[1]
    ):

        plt.text(

            j,

            i,

            format(
                cm[i, j],
                "d"
            ),

            ha="center",

            va="center",

            color=(
                "white"
                if cm[i, j] > threshold
                else "black"
            )
        )


plt.ylabel(
    "True Class"
)


plt.xlabel(
    "Predicted Class"
)


plt.tight_layout()


plt.savefig(
    confusion_matrix_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    "\nSaved confusion matrix:"
)


print(
    confusion_matrix_path
)


# ======================================================================
# SAVE PREDICTIONS
# ======================================================================

predictions_df = test_df[
    [
        "Sample_ID",
        "Level",
        "AFB1_ug_kg",
        "Binary_Label",
        "Binary_Class",
        "Group_ID"
    ]
].copy()


predictions_df[
    "Predicted_Label"
] = y_pred


predictions_df[
    "Predicted_Class"
] = np.where(

    y_pred == 0,

    "Below_Threshold",

    "At_or_Above_Threshold"
)


prediction_path = os.path.join(

    OUTPUT_DIR,

    "pistachio_final_random_forest_predictions.csv"
)


predictions_df.to_csv(

    prediction_path,

    index=False
)


print(
    "\nSaved predictions:"
)


print(
    prediction_path
)


# ======================================================================
# SAVE METRICS
# ======================================================================

metrics_df = pd.DataFrame({

    "model":
        ["Random Forest"],

    "accuracy":
        [accuracy],

    "macro_precision":
        [macro_precision],

    "macro_recall":
        [macro_recall],

    "macro_f1":
        [macro_f1],

    "pca_components":
        [len(pc_columns)],

    "training_samples":
        [len(X_train)],

    "testing_samples":
        [len(X_test)],

    "training_sample_ids":
        [len(train_sample_ids)],

    "testing_sample_ids":
        [len(test_sample_ids)],

    "sample_id_overlap":
        [len(sample_overlap)],

    "training_groups":
        [len(train_groups)],

    "testing_groups":
        [len(test_groups)],

    "group_overlap":
        [len(group_overlap)],

    "n_estimators":
        [N_ESTIMATORS],

    "random_state":
        [RANDOM_STATE]
})


metrics_path = os.path.join(

    OUTPUT_DIR,

    "pistachio_final_random_forest_metrics.csv"
)


metrics_df.to_csv(

    metrics_path,

    index=False
)


print(
    "\nSaved metrics:"
)


print(
    metrics_path
)


# ======================================================================
# SAVE PER-CLASS METRICS
# ======================================================================

per_class_path = os.path.join(

    OUTPUT_DIR,

    "pistachio_final_random_forest_per_class_metrics.csv"
)


per_class_df.to_csv(

    per_class_path,

    index=False
)


print(
    "\nSaved per-class metrics:"
)


print(
    per_class_path
)


# ======================================================================
# FEATURE IMPORTANCE
# ======================================================================

print("\n" + "=" * 70)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 70)


importance_df = pd.DataFrame({

    "feature":
        pc_columns,

    "importance":
        rf.feature_importances_

})


importance_df = importance_df.sort_values(

    "importance",

    ascending=False

).reset_index(
    drop=True
)


print(
    "\nTop 20 PCA features:"
)


print(
    importance_df.head(
        20
    ).to_string(
        index=False
    )
)


importance_path = os.path.join(

    OUTPUT_DIR,

    "pistachio_random_forest_feature_importance.csv"
)


importance_df.to_csv(

    importance_path,

    index=False
)


print(
    "\nSaved feature importance:"
)


print(
    importance_path
)


# ======================================================================
# FINAL SUMMARY
# ======================================================================

print("\n" + "=" * 70)
print("FINAL PISTACHIO HSI RANDOM FOREST COMPLETED")
print("=" * 70)


print(
f"""
Final test accuracy : {accuracy * 100:.2f}%

Macro Precision     : {macro_precision * 100:.2f}%

Macro Recall        : {macro_recall * 100:.2f}%

Macro F1            : {macro_f1 * 100:.2f}%


PCA components      : {len(pc_columns)}

Training samples    : {len(X_train)}

Testing samples     : {len(X_test)}


Training Sample_IDs : {len(train_sample_ids)}

Testing Sample_IDs  : {len(test_sample_ids)}

Sample_ID overlap   : {len(sample_overlap)}


Training levels     : {len(train_groups)}

Testing levels      : {len(test_groups)}

Level overlap       : {len(group_overlap)}


Source-level leakage: PASS


AFB1 threshold      : 8.0 µg/kg


Classification:

0 = Below_Threshold

1 = At_or_Above_Threshold


Random Forest:

Trees               : {N_ESTIMATORS}

Random state         : {RANDOM_STATE}


Predictions:

{prediction_path}


Metrics:

{metrics_path}


Per-class metrics:

{per_class_path}


Feature importance:

{importance_path}


Confusion matrix:

{confusion_matrix_path}
"""
)


print(
"=" * 70
)