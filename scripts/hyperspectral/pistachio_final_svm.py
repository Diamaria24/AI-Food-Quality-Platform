import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ======================================================================
# PISTACHIO HSI FINAL SVM CLASSIFICATION
# ======================================================================

print("=" * 70)
print("PISTACHIO HSI FINAL SVM CLASSIFICATION")
print("=" * 70)


print(
    """
THIRD-CLASSIFIER COMPARISON

Random Forest:
    COMPLETED

Current experiment:
    SVM

XGBoost:
    TO BE RUN AFTER SVM


EXISTING LEAKAGE-SAFE TRAIN/TEST DATA
will be used without creating a new split.
"""
)


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


OUTPUT_DIR = PCA_DIR


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
# SVM CONFIGURATION
# ======================================================================

RANDOM_STATE = 42

SVM_SAMPLES_PER_CLASS = 15000

SVM_KERNEL = "rbf"

SVM_C = 1.0

SVM_GAMMA = "scale"


# ======================================================================
# LOAD DATA
# ======================================================================

print("\n" + "=" * 70)
print("LOADING PISTACHIO 100-PC DATA")
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


pc_columns = sorted(
    pc_columns,
    key=lambda x: int(
        x.replace(
            "PC",
            ""
        )
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
        "Expected exactly 100 PCA components."
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
            f"Missing PCA feature in testing data: "
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
# CONTAMINATION LEVEL LEAKAGE CHECK
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
print("ORIGINAL CLASS DISTRIBUTION")
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
# LABEL VALIDATION
# ======================================================================

print("\n" + "=" * 70)
print("BINARY LABEL VALIDATION")
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


X_train_original = train_df[
    pc_columns
].to_numpy(
    dtype=np.float32
)


X_test = test_df[
    pc_columns
].to_numpy(
    dtype=np.float32
)


y_train_original = train_df[
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
    "\nOriginal training matrix:",
    X_train_original.shape
)


print(
    "Testing matrix          :",
    X_test.shape
)


print(
    "Original training labels:",
    y_train_original.shape
)


print(
    "Testing labels          :",
    y_test.shape
)


# ======================================================================
# NUMERICAL VALIDATION
# ======================================================================

print("\n" + "=" * 70)
print("NUMERICAL VALIDATION")
print("=" * 70)


print(
    "\nTraining NaN:",
    np.isnan(
        X_train_original
    ).sum()
)


print(
    "Testing NaN :",
    np.isnan(
        X_test
    ).sum()
)


print(
    "Training Inf:",
    np.isinf(
        X_train_original
    ).sum()
)


print(
    "Testing Inf :",
    np.isinf(
        X_test
    ).sum()
)


if (
    np.isnan(
        X_train_original
    ).any()
    or
    np.isnan(
        X_test
    ).any()
    or
    np.isinf(
        X_train_original
    ).any()
    or
    np.isinf(
        X_test
    ).any()
):

    raise ValueError(
        "NaN or Inf detected in PCA data."
    )


print(
    "\nNumerical validation: PASS"
)


# ======================================================================
# CREATE STRATIFIED SVM TRAINING SUBSET
# ======================================================================

print("\n" + "=" * 70)
print("CREATING STRATIFIED SVM TRAINING SUBSET")
print("=" * 70)


print(
    "\nSamples per class:",
    SVM_SAMPLES_PER_CLASS
)


print(
    "Classes:",
    sorted(
        train_labels
    )
)


expected_svm_samples = (
    len(
        allowed_labels
    )
    *
    SVM_SAMPLES_PER_CLASS
)


print(
    "Expected SVM training samples:",
    expected_svm_samples
)


rng = np.random.default_rng(
    RANDOM_STATE
)


selected_indices = []


for class_label in sorted(
    allowed_labels
):

    class_indices = np.where(
        y_train_original
        ==
        class_label
    )[0]


    print(
        f"\nClass {class_label}:"
    )


    print(
        "Available samples:",
        len(class_indices)
    )


    if len(class_indices) < SVM_SAMPLES_PER_CLASS:

        raise ValueError(
            f"Not enough samples for class "
            f"{class_label}."
        )


    chosen = rng.choice(

        class_indices,

        size=SVM_SAMPLES_PER_CLASS,

        replace=False
    )


    selected_indices.extend(
        chosen.tolist()
    )


# --------------------------------------------------------------
# Shuffle the combined selected indices
# --------------------------------------------------------------

selected_indices = np.array(
    selected_indices,
    dtype=np.int64
)


rng.shuffle(
    selected_indices
)


# ======================================================================
# CREATE SVM SUBSET
# ======================================================================

X_train = (
    X_train_original[
        selected_indices
    ]
)


y_train = (
    y_train_original[
        selected_indices
    ]
)


print(
    "\n" + "=" * 70
)

print(
    "SVM TRAINING SUBSET CREATED"
)

print(
    "=" * 70
)


print(
    "\nSVM training shape:",
    X_train.shape
)


print(
    "\nSVM training class distribution:"
)


print(
    pd.Series(
        y_train
    ).value_counts().sort_index()
)


# ======================================================================
# VERIFY STRATIFICATION
# ======================================================================

expected_class_count = (
    SVM_SAMPLES_PER_CLASS
)


for class_label in sorted(
    allowed_labels
):

    actual_count = np.sum(
        y_train
        ==
        class_label
    )


    if actual_count != expected_class_count:

        raise ValueError(
            f"Unexpected class count for "
            f"class {class_label}: "
            f"{actual_count}"
        )


print(
    "\nStratified SVM subset validation: PASS"
)


# ======================================================================
# DELETE ORIGINAL TRAINING MATRIX
# ======================================================================

del X_train_original
del y_train_original


# ======================================================================
# STANDARDIZATION
# ======================================================================

print("\n" + "=" * 70)
print("TRAINING-DERIVED STANDARDIZATION")
print("=" * 70)


print(
    "\nFitting StandardScaler using SVM training subset only..."
)


scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(
    X_train
)


print(
    "Transforming full testing dataset..."
)


X_test_scaled = scaler.transform(
    X_test
)


print(
    "Training standardization completed."
)


# ======================================================================
# STANDARDIZATION VALIDATION
# ======================================================================

print("\n" + "=" * 70)
print("STANDARDIZATION VALIDATION")
print("=" * 70)


print(
    "\nTraining NaN:",
    np.isnan(
        X_train_scaled
    ).sum()
)


print(
    "Testing NaN :",
    np.isnan(
        X_test_scaled
    ).sum()
)


print(
    "Training Inf:",
    np.isinf(
        X_train_scaled
    ).sum()
)


print(
    "Testing Inf :",
    np.isinf(
        X_test_scaled
    ).sum()
)


if (
    np.isnan(
        X_train_scaled
    ).any()
    or
    np.isnan(
        X_test_scaled
    ).any()
    or
    np.isinf(
        X_train_scaled
    ).any()
    or
    np.isinf(
        X_test_scaled
    ).any()
):

    raise ValueError(
        "NaN or Inf detected after standardization."
    )


print(
    "\nStandardization validation: PASS"
)


# ======================================================================
# TRAIN FINAL SVM
# ======================================================================

print("\n" + "=" * 70)
print("TRAINING FINAL SVM")
print("=" * 70)


print(
    "\nKernel :",
    SVM_KERNEL
)


print(
    "C      :",
    SVM_C
)


print(
    "Gamma  :",
    SVM_GAMMA
)


print(
    "Training samples:",
    len(X_train_scaled)
)


print(
    "Testing samples :",
    len(X_test_scaled)
)


print(
    "Input features  :",
    len(pc_columns)
)


print(
    "\nFitting SVM..."
)


svm = SVC(

    kernel=SVM_KERNEL,

    C=SVM_C,

    gamma=SVM_GAMMA,

    random_state=RANDOM_STATE
)


svm.fit(
    X_train_scaled,
    y_train
)


print(
    "\nSVM training completed."
)


# ======================================================================
# GENERATE PREDICTIONS
# ======================================================================

print("\n" + "=" * 70)
print("GENERATING TEST PREDICTIONS")
print("=" * 70)


y_pred = svm.predict(
    X_test_scaled
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
print("SVM RESULTS")
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
            int(
                values["support"]
            )
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

    "pistachio_final_svm_confusion_matrix.png"
)


plt.figure(
    figsize=(7, 6)
)


plt.imshow(
    cm,
    interpolation="nearest"
)


plt.title(
    "Pistachio SVM Confusion Matrix"
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

    "pistachio_final_svm_predictions.csv"
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
        ["SVM"],

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

    "original_training_samples":
        [84000],

    "svm_training_samples":
        [len(X_train_scaled)],

    "testing_samples":
        [len(X_test_scaled)],

    "samples_per_class":
        [SVM_SAMPLES_PER_CLASS],

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

    "kernel":
        [SVM_KERNEL],

    "C":
        [SVM_C],

    "gamma":
        [SVM_GAMMA],

    "random_state":
        [RANDOM_STATE]
})


metrics_path = os.path.join(

    OUTPUT_DIR,

    "pistachio_final_svm_metrics.csv"
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

    "pistachio_final_svm_per_class_metrics.csv"
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
# SAVE STANDARD SCALER
# ======================================================================

scaler_path = os.path.join(

    OUTPUT_DIR,

    "pistachio_final_svm_standard_scaler.joblib"
)


import joblib


joblib.dump(

    scaler,

    scaler_path
)


print(
    "\nSaved StandardScaler:"
)


print(
    scaler_path
)


# ======================================================================
# FINAL SUMMARY
# ======================================================================

print("\n" + "=" * 70)
print("FINAL PISTACHIO HSI SVM COMPLETED")
print("=" * 70)


print(
f"""
Final test accuracy : {accuracy * 100:.2f}%

Macro Precision     : {macro_precision * 100:.2f}%

Macro Recall        : {macro_recall * 100:.2f}%

Macro F1            : {macro_f1 * 100:.2f}%


PCA components      : {len(pc_columns)}

Original training samples : 84000

SVM training samples      : {len(X_train_scaled)}

Testing samples           : {len(X_test_scaled)}


Samples per class         : {SVM_SAMPLES_PER_CLASS}


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


SVM:

Kernel              : {SVM_KERNEL}

C                   : {SVM_C}

Gamma               : {SVM_GAMMA}

Random state         : {RANDOM_STATE}


Predictions:

{prediction_path}


Metrics:

{metrics_path}


Per-class metrics:

{per_class_path}


StandardScaler:

{scaler_path}


Confusion matrix:

{confusion_matrix_path}
"""
)


print(
"=" * 70
)


print(
"""
NEXT STEP:

Run XGBoost using the SAME:

    100 PCA components
    84,000 original training spectra
    20,000 testing spectra
    42 training cubes
    10 testing cubes
    21 training contamination levels
    5 testing contamination levels
    0 contamination-level overlap

Then compare:

    Random Forest
    SVM
    XGBoost
"""
)


print(
"=" * 70
)