import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest


# ======================================================================
# STRAWBERRY SENSOR PATTERN ANALYSIS
# ======================================================================

print("=" * 70)
print("STRAWBERRY SENSOR PATTERN ANALYSIS")
print("=" * 70)

print("""
Input:

    Strawberry standardized sensor data
    PCA representation

Selected representation:

    8 principal components
    ~95.15% cumulative explained variance

Important:

    No artificial freshness labels are created.

    This is an unsupervised sensor-condition analysis.
""")


# ======================================================================
# PROJECT PATHS
# ======================================================================

PROJECT_ROOT = r"D:\Projects\AI-Food-Quality-Platform"


INPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "processed",
    "strawberry_sensor"
)


PCA_FILE = os.path.join(
    INPUT_DIR,
    "strawberry_sensor_pca.csv"
)


VARIANCE_FILE = os.path.join(
    INPUT_DIR,
    "strawberry_sensor_pca_explained_variance.csv"
)


OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "strawberry_sensor_analysis"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ======================================================================
# CONFIGURATION
# ======================================================================

N_PCS = 8

WINDOW_SIZE = 300

RANDOM_STATE = 42

CONTAMINATION = 0.01


# ======================================================================
# LOAD PCA DATA
# ======================================================================

print("\n" + "=" * 70)
print("LOADING PCA DATA")
print("=" * 70)


if not os.path.exists(PCA_FILE):

    raise FileNotFoundError(
        f"PCA file not found:\n{PCA_FILE}"
    )


df = pd.read_csv(
    PCA_FILE
)


print(
    "\nPCA dataset shape:",
    df.shape
)


# ======================================================================
# VALIDATE REQUIRED COLUMNS
# ======================================================================

print("\n" + "=" * 70)
print("COLUMN VALIDATION")
print("=" * 70)


required_columns = [

    "HHMMSS",
    "experiment"

]


for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Required column missing: {column}"
        )


pc_columns = [

    f"PC{i}"

    for i in range(
        1,
        N_PCS + 1
    )

]


for column in pc_columns:

    if column not in df.columns:

        raise ValueError(
            f"Required PCA column missing: {column}"
        )


print(
    "\nAll required columns found."
)


# ======================================================================
# TIMESTAMP CONVERSION
# ======================================================================

print("\n" + "=" * 70)
print("TIMESTAMP PROCESSING")
print("=" * 70)


df["datetime"] = pd.to_datetime(
    df["HHMMSS"],
    errors="coerce"
)


invalid_timestamps = df[
    "datetime"
].isna().sum()


print(
    "\nInvalid timestamps:",
    invalid_timestamps
)


if invalid_timestamps > 0:

    raise ValueError(
        "Invalid timestamps detected."
    )


df = df.sort_values(
    [
        "experiment",
        "datetime"
    ]
).reset_index(
    drop=True
)


print(
    "\nTimestamp processing: PASS"
)


# ======================================================================
# PCA NUMERICAL VALIDATION
# ======================================================================

print("\n" + "=" * 70)
print("PCA NUMERICAL VALIDATION")
print("=" * 70)


X_pca = df[
    pc_columns
].to_numpy(
    dtype=np.float64
)


nan_count = np.isnan(
    X_pca
).sum()


inf_count = np.isinf(
    X_pca
).sum()


print(
    "\nPCA matrix shape:",
    X_pca.shape
)


print(
    "NaN values:",
    int(nan_count)
)


print(
    "Inf values:",
    int(inf_count)
)


if nan_count > 0:

    raise ValueError(
        "NaN values found in PCA data."
    )


if inf_count > 0:

    raise ValueError(
        "Infinite values found in PCA data."
    )


print(
    "\nPCA numerical validation: PASS"
)


# ======================================================================
# PCA EXPLAINED VARIANCE
# ======================================================================

print("\n" + "=" * 70)
print("PCA EXPLAINED VARIANCE")
print("=" * 70)


if os.path.exists(
    VARIANCE_FILE
):

    variance_df = pd.read_csv(
        VARIANCE_FILE
    )


    selected_variance = variance_df[
        variance_df["PC"].isin(
            pc_columns
        )
    ]


    print(
        "\nSelected PCA components:"
    )


    print(
        selected_variance.to_string(
            index=False
        )
    )


    pc8_row = variance_df[
        variance_df["PC"] == "PC8"
    ]


    if len(pc8_row) > 0:

        cumulative_variance = (
            pc8_row[
                "cumulative_variance_percent"
            ].iloc[0]
        )

    else:

        cumulative_variance = np.nan

else:

    cumulative_variance = np.nan


print(
    "\nSelected components:",
    N_PCS
)


if not np.isnan(
    cumulative_variance
):

    print(
        "Cumulative explained variance:",
        f"{cumulative_variance:.4f}%"
    )


# ======================================================================
# OVERALL PCA STATISTICS
# ======================================================================

print("\n" + "=" * 70)
print("OVERALL PCA STATISTICS")
print("=" * 70)


overall_statistics = df[
    pc_columns
].describe().T


overall_statistics[
    "variance"
] = df[
    pc_columns
].var()


print(
    "\nPCA statistics:"
)


print(
    overall_statistics.to_string()
)


# ======================================================================
# EXPERIMENT-LEVEL PCA STATISTICS
# ======================================================================

print("\n" + "=" * 70)
print("EXPERIMENT-LEVEL PCA ANALYSIS")
print("=" * 70)


experiment_rows = []


for experiment_name, group in df.groupby(
    "experiment"
):

    print(
        f"\nExperiment: {experiment_name}"
    )


    print(
        "Observations:",
        len(group)
    )


    row = {

        "experiment":
            experiment_name,

        "observations":
            len(group)

    }


    for pc in pc_columns:

        row[
            f"{pc}_mean"
        ] = group[
            pc
        ].mean()


        row[
            f"{pc}_std"
        ] = group[
            pc
        ].std()


    experiment_rows.append(
        row
    )


experiment_statistics_df = pd.DataFrame(
    experiment_rows
)


print(
    "\nExperiment-level statistics:"
)


print(
    experiment_statistics_df.to_string(
        index=False
    )
)


# ======================================================================
# PCA CENTROID DISTANCE BETWEEN EXPERIMENTS
# ======================================================================

print("\n" + "=" * 70)
print("EXPERIMENT PCA CENTROID ANALYSIS")
print("=" * 70)


experiment_centroids = (
    df.groupby(
        "experiment"
    )[pc_columns]
    .mean()
)


print(
    "\nExperiment centroids:"
)


print(
    experiment_centroids.to_string()
)


if len(
    experiment_centroids
) == 2:

    centroid_a = (
        experiment_centroids.iloc[0]
        .to_numpy()
    )


    centroid_b = (
        experiment_centroids.iloc[1]
        .to_numpy()
    )


    centroid_distance = np.linalg.norm(
        centroid_a - centroid_b
    )


    print(
        "\nEuclidean distance between "
        "experiment centroids:",
        centroid_distance
    )

else:

    centroid_distance = np.nan


# ======================================================================
# PC1-PC2 TRAJECTORY PLOT
# ======================================================================

print("\n" + "=" * 70)
print("CREATING PC1-PC2 PLOT")
print("=" * 70)


plt.figure(
    figsize=(10, 7)
)


for experiment_name, group in df.groupby(
    "experiment"
):

    plt.plot(

        group["PC1"].to_numpy(),

        group["PC2"].to_numpy(),

        linewidth=0.7,

        label=experiment_name

    )


plt.xlabel(
    "PC1"
)

plt.ylabel(
    "PC2"
)

plt.title(
    "Strawberry Sensor PCA Trajectory: PC1 vs PC2"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


pc12_path = os.path.join(
    OUTPUT_DIR,
    "strawberry_pc1_pc2_trajectory.png"
)


plt.savefig(
    pc12_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    "\nSaved:",
    pc12_path
)


# ======================================================================
# PC1-PC3 TRAJECTORY PLOT
# ======================================================================

print("\n" + "=" * 70)
print("CREATING PC1-PC3 PLOT")
print("=" * 70)


plt.figure(
    figsize=(10, 7)
)


for experiment_name, group in df.groupby(
    "experiment"
):

    plt.plot(

        group["PC1"].to_numpy(),

        group["PC3"].to_numpy(),

        linewidth=0.7,

        label=experiment_name

    )


plt.xlabel(
    "PC1"
)

plt.ylabel(
    "PC3"
)

plt.title(
    "Strawberry Sensor PCA Trajectory: PC1 vs PC3"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


pc13_path = os.path.join(
    OUTPUT_DIR,
    "strawberry_pc1_pc3_trajectory.png"
)


plt.savefig(
    pc13_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    "\nSaved:",
    pc13_path
)


# ======================================================================
# TEMPORAL PCA ANALYSIS
# ======================================================================

print("\n" + "=" * 70)
print("TEMPORAL PCA ANALYSIS")
print("=" * 70)


temporal_frames = []


for experiment_name, group in df.groupby(
    "experiment"
):

    group = group.sort_values(
        "datetime"
    ).copy()


    # Create sequential observation number

    group[
        "observation_index"
    ] = np.arange(
        len(group)
    )


    # Normalize observation position

    if len(group) > 1:

        group[
            "relative_position"
        ] = (

            group[
                "observation_index"
            ]

            /

            (
                len(group) - 1
            )

        )

    else:

        group[
            "relative_position"
        ] = 0.0


    # Window number

    group[
        "window"
    ] = (

        group[
            "observation_index"
        ]

        //

        WINDOW_SIZE

    )


    temporal_frames.append(
        group
    )


df_temporal = pd.concat(
    temporal_frames,
    ignore_index=True
)


print(
    "\nTemporal window size:",
    WINDOW_SIZE
)


print(
    "Number of temporal windows:",
    df_temporal[
        "window"
    ].nunique()
)


# ======================================================================
# WINDOWED PCA STATISTICS
# ======================================================================

print("\n" + "=" * 70)
print("CALCULATING WINDOWED PCA STATISTICS")
print("=" * 70)


window_rows = []


for (
    experiment_name,
    experiment_group
) in df_temporal.groupby(
    "experiment"
):

    for (
        window_number,
        group
    ) in experiment_group.groupby(
        "window"
    ):

        row = {

            "experiment":
                experiment_name,

            "window":
                int(window_number),

            "observations":
                len(group),

            "relative_position_mean":
                group[
                    "relative_position"
                ].mean()

        }


        for pc in pc_columns:

            row[
                f"{pc}_mean"
            ] = group[
                pc
            ].mean()


            row[
                f"{pc}_std"
            ] = group[
                pc
            ].std()


            row[
                f"{pc}_min"
            ] = group[
                pc
            ].min()


            row[
                f"{pc}_max"
            ] = group[
                pc
            ].max()


        window_rows.append(
            row
        )


window_statistics_df = pd.DataFrame(
    window_rows
)


print(
    "\nWindowed analysis completed."
)


print(
    "Windowed rows:",
    len(window_statistics_df)
)


# ======================================================================
# PC1 TEMPORAL PLOT
# ======================================================================

print("\n" + "=" * 70)
print("CREATING PC1 TEMPORAL PLOT")
print("=" * 70)


plt.figure(
    figsize=(12, 7)
)


for experiment_name, group in (
    window_statistics_df.groupby(
        "experiment"
    )
):

    plt.plot(

        group[
            "relative_position_mean"
        ],

        group[
            "PC1_mean"
        ],

        marker=".",

        linewidth=1.0,

        label=experiment_name

    )


plt.xlabel(
    "Relative Experiment Progress"
)

plt.ylabel(
    "Mean PC1"
)

plt.title(
    "Strawberry Sensor PC1 Temporal Trend"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


pc1_temporal_path = os.path.join(
    OUTPUT_DIR,
    "strawberry_pc1_temporal_trend.png"
)


plt.savefig(
    pc1_temporal_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    "\nSaved:",
    pc1_temporal_path
)


# ======================================================================
# PC2 TEMPORAL PLOT
# ======================================================================

print("\n" + "=" * 70)
print("CREATING PC2 TEMPORAL PLOT")
print("=" * 70)


plt.figure(
    figsize=(12, 7)
)


for experiment_name, group in (
    window_statistics_df.groupby(
        "experiment"
    )
):

    plt.plot(

        group[
            "relative_position_mean"
        ],

        group[
            "PC2_mean"
        ],

        marker=".",

        linewidth=1.0,

        label=experiment_name

    )


plt.xlabel(
    "Relative Experiment Progress"
)

plt.ylabel(
    "Mean PC2"
)

plt.title(
    "Strawberry Sensor PC2 Temporal Trend"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


pc2_temporal_path = os.path.join(
    OUTPUT_DIR,
    "strawberry_pc2_temporal_trend.png"
)


plt.savefig(
    pc2_temporal_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    "\nSaved:",
    pc2_temporal_path
)


# ======================================================================
# SENSOR CONDITION SCORE
# ======================================================================

print("\n" + "=" * 70)
print("CALCULATING SENSOR CONDITION SCORE")
print("=" * 70)


# Calculate PCA distance from the global PCA centre.

pca_mean = df[
    pc_columns
].mean().to_numpy()


pca_distance = np.linalg.norm(

    X_pca - pca_mean,

    axis=1

)


df[
    "PCA_distance"
] = pca_distance


# Scale distance to 0-100.

distance_scaler = MinMaxScaler(
    feature_range=(0, 100)
)


df[
    "condition_deviation_score"
] = distance_scaler.fit_transform(

    pca_distance.reshape(-1, 1)

).ravel()


print(
    "\nCondition deviation score created."
)


print(
    "Range:",
    f"{df['condition_deviation_score'].min():.2f}",
    "to",
    f"{df['condition_deviation_score'].max():.2f}"
)


# IMPORTANT:
#
# This is NOT a freshness probability.
#
# It is a relative sensor-deviation score.
#
# Higher value = farther from the overall sensor centre.
#
# It must not be interpreted as direct freshness
# without ground-truth validation.


# ======================================================================
# ISOLATION FOREST
# ======================================================================

print("\n" + "=" * 70)
print("UNSUPERVISED ANOMALY ANALYSIS")
print("=" * 70)


print(
    "\nTraining Isolation Forest on PCA features..."
)


isolation_model = IsolationForest(

    n_estimators=200,

    contamination=CONTAMINATION,

    random_state=RANDOM_STATE,

    n_jobs=-1

)


isolation_model.fit(
    df[
        pc_columns
    ]
)


df[
    "anomaly_prediction"
] = isolation_model.predict(
    df[
        pc_columns
    ]
)


df[
    "anomaly_score"
] = isolation_model.score_samples(
    df[
        pc_columns
    ]
)


df[
    "is_anomaly"
] = (

    df[
        "anomaly_prediction"
    ]

    ==

    -1

)


anomaly_count = df[
    "is_anomaly"
].sum()


anomaly_percentage = (

    anomaly_count

    /

    len(df)

    *

    100

)


print(
    "\nAnomalies detected:",
    int(anomaly_count)
)


print(
    "Anomaly percentage:",
    f"{anomaly_percentage:.4f}%"
)


# ======================================================================
# ANOMALY SUMMARY BY EXPERIMENT
# ======================================================================

print("\n" + "=" * 70)
print("ANOMALY SUMMARY BY EXPERIMENT")
print("=" * 70)


anomaly_experiment_rows = []


for experiment_name, group in df.groupby(
    "experiment"
):

    count = group[
        "is_anomaly"
    ].sum()


    percentage = (

        count

        /

        len(group)

        *

        100

    )


    anomaly_experiment_rows.append({

        "experiment":
            experiment_name,

        "observations":
            len(group),

        "anomalies":
            int(count),

        "anomaly_percentage":
            percentage,

        "mean_anomaly_score":
            group[
                "anomaly_score"
            ].mean(),

        "mean_condition_deviation":
            group[
                "condition_deviation_score"
            ].mean()

    })


anomaly_experiment_df = pd.DataFrame(
    anomaly_experiment_rows
)


print(
    anomaly_experiment_df.to_string(
        index=False
    )
)


# ======================================================================
# CONDITION SCORE BY EXPERIMENT
# ======================================================================

print("\n" + "=" * 70)
print("CONDITION SCORE BY EXPERIMENT")
print("=" * 70)


condition_summary = (

    df.groupby(
        "experiment"
    )[

        [
            "condition_deviation_score",
            "PCA_distance",
            "anomaly_score"

        ]

    ]

    .agg(
        [
            "mean",
            "std",
            "min",
            "max"
        ]
    )

)


print(
    condition_summary.to_string()
)


# ======================================================================
# TEMPORAL CONDITION SCORE
# ======================================================================

print("\n" + "=" * 70)
print("TEMPORAL CONDITION SCORE")
print("=" * 70)


condition_window_rows = []


for (
    experiment_name,
    experiment_group
) in df_temporal.groupby(
    "experiment"
):

    # Add scores from main dataframe by matching index

    group_indices = experiment_group.index

    score_group = df.loc[
        group_indices
    ].copy()


    score_group[
        "window"
    ] = (

        score_group[
            "index"
        ]

        if "index" in score_group.columns

        else np.arange(
            len(score_group)
        )

    )


# Rebuild using ordered groups.

condition_frames = []


for experiment_name, group in df.groupby(
    "experiment"
):

    group = group.sort_values(
        "datetime"
    ).copy()


    group[
        "observation_index"
    ] = np.arange(
        len(group)
    )


    group[
        "window"
    ] = (

        group[
            "observation_index"
        ]

        //

        WINDOW_SIZE

    )


    condition_frames.append(
        group
    )


condition_df = pd.concat(
    condition_frames,
    ignore_index=True
)


condition_window_df = (

    condition_df

    .groupby(
        [
            "experiment",
            "window"
        ]
    )

    .agg(

        observations=(
            "PC1",
            "size"
        ),

        relative_position=(
            "observation_index",
            "mean"
        ),

        mean_condition_deviation=(
            "condition_deviation_score",
            "mean"
        ),

        std_condition_deviation=(
            "condition_deviation_score",
            "std"
        ),

        anomaly_fraction=(
            "is_anomaly",
            "mean"
        )

    )

    .reset_index()

)


# Normalize position inside each experiment.

condition_window_df[
    "relative_position"
] = condition_window_df.groupby(
    "experiment"
)[
    "relative_position"
].transform(

    lambda x:

    x / x.max()

    if x.max() > 0

    else 0

)


condition_window_df[
    "anomaly_percentage"
] = (

    condition_window_df[
        "anomaly_fraction"
    ]

    *

    100

)


print(
    "\nCondition windows:",
    len(condition_window_df)
)


# ======================================================================
# CONDITION SCORE PLOT
# ======================================================================

print("\n" + "=" * 70)
print("CREATING CONDITION SCORE PLOT")
print("=" * 70)


plt.figure(
    figsize=(12, 7)
)


for experiment_name, group in (
    condition_window_df.groupby(
        "experiment"
    )
):

    plt.plot(

        group[
            "relative_position"
        ],

        group[
            "mean_condition_deviation"
        ],

        marker=".",

        linewidth=1.0,

        label=experiment_name

    )


plt.xlabel(
    "Relative Experiment Progress"
)

plt.ylabel(
    "Mean Sensor Deviation Score"
)

plt.title(
    "Strawberry Relative Sensor-Condition Trend"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


condition_plot_path = os.path.join(
    OUTPUT_DIR,
    "strawberry_sensor_condition_trend.png"
)


plt.savefig(
    condition_plot_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    "\nSaved:",
    condition_plot_path
)


# ======================================================================
# SAVE OBSERVATION-LEVEL RESULTS
# ======================================================================

print("\n" + "=" * 70)
print("SAVING OBSERVATION-LEVEL RESULTS")
print("=" * 70)


observation_columns = [

    "HHMMSS",
    "datetime",
    "experiment"

]


if "FTime" in df.columns:

    observation_columns.append(
        "FTime"
    )


observation_columns += pc_columns


observation_columns += [

    "PCA_distance",

    "condition_deviation_score",

    "anomaly_score",

    "anomaly_prediction",

    "is_anomaly"

]


observation_results = df[
    observation_columns
].copy()


observation_output_path = os.path.join(

    OUTPUT_DIR,

    "strawberry_sensor_pca_condition_analysis.csv"

)


observation_results.to_csv(

    observation_output_path,

    index=False

)


print(
    "\nSaved observation-level results:"
)


print(
    observation_output_path
)


# ======================================================================
# SAVE EXPERIMENT STATISTICS
# ======================================================================

experiment_output_path = os.path.join(

    OUTPUT_DIR,

    "strawberry_sensor_experiment_pca_statistics.csv"

)


experiment_statistics_df.to_csv(

    experiment_output_path,

    index=False

)


print(
    "\nSaved experiment statistics:"
)


print(
    experiment_output_path
)


# ======================================================================
# SAVE WINDOW STATISTICS
# ======================================================================

window_output_path = os.path.join(

    OUTPUT_DIR,

    "strawberry_sensor_temporal_window_statistics.csv"

)


window_statistics_df.to_csv(

    window_output_path,

    index=False

)


print(
    "\nSaved temporal window statistics:"
)


print(
    window_output_path
)


# ======================================================================
# SAVE CONDITION WINDOWS
# ======================================================================

condition_window_output_path = os.path.join(

    OUTPUT_DIR,

    "strawberry_sensor_condition_windows.csv"

)


condition_window_df.to_csv(

    condition_window_output_path,

    index=False

)


print(
    "\nSaved condition-window analysis:"
)


print(
    condition_window_output_path
)


# ======================================================================
# SAVE ANOMALY SUMMARY
# ======================================================================

anomaly_summary_path = os.path.join(

    OUTPUT_DIR,

    "strawberry_sensor_anomaly_summary.csv"

)


anomaly_experiment_df.to_csv(

    anomaly_summary_path,

    index=False

)


print(
    "\nSaved anomaly summary:"
)


print(
    anomaly_summary_path
)


# ======================================================================
# SAVE ISOLATION FOREST MODEL
# ======================================================================

import joblib


model_path = os.path.join(

    OUTPUT_DIR,

    "strawberry_sensor_isolation_forest.joblib"

)


joblib.dump(

    isolation_model,

    model_path

)


print(
    "\nSaved Isolation Forest model:"
)


print(
    model_path
)


# ======================================================================
# FINAL SUMMARY
# ======================================================================

print("\n" + "=" * 70)
print("STRAWBERRY SENSOR PATTERN ANALYSIS COMPLETED")
print("=" * 70)


print(
f"""
Total observations          : {len(df)}

Experiments                 : {
    df["experiment"].nunique()
}

PCA components used         : {N_PCS}

Cumulative PCA variance    : {
    f"{cumulative_variance:.4f}%"
    if not np.isnan(cumulative_variance)
    else "Not available"
}

Temporal window size        : {WINDOW_SIZE}

Detected anomalies          : {int(anomaly_count)}

Anomaly percentage          : {anomaly_percentage:.4f}%


Important interpretation:

The condition_deviation_score
is a relative sensor-deviation measure.

It is NOT a direct freshness percentage.

The Isolation Forest result is an
unsupervised anomaly indicator.

No artificial freshness labels were created.

No supervised classifier was trained.


Main output:

{observation_output_path}

Experiment statistics:

{experiment_output_path}

Temporal statistics:

{window_output_path}

Condition windows:

{condition_window_output_path}

Anomaly summary:

{anomaly_summary_path}
"""
)


print("=" * 70)

print("""
NEXT STEP:

Review the Strawberry sensor-condition results.

Then we can decide whether the data supports
a meaningful freshness/condition index and how
to present it in the final AI Food Quality Platform.

After Strawberry is completed, we will move to:

    Tomato spectral signatures
""")

print("=" * 70)