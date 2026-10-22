import os
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ======================================================================
# STRAWBERRY SENSOR PREPROCESSING + PCA
# ======================================================================

print("=" * 70)
print("STRAWBERRY SENSOR PREPROCESSING + PCA")
print("=" * 70)

print("""
This experiment uses the two available Strawberry sensor datasets:

    exp-1-plant.csv
    exp-2-plant.csv

No artificial target/quality label will be created.

Purpose:

    Raw sensor data
          ->
    combine experiments
          ->
    remove constant features
          ->
    exclude metadata/time fields
          ->
    sensor feature matrix
          ->
    numerical validation
          ->
    standardization
          ->
    PCA
          ->
    explained variance analysis
          ->
    PCA-transformed dataset
""")


# ======================================================================
# PROJECT PATHS
# ======================================================================

PROJECT_ROOT = r"D:\Projects\AI-Food-Quality-Platform"


DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "raw",
    "hyperspectral",
    "strawberry_sensor"
)


OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "datasets",
    "processed",
    "strawberry_sensor"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ======================================================================
# INPUT FILES
# ======================================================================

FILES = [

    "exp-1-plant.csv",

    "exp-2-plant.csv"

]


# ======================================================================
# CONSTANT FEATURES IDENTIFIED DURING ANALYSIS
# ======================================================================

CONSTANT_FEATURES = [

    "EBal?",
    "Area",
    "StmRat",
    "f_parin",
    "f_parout",
    "Status",
    "HsMch",
    "alphaK",
    "BLCslope",
    "BLC_1",
    "BLCond",
    "BLCoffst",
    "CsMch"

]


# ======================================================================
# NON-SENSOR / METADATA COLUMNS
# ======================================================================

METADATA_COLUMNS = [

    "HHMMSS",
    "experiment",
    "FTime"

]


# ======================================================================
# LOAD DATASETS
# ======================================================================

print("\n" + "=" * 70)
print("LOADING STRAWBERRY SENSOR DATA")
print("=" * 70)


dataframes = []


for filename in FILES:

    filepath = os.path.join(
        DATA_DIR,
        filename
    )


    print("\nLoading:")
    print(filepath)


    if not os.path.exists(filepath):

        raise FileNotFoundError(
            f"File not found:\n{filepath}"
        )


    df = pd.read_csv(
        filepath,
        low_memory=False
    )


    experiment_name = filename.replace(
        ".csv",
        ""
    )


    df["experiment"] = experiment_name


    dataframes.append(
        df
    )


    print(
        "Shape:",
        df.shape
    )


# ======================================================================
# COMBINE DATASETS
# ======================================================================

print("\n" + "=" * 70)
print("COMBINING EXPERIMENTS")
print("=" * 70)


combined_df = pd.concat(
    dataframes,
    ignore_index=True
)


print(
    "\nCombined shape:",
    combined_df.shape
)


print(
    "\nExperiment distribution:"
)


print(
    combined_df[
        "experiment"
    ].value_counts()
)


# ======================================================================
# TIMESTAMP VALIDATION
# ======================================================================

print("\n" + "=" * 70)
print("TIMESTAMP VALIDATION")
print("=" * 70)


if "HHMMSS" in combined_df.columns:

    timestamp_series = pd.to_datetime(
        combined_df["HHMMSS"],
        errors="coerce"
    )


    invalid_timestamps = (
        timestamp_series.isna().sum()
    )


    print(
        "\nTimestamp column: HHMMSS"
    )


    print(
        "Invalid timestamps:",
        invalid_timestamps
    )


    if invalid_timestamps > 0:

        print(
            "\nWARNING:"
        )

        print(
            "Some timestamp values could not be parsed."
        )

    else:

        print(
            "\nTimestamp validation: PASS"
        )


    print(
        "\nFirst timestamps:"
    )


    print(
        timestamp_series.head(
            10
        ).to_string(
            index=False
        )
    )


else:

    print(
        "\nHHMMSS column not found."
    )


# ======================================================================
# ORIGINAL COLUMN COUNT
# ======================================================================

print("\n" + "=" * 70)
print("ORIGINAL DATASET STRUCTURE")
print("=" * 70)


print(
    "\nTotal rows:",
    len(combined_df)
)


print(
    "Total columns:",
    len(combined_df.columns)
)


print(
    "\nOriginal columns:"
)


for i, column in enumerate(
    combined_df.columns,
    start=1
):

    print(
        f"{i:3d}. {column}"
    )


# ======================================================================
# VERIFY CONSTANT FEATURES
# ======================================================================

print("\n" + "=" * 70)
print("VERIFYING CONSTANT FEATURES")
print("=" * 70)


verified_constant_features = []


for feature in CONSTANT_FEATURES:

    if feature not in combined_df.columns:

        print(
            f"\nWARNING: {feature} not found."
        )

        continue


    unique_count = combined_df[
        feature
    ].nunique(
        dropna=True
    )


    print(
        f"{feature:12s} -> "
        f"{unique_count} unique value(s)"
    )


    if unique_count <= 1:

        verified_constant_features.append(
            feature
        )


print(
    "\nVerified constant features:",
    len(verified_constant_features)
)


# ======================================================================
# REMOVE CONSTANT FEATURES
# ======================================================================

print("\n" + "=" * 70)
print("REMOVING CONSTANT FEATURES")
print("=" * 70)


clean_df = combined_df.drop(
    columns=verified_constant_features,
    errors="ignore"
).copy()


print(
    "\nRows after constant-feature removal:",
    clean_df.shape[0]
)


print(
    "Columns after constant-feature removal:",
    clean_df.shape[1]
)


print(
    "\nRemoved features:"
)


for feature in verified_constant_features:

    print(
        feature
    )


# ======================================================================
# IDENTIFY CANDIDATE SENSOR FEATURES
# ======================================================================

print("\n" + "=" * 70)
print("IDENTIFYING SENSOR FEATURES")
print("=" * 70)


numeric_columns = clean_df.select_dtypes(
    include=np.number
).columns.tolist()


sensor_features = [

    column

    for column in numeric_columns

    if column not in METADATA_COLUMNS

]


print(
    "\nNumeric columns:",
    len(numeric_columns)
)


print(
    "Sensor/measurement features:",
    len(sensor_features)
)


print(
    "\nSensor features:"
)


for feature in sensor_features:

    print(
        feature
    )


# ======================================================================
# EXPECTED FEATURE COUNT
# ======================================================================

print("\n" + "=" * 70)
print("FEATURE COUNT VALIDATION")
print("=" * 70)


expected_sensor_features = 63


print(
    "\nExpected variable sensor features:",
    expected_sensor_features
)


print(
    "Detected variable sensor features:",
    len(sensor_features)
)


if len(sensor_features) != expected_sensor_features:

    print(
        "\nWARNING:"
    )

    print(
        "Detected sensor feature count differs "
        "from the expected 63."
    )

    print(
        "The script will continue using the "
        "features actually detected."
    )

else:

    print(
        "\nFeature count validation: PASS"
    )


# ======================================================================
# CHECK SENSOR DATA TYPES
# ======================================================================

print("\n" + "=" * 70)
print("SENSOR DATA TYPE VALIDATION")
print("=" * 70)


non_numeric_sensor_features = []


for feature in sensor_features:

    if not pd.api.types.is_numeric_dtype(
        clean_df[feature]
    ):

        non_numeric_sensor_features.append(
            feature
        )


if non_numeric_sensor_features:

    print(
        "\nNon-numeric sensor features detected:"
    )

    for feature in non_numeric_sensor_features:

        print(
            feature
        )


    raise ValueError(
        "Sensor feature matrix contains "
        "non-numeric columns."
    )


print(
    "\nAll sensor features are numeric."
)


# ======================================================================
# MISSING VALUE ANALYSIS
# ======================================================================

print("\n" + "=" * 70)
print("MISSING VALUE ANALYSIS")
print("=" * 70)


missing_counts = clean_df[
    sensor_features
].isna().sum()


total_missing = (
    missing_counts.sum()
)


print(
    "\nTotal missing sensor values:",
    int(total_missing)
)


if total_missing > 0:

    print(
        "\nMissing values by feature:"
    )

    print(
        missing_counts[
            missing_counts > 0
        ].to_string()
    )

else:

    print(
        "\nMissing-value validation: PASS"
    )


# ======================================================================
# INFINITE VALUE ANALYSIS
# ======================================================================

print("\n" + "=" * 70)
print("INFINITE VALUE ANALYSIS")
print("=" * 70)


infinite_counts = pd.Series(
    np.isinf(
        clean_df[
            sensor_features
        ].to_numpy(
            dtype=np.float64
        )
    ).sum(
        axis=0
    ),
    index=sensor_features
)


total_infinite = (
    infinite_counts.sum()
)


print(
    "\nTotal infinite sensor values:",
    int(total_infinite)
)


if total_infinite > 0:

    print(
        "\nInfinite values by feature:"
    )

    print(
        infinite_counts[
            infinite_counts > 0
        ].to_string()
    )

else:

    print(
        "\nInfinite-value validation: PASS"
    )


# ======================================================================
# HANDLE MISSING / INFINITE VALUES
# ======================================================================

print("\n" + "=" * 70)
print("NUMERICAL CLEANING")
print("=" * 70)


sensor_df = clean_df[
    sensor_features
].copy()


# Convert any unexpected values to NaN

sensor_df = sensor_df.replace(
    [np.inf, -np.inf],
    np.nan
)


remaining_missing = (
    sensor_df.isna().sum().sum()
)


print(
    "\nMissing/invalid values requiring handling:",
    int(remaining_missing)
)


if remaining_missing > 0:

    print(
        "\nApplying median imputation "
        "using the combined sensor dataset..."
    )


    for feature in sensor_features:

        if sensor_df[
            feature
        ].isna().any():

            median_value = sensor_df[
                feature
            ].median()


            if pd.isna(
                median_value
            ):

                raise ValueError(
                    f"Unable to calculate median "
                    f"for feature: {feature}"
                )


            sensor_df[
                feature
            ] = sensor_df[
                feature
            ].fillna(
                median_value
            )


    print(
        "Median imputation completed."
    )

else:

    print(
        "\nNo imputation required."
    )


# ======================================================================
# FINAL NUMERICAL VALIDATION
# ======================================================================

print("\n" + "=" * 70)
print("FINAL NUMERICAL VALIDATION")
print("=" * 70)


X = sensor_df[
    sensor_features
].to_numpy(
    dtype=np.float64
)


nan_count = np.isnan(
    X
).sum()


inf_count = np.isinf(
    X
).sum()


print(
    "\nMatrix shape:",
    X.shape
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
        "NaN values remain after cleaning."
    )


if inf_count > 0:

    raise ValueError(
        "Infinite values remain after cleaning."
    )


print(
    "\nNumerical validation: PASS"
)


# ======================================================================
# FEATURE STATISTICS BEFORE STANDARDIZATION
# ======================================================================

print("\n" + "=" * 70)
print("FEATURE STATISTICS")
print("=" * 70)


feature_statistics = pd.DataFrame({

    "feature":
        sensor_features,

    "mean":
        sensor_df.mean().values,

    "std":
        sensor_df.std().values,

    "minimum":
        sensor_df.min().values,

    "maximum":
        sensor_df.max().values,

    "median":
        sensor_df.median().values

})


print(
    "\nFirst feature statistics:"
)


print(
    feature_statistics.head(
        20
    ).to_string(
        index=False
    )
)


# ======================================================================
# STANDARDIZATION
# ======================================================================

print("\n" + "=" * 70)
print("STANDARDIZATION")
print("=" * 70)


print(
    "\nFitting StandardScaler..."
)


scaler = StandardScaler()


X_scaled = scaler.fit_transform(
    X
)


print(
    "Standardization completed."
)


print(
    "\nStandardized matrix shape:",
    X_scaled.shape
)


# ======================================================================
# STANDARDIZATION VALIDATION
# ======================================================================

print("\n" + "=" * 70)
print("STANDARDIZATION VALIDATION")
print("=" * 70)


standardized_means = X_scaled.mean(
    axis=0
)


standardized_stds = X_scaled.std(
    axis=0
)


max_abs_mean = np.max(
    np.abs(
        standardized_means
    )
)


max_std_difference = np.max(
    np.abs(
        standardized_stds - 1.0
    )
)


print(
    "\nMaximum absolute feature mean:",
    max_abs_mean
)


print(
    "Maximum deviation from unit standard deviation:",
    max_std_difference
)


if max_abs_mean > 1e-6:

    print(
        "\nWARNING:"
    )

    print(
        "Some standardized feature means "
        "are not exactly zero due to numerical precision."
    )


print(
    "\nStandardization validation completed."
)


# ======================================================================
# PCA
# ======================================================================

print("\n" + "=" * 70)
print("PRINCIPAL COMPONENT ANALYSIS")
print("=" * 70)


n_features = X_scaled.shape[1]


print(
    "\nNumber of input sensor features:",
    n_features
)


print(
    "Maximum possible PCA components:",
    n_features
)


print(
    "\nFitting full PCA..."
)


pca = PCA(
    n_components=n_features
)


X_pca = pca.fit_transform(
    X_scaled
)


print(
    "PCA fitting completed."
)


print(
    "\nPCA matrix shape:",
    X_pca.shape
)


# ======================================================================
# EXPLAINED VARIANCE
# ======================================================================

print("\n" + "=" * 70)
print("PCA EXPLAINED VARIANCE")
print("=" * 70)


explained_variance_ratio = (
    pca.explained_variance_ratio_
)


cumulative_variance = np.cumsum(
    explained_variance_ratio
)


explained_variance_df = pd.DataFrame({

    "PC":
        [
            f"PC{i}"
            for i in range(
                1,
                len(
                    explained_variance_ratio
                ) + 1
            )
        ],

    "explained_variance_ratio":
        explained_variance_ratio,

    "explained_variance_percent":
        explained_variance_ratio * 100,

    "cumulative_variance_ratio":
        cumulative_variance,

    "cumulative_variance_percent":
        cumulative_variance * 100

})


print(
    "\nPCA explained variance:"
)


print(
    explained_variance_df.to_string(
        index=False
    )
)


# ======================================================================
# VARIANCE THRESHOLD ANALYSIS
# ======================================================================

print("\n" + "=" * 70)
print("PCA VARIANCE THRESHOLD ANALYSIS")
print("=" * 70)


variance_thresholds = [

    0.80,
    0.90,
    0.95,
    0.99

]


threshold_results = []


for threshold in variance_thresholds:

    components_required = (
        np.argmax(
            cumulative_variance
            >=
            threshold
        )
        + 1
    )


    achieved_variance = (
        cumulative_variance[
            components_required - 1
        ]
    )


    threshold_results.append({

        "target_variance":
            threshold,

        "target_variance_percent":
            threshold * 100,

        "components_required":
            components_required,

        "achieved_variance":
            achieved_variance,

        "achieved_variance_percent":
            achieved_variance * 100

    })


    print(
        f"\n{threshold * 100:.0f}% variance:"
    )


    print(
        "Required PCs:",
        components_required
    )


    print(
        "Achieved variance:",
        f"{achieved_variance * 100:.4f}%"
    )


threshold_df = pd.DataFrame(
    threshold_results
)


# ======================================================================
# TOP PCA COMPONENTS
# ======================================================================

print("\n" + "=" * 70)
print("TOP PCA COMPONENTS")
print("=" * 70)


top_n = min(
    20,
    len(explained_variance_df)
)


print(
    "\nFirst PCA components:"
)


print(
    explained_variance_df.head(
        top_n
    ).to_string(
        index=False
    )
)


# ======================================================================
# CREATE PCA DATAFRAME
# ======================================================================

print("\n" + "=" * 70)
print("CREATING PCA DATASET")
print("=" * 70)


pca_columns = [

    f"PC{i}"

    for i in range(
        1,
        n_features + 1
    )

]


pca_df = pd.DataFrame(
    X_pca,
    columns=pca_columns
)


# Add metadata back

pca_df.insert(
    0,
    "experiment",
    combined_df[
        "experiment"
    ].values
)


pca_df.insert(
    0,
    "HHMMSS",
    combined_df[
        "HHMMSS"
    ].values
)


if "FTime" in combined_df.columns:

    pca_df.insert(
        2,
        "FTime",
        combined_df[
            "FTime"
        ].values
    )


print(
    "\nPCA dataset shape:",
    pca_df.shape
)


# ======================================================================
# SAVE CLEAN SENSOR DATA
# ======================================================================

clean_sensor_path = os.path.join(

    OUTPUT_DIR,

    "strawberry_sensor_clean_63_features.csv"
)


clean_output_df = pd.DataFrame(
    X,
    columns=sensor_features
)


clean_output_df.insert(
    0,
    "experiment",
    combined_df[
        "experiment"
    ].values
)


clean_output_df.insert(
    0,
    "HHMMSS",
    combined_df[
        "HHMMSS"
    ].values
)


if "FTime" in combined_df.columns:

    clean_output_df.insert(
        2,
        "FTime",
        combined_df[
            "FTime"
        ].values
    )


clean_output_df.to_csv(
    clean_sensor_path,
    index=False
)


print(
    "\nSaved cleaned sensor dataset:"
)


print(
    clean_sensor_path
)


# ======================================================================
# SAVE PCA DATASET
# ======================================================================

pca_output_path = os.path.join(

    OUTPUT_DIR,

    "strawberry_sensor_pca.csv"
)


pca_df.to_csv(
    pca_output_path,
    index=False
)


print(
    "\nSaved PCA dataset:"
)


print(
    pca_output_path
)


# ======================================================================
# SAVE EXPLAINED VARIANCE
# ======================================================================

variance_path = os.path.join(

    OUTPUT_DIR,

    "strawberry_sensor_pca_explained_variance.csv"
)


explained_variance_df.to_csv(

    variance_path,

    index=False
)


print(
    "\nSaved explained variance:"
)


print(
    variance_path
)


# ======================================================================
# SAVE VARIANCE THRESHOLD TABLE
# ======================================================================

threshold_path = os.path.join(

    OUTPUT_DIR,

    "strawberry_sensor_pca_variance_thresholds.csv"
)


threshold_df.to_csv(

    threshold_path,

    index=False
)


print(
    "\nSaved variance threshold analysis:"
)


print(
    threshold_path
)


# ======================================================================
# SAVE FEATURE STATISTICS
# ======================================================================

feature_statistics_path = os.path.join(

    OUTPUT_DIR,

    "strawberry_sensor_feature_statistics.csv"
)


feature_statistics.to_csv(

    feature_statistics_path,

    index=False
)


print(
    "\nSaved feature statistics:"
)


print(
    feature_statistics_path
)


# ======================================================================
# SAVE STANDARD SCALER
# ======================================================================

scaler_path = os.path.join(

    OUTPUT_DIR,

    "strawberry_sensor_standard_scaler.joblib"
)


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
# SAVE PCA MODEL
# ======================================================================

pca_path = os.path.join(

    OUTPUT_DIR,

    "strawberry_sensor_pca_model.joblib"
)


joblib.dump(

    pca,

    pca_path
)


print(
    "\nSaved PCA model:"
)


print(
    pca_path
)


# ======================================================================
# SAVE SELECTED FEATURE LIST
# ======================================================================

feature_list_path = os.path.join(

    OUTPUT_DIR,

    "strawberry_sensor_features.txt"
)


with open(
    feature_list_path,
    "w",
    encoding="utf-8"
) as file:

    for feature in sensor_features:

        file.write(
            feature + "\n"
        )


print(
    "\nSaved sensor feature list:"
)


print(
    feature_list_path
)


# ======================================================================
# SAVE REMOVED FEATURE LIST
# ======================================================================

removed_features_path = os.path.join(

    OUTPUT_DIR,

    "strawberry_sensor_removed_constant_features.txt"
)


with open(
    removed_features_path,
    "w",
    encoding="utf-8"
) as file:

    for feature in verified_constant_features:

        file.write(
            feature + "\n"
        )


print(
    "\nSaved removed-feature list:"
)


print(
    removed_features_path
)


# ======================================================================
# FINAL SUMMARY
# ======================================================================

print("\n" + "=" * 70)
print("STRAWBERRY SENSOR PREPROCESSING + PCA COMPLETED")
print("=" * 70)


print(
f"""
Experiments                  : 2

Total observations           : {len(combined_df)}

Original columns             : {len(combined_df.columns)}

Constant features removed   : {len(verified_constant_features)}

Variable sensor features    : {len(sensor_features)}

PCA components calculated    : {n_features}


PCA variance thresholds:

80% variance PCs             : {
    threshold_df.loc[
        threshold_df["target_variance"] == 0.80,
        "components_required"
    ].iloc[0]
}

90% variance PCs             : {
    threshold_df.loc[
        threshold_df["target_variance"] == 0.90,
        "components_required"
    ].iloc[0]
}

95% variance PCs             : {
    threshold_df.loc[
        threshold_df["target_variance"] == 0.95,
        "components_required"
    ].iloc[0]
}

99% variance PCs             : {
    threshold_df.loc[
        threshold_df["target_variance"] == 0.99,
        "components_required"
    ].iloc[0]
}


No artificial target created.

No supervised classifier trained.


Clean sensor dataset:

{clean_sensor_path}


PCA dataset:

{pca_output_path}


Explained variance:

{variance_path}


Variance thresholds:

{threshold_path}


StandardScaler:

{scaler_path}


PCA model:

{pca_path}
"""
)


print(
"=" * 70
)


print("""
NEXT STEP:

Review the PCA explained-variance results.

Then select an appropriate PCA dimensionality
for the Strawberry sensor analysis.

After that:

    PCA
      ->
    sensor-pattern analysis
      ->
    unsupervised modelling / freshness-condition index

No artificial freshness labels will be introduced.
""")


print("=" * 70)