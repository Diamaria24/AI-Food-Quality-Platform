import os
import numpy as np
import pandas as pd


# ======================================================================
# STRAWBERRY SENSOR FEATURE ANALYSIS
# ======================================================================

print("=" * 70)
print("STRAWBERRY SENSOR FEATURE ANALYSIS")
print("=" * 70)

print("""
Purpose:

    Analyze both Strawberry sensor experiments
    and identify:

    1. Constant columns
    2. Near-constant columns
    3. Useful variable sensor columns
    4. Missing / infinite values
    5. Extreme-value behaviour
    6. Highly correlated sensor features

No machine-learning model will be trained yet.
""")


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


FILES = [
    "exp-1-plant.csv",
    "exp-2-plant.csv"
]


# ======================================================================
# LOAD BOTH EXPERIMENTS
# ======================================================================

print("\n" + "=" * 70)
print("LOADING EXPERIMENTS")
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

    # Add experiment identifier
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
# COMBINE DATA
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
# TIMESTAMP ANALYSIS
# ======================================================================

print("\n" + "=" * 70)
print("TIMESTAMP ANALYSIS")
print("=" * 70)


if "HHMMSS" in combined_df.columns:

    print(
        "\nConverting HHMMSS to time..."
    )

    # Keep original column untouched.
    # Create a separate time column.

    combined_df[
        "time_parsed"
    ] = pd.to_datetime(

        combined_df[
            "HHMMSS"
        ].astype(str),

        format="%H%M%S",

        errors="coerce"
    )


    invalid_time = combined_df[
        "time_parsed"
    ].isna().sum()


    print(
        "\nInvalid timestamps:",
        invalid_time
    )


    print(
        "\nFirst timestamps:"
    )

    print(
        combined_df[
            [
                "HHMMSS",
                "time_parsed",
                "experiment"
            ]
        ].head(10).to_string(
            index=False
        )
    )

else:

    print(
        "\nHHMMSS column not found."
    )


# ======================================================================
# IDENTIFY NUMERIC FEATURES
# ======================================================================

print("\n" + "=" * 70)
print("IDENTIFYING NUMERIC SENSOR FEATURES")
print("=" * 70)


numeric_columns = combined_df.select_dtypes(
    include=np.number
).columns.tolist()


# Remove any columns that are identifiers / not actual sensors.

excluded_numeric = [
    "experiment"
]


sensor_columns = [

    column

    for column in numeric_columns

    if column not in excluded_numeric
]


print(
    "\nNumeric columns:",
    len(numeric_columns)
)

print(
    "Candidate sensor columns:",
    len(sensor_columns)
)


print(
    "\nCandidate sensor features:"
)

for column in sensor_columns:

    print(
        column
    )


# ======================================================================
# MISSING AND INFINITE VALUES
# ======================================================================

print("\n" + "=" * 70)
print("MISSING / INFINITE VALUES")
print("=" * 70)


missing_table = []

for column in sensor_columns:

    missing_count = combined_df[
        column
    ].isna().sum()

    infinite_count = np.isinf(
        pd.to_numeric(
            combined_df[column],
            errors="coerce"
        )
    ).sum()

    missing_table.append({

        "feature":
            column,

        "missing":
            int(missing_count),

        "infinite":
            int(infinite_count)

    })


missing_df = pd.DataFrame(
    missing_table
)


print(
    missing_df.to_string(
        index=False
    )
)


# ======================================================================
# VARIANCE ANALYSIS
# ======================================================================

print("\n" + "=" * 70)
print("VARIANCE ANALYSIS")
print("=" * 70)


variance_rows = []


for column in sensor_columns:

    series = pd.to_numeric(
        combined_df[column],
        errors="coerce"
    )

    variance_rows.append({

        "feature":
            column,

        "unique_values":
            int(
                series.nunique(
                    dropna=True
                )
            ),

        "mean":
            series.mean(),

        "std":
            series.std(),

        "variance":
            series.var(),

        "minimum":
            series.min(),

        "maximum":
            series.max(),

        "median":
            series.median(),

        "q01":
            series.quantile(0.01),

        "q99":
            series.quantile(0.99)

    })


variance_df = pd.DataFrame(
    variance_rows
)


# Sort by variance

variance_df = variance_df.sort_values(
    "variance",
    ascending=True
).reset_index(
    drop=True
)


print(
    "\nComplete variance analysis:"
)

print(
    variance_df.to_string(
        index=False
    )
)


# ======================================================================
# CONSTANT FEATURES
# ======================================================================

print("\n" + "=" * 70)
print("CONSTANT FEATURES")
print("=" * 70)


constant_features = variance_df[
    variance_df[
        "unique_values"
    ] <= 1
][
    "feature"
].tolist()


print(
    "\nNumber of constant features:",
    len(constant_features)
)


if constant_features:

    for feature in constant_features:

        print(
            feature
        )

else:

    print(
        "No constant features found."
    )


# ======================================================================
# NEAR-CONSTANT FEATURES
# ======================================================================

print("\n" + "=" * 70)
print("NEAR-CONSTANT FEATURES")
print("=" * 70)


# A feature is flagged as near-constant if
# 99.9% or more of its non-null observations
# contain the same value.


near_constant_rows = []


for column in sensor_columns:

    series = pd.to_numeric(
        combined_df[column],
        errors="coerce"
    ).dropna()


    if len(series) == 0:

        continue


    value_counts = (
        series
        .value_counts(
            normalize=True
        )
    )


    dominant_fraction = (
        value_counts.iloc[0]
    )


    near_constant_rows.append({

        "feature":
            column,

        "dominant_value":
            value_counts.index[0],

        "dominant_fraction":
            dominant_fraction,

        "unique_values":
            series.nunique()

    })


near_constant_df = pd.DataFrame(
    near_constant_rows
)


near_constant_df = near_constant_df.sort_values(
    "dominant_fraction",
    ascending=False
)


near_constant_features = near_constant_df[
    near_constant_df[
        "dominant_fraction"
    ] >= 0.999
][
    "feature"
].tolist()


print(
    "\nNear-constant threshold:",
    "99.9%"
)


print(
    "\nNumber of near-constant features:",
    len(near_constant_features)
)


if near_constant_features:

    print(
        "\nNear-constant features:"
    )

    print(
        near_constant_df[
            near_constant_df[
                "dominant_fraction"
            ] >= 0.999
        ].to_string(
            index=False
        )
    )

else:

    print(
        "No near-constant features found."
    )


# ======================================================================
# SENSOR FEATURE SUMMARY
# ======================================================================

print("\n" + "=" * 70)
print("SENSOR FEATURE SUMMARY")
print("=" * 70)


useful_features = [

    feature

    for feature in sensor_columns

    if feature not in constant_features

    and feature not in near_constant_features

]


print(
    "\nTotal candidate sensor features:",
    len(sensor_columns)
)


print(
    "Constant features:",
    len(constant_features)
)


print(
    "Near-constant features:",
    len(near_constant_features)
)


print(
    "Remaining variable features:",
    len(useful_features)
)


print(
    "\nRemaining variable sensor features:"
)

for feature in useful_features:

    print(
        feature
    )


# ======================================================================
# CORRELATION ANALYSIS
# ======================================================================

print("\n" + "=" * 70)
print("CORRELATION ANALYSIS")
print("=" * 70)


if len(useful_features) > 1:

    correlation_df = combined_df[
        useful_features
    ].corr()


    # Find highly correlated pairs.

    correlation_pairs = []


    for i in range(
        len(useful_features)
    ):

        for j in range(
            i + 1,
            len(useful_features)
        ):

            feature_a = useful_features[i]

            feature_b = useful_features[j]


            correlation = correlation_df.loc[
                feature_a,
                feature_b
            ]


            if abs(correlation) >= 0.95:

                correlation_pairs.append({

                    "feature_1":
                        feature_a,

                    "feature_2":
                        feature_b,

                    "correlation":
                        correlation

                })


    correlation_pairs_df = pd.DataFrame(
        correlation_pairs
    )


    if len(
        correlation_pairs_df
    ) > 0:

        correlation_pairs_df = (
            correlation_pairs_df
            .sort_values(
                "correlation",
                key=lambda x: x.abs(),
                ascending=False
            )
        )


        print(
            "\nHighly correlated pairs "
            "(absolute correlation >= 0.95):"
        )

        print(
            correlation_pairs_df.to_string(
                index=False
            )
        )

    else:

        correlation_pairs_df = pd.DataFrame(
            columns=[
                "feature_1",
                "feature_2",
                "correlation"
            ]
        )

        print(
            "\nNo feature pairs exceeded "
            "absolute correlation of 0.95."
        )

else:

    correlation_pairs_df = pd.DataFrame()

    print(
        "\nNot enough variable features "
        "for correlation analysis."
    )


# ======================================================================
# EXPERIMENT COMPARISON
# ======================================================================

print("\n" + "=" * 70)
print("EXPERIMENT COMPARISON")
print("=" * 70)


experiment_summary_rows = []


for experiment_name, df in zip(
    [
        "exp-1-plant",
        "exp-2-plant"
    ],
    dataframes
):

    experiment_summary_rows.append({

        "experiment":
            experiment_name,

        "rows":
            len(df),

        "columns":
            len(df.columns)

    })


experiment_summary_df = pd.DataFrame(
    experiment_summary_rows
)


print(
    experiment_summary_df.to_string(
        index=False
    )
)


# ======================================================================
# SAVE ANALYSIS FILES
# ======================================================================

print("\n" + "=" * 70)
print("SAVING FEATURE ANALYSIS")
print("=" * 70)


variance_path = os.path.join(
    OUTPUT_DIR,
    "strawberry_sensor_variance_analysis.csv"
)


variance_df.to_csv(
    variance_path,
    index=False
)


print(
    "\nSaved variance analysis:"
)

print(
    variance_path
)


missing_path = os.path.join(
    OUTPUT_DIR,
    "strawberry_sensor_missing_values.csv"
)


missing_df.to_csv(
    missing_path,
    index=False
)


print(
    "\nSaved missing-value analysis:"
)

print(
    missing_path
)


near_constant_path = os.path.join(
    OUTPUT_DIR,
    "strawberry_sensor_near_constant_features.csv"
)


near_constant_df.to_csv(
    near_constant_path,
    index=False
)


print(
    "\nSaved near-constant analysis:"
)

print(
    near_constant_path
)


if len(
    correlation_pairs_df
) > 0:

    correlation_path = os.path.join(
        OUTPUT_DIR,
        "strawberry_sensor_high_correlation_pairs.csv"
    )

    correlation_pairs_df.to_csv(
        correlation_path,
        index=False
    )

    print(
        "\nSaved correlation analysis:"
    )

    print(
        correlation_path
    )


# ======================================================================
# SAVE FEATURE LIST
# ======================================================================

feature_list_path = os.path.join(
    OUTPUT_DIR,
    "strawberry_sensor_selected_features.txt"
)


with open(
    feature_list_path,
    "w",
    encoding="utf-8"
) as file:

    for feature in useful_features:

        file.write(
            feature + "\n"
        )


print(
    "\nSaved selected feature list:"
)

print(
    feature_list_path
)


# ======================================================================
# SAVE COMBINED CLEAN BASE DATA
# ======================================================================

clean_base_columns = [

    "HHMMSS",
    "experiment"

] + useful_features


clean_base_df = combined_df[
    clean_base_columns
].copy()


clean_base_path = os.path.join(
    OUTPUT_DIR,
    "strawberry_sensor_combined_variable_features.csv"
)


clean_base_df.to_csv(
    clean_base_path,
    index=False
)


print(
    "\nSaved combined variable-feature dataset:"
)

print(
    clean_base_path
)


# ======================================================================
# FINAL SUMMARY
# ======================================================================

print("\n" + "=" * 70)
print("STRAWBERRY SENSOR FEATURE ANALYSIS COMPLETED")
print("=" * 70)


print(
f"""
Experiments                  : 2

Total observations           : {len(combined_df)}

Candidate sensor features   : {len(sensor_columns)}

Constant features            : {len(constant_features)}

Near-constant features       : {len(near_constant_features)}

Remaining variable features : {len(useful_features)}


IMPORTANT:

No target variable was created.

No classifier was trained.

No artificial freshness labels
were introduced.

The selected features will be
reviewed before the next preprocessing step.
"""
)


print("=" * 70)