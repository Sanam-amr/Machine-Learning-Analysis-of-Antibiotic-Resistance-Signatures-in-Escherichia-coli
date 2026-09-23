import pandas as pd
from pathlib import Path

# ============================================================
# STEP 18
# ANALYZE PREDICTION CONFIDENCE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

input_file = (
    PROJECT_ROOT
    / "results"
    / "random_forest_test_predictions.csv"
)

output_file = (
    PROJECT_ROOT
    / "results"
    / "prediction_confidence_analysis.csv"
)

print("Loading predictions...")

df = pd.read_csv(input_file)

print("\n============================================================")
print("PREDICTION CONFIDENCE ANALYSIS")
print("============================================================")

print("\nColumns:")
print(df.columns.tolist())

# ------------------------------------------------------------
# Check required columns
# ------------------------------------------------------------

required_columns = [
    "genome_id",
    "actual_cip_resistant",
    "predicted_cip_resistant",
    "resistance_probability"
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

# ------------------------------------------------------------
# Determine whether prediction was correct
# ------------------------------------------------------------

df["prediction_correct"] = (
    df["actual_cip_resistant"]
    == df["predicted_cip_resistant"]
)

# ------------------------------------------------------------
# Calculate confidence
#
# Distance from 0.5:
# 0.5 = low confidence
# 0 or 1 = high confidence
# ------------------------------------------------------------

df["prediction_confidence"] = (
    abs(df["resistance_probability"] - 0.5) * 2
)

# ------------------------------------------------------------
# Identify prediction type
# ------------------------------------------------------------

def classify_prediction(row):

    actual = row["actual_cip_resistant"]
    predicted = row["predicted_cip_resistant"]

    if actual == 0 and predicted == 0:
        return "True Negative"

    elif actual == 0 and predicted == 1:
        return "False Positive"

    elif actual == 1 and predicted == 0:
        return "False Negative"

    elif actual == 1 and predicted == 1:
        return "True Positive"


df["prediction_type"] = df.apply(
    classify_prediction,
    axis=1
)

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

errors = df[
    df["prediction_correct"] == False
].copy()

print("\nTotal test isolates:", len(df))
print("Correct predictions:", df["prediction_correct"].sum())
print("Incorrect predictions:", len(errors))

# ------------------------------------------------------------
# Display errors from highest to lowest confidence
# ------------------------------------------------------------

print("\n============================================================")
print("MISCLASSIFIED ISOLATES")
print("ORDERED BY MODEL CONFIDENCE")
print("============================================================")

error_columns = [
    "genome_id",
    "actual_cip_resistant",
    "predicted_cip_resistant",
    "resistance_probability",
    "prediction_confidence",
    "prediction_type"
]

print(
    errors[
        error_columns
    ]
    .sort_values(
        "prediction_confidence",
        ascending=False
    )
    .to_string(index=False)
)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

df.to_csv(
    output_file,
    index=False
)

print("\nConfidence analysis saved to:")
print(output_file)