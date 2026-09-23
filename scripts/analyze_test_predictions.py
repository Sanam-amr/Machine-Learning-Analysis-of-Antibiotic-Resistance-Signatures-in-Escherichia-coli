import pandas as pd
from pathlib import Path

# ============================================================
# ANALYZE FINAL RANDOM FOREST TEST PREDICTIONS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

input_file = PROJECT_ROOT / "results" / "random_forest_test_predictions.csv"
output_file = PROJECT_ROOT / "results" / "test_prediction_error_analysis.csv"

print("Loading test predictions...")

df = pd.read_csv(input_file)

print("\n============================================================")
print("TEST PREDICTION ANALYSIS")
print("============================================================")

print(f"Number of test isolates: {len(df)}")

print("\nColumns:")
print(df.columns.tolist())

# ------------------------------------------------------------
# Create prediction correctness
# ------------------------------------------------------------

df["correct"] = (
    df["actual_cip_resistant"]
    == df["predicted_cip_resistant"]
)

# ------------------------------------------------------------
# Identify error type
# ------------------------------------------------------------

def classify_error(row):

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


df["prediction_type"] = df.apply(classify_error, axis=1)

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print("\nPrediction summary:")
print(df["prediction_type"].value_counts())

# ------------------------------------------------------------
# Show false positives
# ------------------------------------------------------------

print("\n============================================================")
print("FALSE POSITIVES")
print("============================================================")

false_positive = df[
    df["prediction_type"] == "False Positive"
]

print(false_positive.to_string(index=False))

# ------------------------------------------------------------
# Show false negatives
# ------------------------------------------------------------

print("\n============================================================")
print("FALSE NEGATIVES")
print("============================================================")

false_negative = df[
    df["prediction_type"] == "False Negative"
]

print(false_negative.to_string(index=False))

# ------------------------------------------------------------
# Save complete analysis
# ------------------------------------------------------------

df.to_csv(output_file, index=False)

print("\nError analysis saved to:")
print(output_file)