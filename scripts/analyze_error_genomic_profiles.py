import pandas as pd
from pathlib import Path

# ============================================================
# STEP 21
# ANALYZE GENOMIC PROFILES OF MISCLASSIFIED ISOLATES
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

test_file = (
    PROJECT_ROOT
    / "data"
    / "ml_test.csv"
)

error_file = (
    PROJECT_ROOT
    / "results"
    / "prediction_confidence_analysis.csv"
)

output_file = (
    PROJECT_ROOT
    / "results"
    / "misclassified_genomic_profiles.csv"
)

print("Loading test genomic data...")

test = pd.read_csv(test_file)

print("Loading prediction analysis...")

errors = pd.read_csv(error_file)

# ------------------------------------------------------------
# Keep only misclassified isolates
# ------------------------------------------------------------

errors = errors[
    errors["prediction_correct"] == False
].copy()

error_ids = errors["genome_id"].tolist()

print("\nNumber of misclassified isolates:", len(error_ids))

# ------------------------------------------------------------
# Identify genomic feature columns
# ------------------------------------------------------------

metadata_columns = [
    "genome_id",
    "cip_resistant"
]

feature_columns = [
    col for col in test.columns
    if col not in metadata_columns
]

# ------------------------------------------------------------
# Extract genomic profiles
# ------------------------------------------------------------

profiles = test[
    test["genome_id"].isin(error_ids)
].copy()

# ------------------------------------------------------------
# Add prediction information
# ------------------------------------------------------------

profiles = profiles.merge(
    errors[
        [
            "genome_id",
            "predicted_cip_resistant",
            "resistance_probability",
            "prediction_confidence",
            "prediction_type"
        ]
    ],
    on="genome_id",
    how="left"
)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

profiles.to_csv(
    output_file,
    index=False
)

print("\nMisclassified genomic profiles saved to:")
print(output_file)

# ------------------------------------------------------------
# Display summary
# ------------------------------------------------------------

print("\n============================================================")
print("MISCLASSIFIED ISOLATE SUMMARY")
print("============================================================")

print(
    profiles[
        [
            "genome_id",
            "cip_resistant",
            "predicted_cip_resistant",
            "resistance_probability",
            "prediction_confidence",
            "prediction_type"
        ]
    ]
    .sort_values(
        "prediction_confidence",
        ascending=False
    )
    .to_string(index=False)
)