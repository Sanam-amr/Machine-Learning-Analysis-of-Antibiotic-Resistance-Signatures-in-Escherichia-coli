import pandas as pd
from pathlib import Path

# ============================================================
# STEP 23
# IDENTIFY HIGHLY CORRELATED KEY FEATURES
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

input_file = (
    PROJECT_ROOT
    / "results"
    / "key_feature_correlation.csv"
)

output_file = (
    PROJECT_ROOT
    / "results"
    / "high_correlation_pairs.csv"
)

print("Loading correlation matrix...")

corr = pd.read_csv(
    input_file,
    index_col=0
)

# ------------------------------------------------------------
# Extract unique feature pairs
# ------------------------------------------------------------

pairs = []

features = corr.columns.tolist()

for i in range(len(features)):

    for j in range(i + 1, len(features)):

        feature_1 = features[i]
        feature_2 = features[j]

        correlation = corr.loc[
            feature_1,
            feature_2
        ]

        if abs(correlation) >= 0.80:

            pairs.append(
                {
                    "feature_1": feature_1,
                    "feature_2": feature_2,
                    "correlation": correlation,
                    "absolute_correlation": abs(correlation)
                }
            )

# ------------------------------------------------------------
# Create table
# ------------------------------------------------------------

result = pd.DataFrame(pairs)

if len(result) > 0:

    result = result.sort_values(
        "absolute_correlation",
        ascending=False
    )

else:

    print("\nNo feature pairs reached |correlation| >= 0.80.")

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

result.to_csv(
    output_file,
    index=False
)

print("\n============================================================")
print("HIGHLY CORRELATED FEATURE PAIRS")
print("============================================================")

if len(result) > 0:
    print(result.to_string(index=False))

print("\nNumber of highly correlated pairs:", len(result))

print("\nResults saved to:")
print(output_file)