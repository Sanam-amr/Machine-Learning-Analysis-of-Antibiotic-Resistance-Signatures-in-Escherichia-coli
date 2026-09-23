import pandas as pd
from pathlib import Path

# ============================================================
# STEP 24
# PREPARE REDUCED FEATURE DATASET
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

train_file = (
    PROJECT_ROOT
    / "data"
    / "ml_train.csv"
)

test_file = (
    PROJECT_ROOT
    / "data"
    / "ml_test.csv"
)

correlation_file = (
    PROJECT_ROOT
    / "results"
    / "high_correlation_pairs.csv"
)

train_output = (
    PROJECT_ROOT
    / "data"
    / "ml_train_reduced.csv"
)

test_output = (
    PROJECT_ROOT
    / "data"
    / "ml_test_reduced.csv"
)

print("Loading datasets...")

train = pd.read_csv(train_file)
test = pd.read_csv(test_file)
correlations = pd.read_csv(correlation_file)

# ------------------------------------------------------------
# Remove the second feature from each highly correlated pair
# ------------------------------------------------------------

features_to_remove = sorted(
    set(
        correlations["feature_2"]
    )
)

print("\nFeatures selected for removal:")
for feature in features_to_remove:
    print(" -", feature)

# ------------------------------------------------------------
# Remove only features that actually exist
# ------------------------------------------------------------

features_to_remove = [
    feature
    for feature in features_to_remove
    if feature in train.columns
]

train_reduced = train.drop(
    columns=features_to_remove
)

test_reduced = test.drop(
    columns=features_to_remove
)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

train_reduced.to_csv(
    train_output,
    index=False
)

test_reduced.to_csv(
    test_output,
    index=False
)

# ------------------------------------------------------------
# Report
# ------------------------------------------------------------

print("\n============================================================")
print("REDUCED DATASET")
print("============================================================")

print("Original training columns:", train.shape[1])
print("Reduced training columns:", train_reduced.shape[1])

print("Original test columns:", test.shape[1])
print("Reduced test columns:", test_reduced.shape[1])

print("\nTraining isolates:", len(train_reduced))
print("Test isolates:", len(test_reduced))

print("\nReduced training dataset saved to:")
print(train_output)

print("\nReduced test dataset saved to:")
print(test_output)