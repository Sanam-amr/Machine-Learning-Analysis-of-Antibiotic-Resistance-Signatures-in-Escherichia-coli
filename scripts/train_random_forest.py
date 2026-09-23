import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate


# --------------------------------------------------
# Project paths
# --------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"

TRAIN_FILE = DATA_DIR / "ml_train.csv"
OUTPUT_FILE = RESULTS_DIR / "random_forest_cv_results.csv"


# --------------------------------------------------
# Load training dataset
# --------------------------------------------------

print("Loading training dataset...")

train = pd.read_csv(TRAIN_FILE)

print("Training dataset shape:", train.shape)


# --------------------------------------------------
# Separate features and phenotype
# --------------------------------------------------

feature_columns = [
    column
    for column in train.columns
    if column not in ["genome_id", "cip_resistant"]
]

X = train[feature_columns]
y = train["cip_resistant"]


print("Number of genomic features:", len(feature_columns))
print("Number of isolates:", len(train))


# --------------------------------------------------
# Random Forest model
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=500,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


# --------------------------------------------------
# Stratified 5-fold cross-validation
# --------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


scoring = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}


print("\nRunning 5-fold cross-validation...")


cv_results = cross_validate(
    model,
    X,
    y,
    cv=cv,
    scoring=scoring,
    return_train_score=False
)


# --------------------------------------------------
# Summarize results
# --------------------------------------------------

summary = []

for metric in scoring.keys():

    values = cv_results[f"test_{metric}"]

    summary.append({
        "metric": metric,
        "mean": np.mean(values),
        "std": np.std(values),
        "fold_1": values[0],
        "fold_2": values[1],
        "fold_3": values[2],
        "fold_4": values[3],
        "fold_5": values[4]
    })


results_df = pd.DataFrame(summary)


# --------------------------------------------------
# Save results
# --------------------------------------------------

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# Display results
# --------------------------------------------------

print("\n" + "=" * 60)
print("RANDOM FOREST CROSS-VALIDATION RESULTS")
print("=" * 60)

print(
    results_df[
        ["metric", "mean", "std"]
    ].to_string(index=False)
)

print("\nDetailed fold results:")

print(
    results_df.to_string(index=False)
)

print("\nResults saved to:")

print(OUTPUT_FILE)