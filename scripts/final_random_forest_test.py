import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# --------------------------------------------------
# Project paths
# --------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"

TRAIN_FILE = DATA_DIR / "ml_train.csv"
TEST_FILE = DATA_DIR / "ml_test.csv"

OUTPUT_FILE = RESULTS_DIR / "random_forest_test_results.csv"
PREDICTIONS_FILE = RESULTS_DIR / "random_forest_test_predictions.csv"


# --------------------------------------------------
# Load datasets
# --------------------------------------------------

print("Loading training dataset...")
train = pd.read_csv(TRAIN_FILE)

print("Loading untouched test dataset...")
test = pd.read_csv(TEST_FILE)

print("Training shape:", train.shape)
print("Test shape:", test.shape)


# --------------------------------------------------
# Separate features and phenotype
# --------------------------------------------------

feature_columns = [
    column
    for column in train.columns
    if column not in ["genome_id", "cip_resistant"]
]

X_train = train[feature_columns]
y_train = train["cip_resistant"]

X_test = test[feature_columns]
y_test = test["cip_resistant"]


print("\nNumber of genomic features:", len(feature_columns))
print("Training isolates:", len(train))
print("Test isolates:", len(test))


# --------------------------------------------------
# Train final Random Forest
# --------------------------------------------------

print("\nTraining final Random Forest on all training isolates...")

model = RandomForestClassifier(
    n_estimators=500,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


# --------------------------------------------------
# Predict test set
# --------------------------------------------------

print("Evaluating on untouched test set...")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


# --------------------------------------------------
# Calculate metrics
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

balanced_accuracy = balanced_accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_prob
)


# --------------------------------------------------
# Confusion matrix
# --------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

tn, fp, fn, tp = cm.ravel()

specificity = tn / (tn + fp)


# --------------------------------------------------
# Create results table
# --------------------------------------------------

results = pd.DataFrame([
    {
        "metric": "accuracy",
        "value": accuracy
    },
    {
        "metric": "balanced_accuracy",
        "value": balanced_accuracy
    },
    {
        "metric": "precision",
        "value": precision
    },
    {
        "metric": "recall",
        "value": recall
    },
    {
        "metric": "specificity",
        "value": specificity
    },
    {
        "metric": "f1",
        "value": f1
    },
    {
        "metric": "roc_auc",
        "value": roc_auc
    }
])


# --------------------------------------------------
# Save metrics
# --------------------------------------------------

results.to_csv(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# Save predictions
# --------------------------------------------------

predictions = pd.DataFrame({
    "genome_id": test["genome_id"],
    "actual_cip_resistant": y_test,
    "predicted_cip_resistant": y_pred,
    "resistance_probability": y_prob
})

predictions.to_csv(
    PREDICTIONS_FILE,
    index=False
)


# --------------------------------------------------
# Display results
# --------------------------------------------------

print("\n" + "=" * 60)
print("FINAL RANDOM FOREST — INDEPENDENT TEST RESULTS")
print("=" * 60)

print(
    results.to_string(index=False)
)

print("\nConfusion matrix:")
print(cm)

print("\nConfusion matrix interpretation:")
print("True negatives:", tn)
print("False positives:", fp)
print("False negatives:", fn)
print("True positives:", tp)

print("\nResults saved to:")
print(OUTPUT_FILE)

print("\nTest predictions saved to:")
print(PREDICTIONS_FILE)

print("\nIMPORTANT:")
print("The 145-isolate test set was not used during model development.")