import pandas as pd
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

# ============================================================
# STEP 25
# REDUCED RANDOM FOREST SENSITIVITY ANALYSIS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

train_file = (
    PROJECT_ROOT
    / "data"
    / "ml_train_reduced.csv"
)

test_file = (
    PROJECT_ROOT
    / "data"
    / "ml_test_reduced.csv"
)

output_file = (
    PROJECT_ROOT
    / "results"
    / "reduced_random_forest_test_results.csv"
)

print("Loading reduced datasets...")

train = pd.read_csv(train_file)
test = pd.read_csv(test_file)

# ------------------------------------------------------------
# Separate features and phenotype
# ------------------------------------------------------------

X_train = train.drop(
    columns=["genome_id", "cip_resistant"]
)

y_train = train["cip_resistant"]

X_test = test.drop(
    columns=["genome_id", "cip_resistant"]
)

y_test = test["cip_resistant"]

print("\nTraining isolates:", len(X_train))
print("Test isolates:", len(X_test))
print("Number of reduced features:", X_train.shape[1])

# ------------------------------------------------------------
# Train model
# ------------------------------------------------------------

print("\nTraining reduced Random Forest...")

model = RandomForestClassifier(
    n_estimators=500,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

# ------------------------------------------------------------
# Predictions
# ------------------------------------------------------------

y_pred = model.predict(X_test)

y_probability = model.predict_proba(
    X_test
)[:, 1]

# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

balanced_accuracy = balanced_accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

cm = confusion_matrix(
    y_test,
    y_pred
)

tn, fp, fn, tp = cm.ravel()

specificity = tn / (tn + fp)

# ------------------------------------------------------------
# Print results
# ------------------------------------------------------------

print("\n============================================================")
print("REDUCED RANDOM FOREST TEST RESULTS")
print("============================================================")

print(f"Accuracy:            {accuracy:.4f}")
print(f"Balanced accuracy:   {balanced_accuracy:.4f}")
print(f"Precision:            {precision:.4f}")
print(f"Recall:               {recall:.4f}")
print(f"Specificity:          {specificity:.4f}")
print(f"F1 score:             {f1:.4f}")
print(f"ROC-AUC:              {roc_auc:.4f}")

print("\nConfusion matrix:")
print(cm)

# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------

results = pd.DataFrame(
    {
        "model": [
            "Reduced Random Forest"
        ],
        "accuracy": [
            accuracy
        ],
        "balanced_accuracy": [
            balanced_accuracy
        ],
        "precision": [
            precision
        ],
        "recall": [
            recall
        ],
        "specificity": [
            specificity
        ],
        "f1": [
            f1
        ],
        "roc_auc": [
            roc_auc
        ],
        "true_negative": [
            tn
        ],
        "false_positive": [
            fp
        ],
        "false_negative": [
            fn
        ],
        "true_positive": [
            tp
        ],
        "number_of_features": [
            X_train.shape[1]
        ]
    }
)

results.to_csv(
    output_file,
    index=False
)

print("\nResults saved to:")
print(output_file)