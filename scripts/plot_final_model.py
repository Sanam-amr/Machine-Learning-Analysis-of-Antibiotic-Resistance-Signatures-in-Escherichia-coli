import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc
)


# --------------------------------------------------
# Project paths
# --------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"

PREDICTIONS_FILE = RESULTS_DIR / "random_forest_test_predictions.csv"


# Create figures directory if needed
FIGURES_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# Load predictions
# --------------------------------------------------

predictions = pd.read_csv(PREDICTIONS_FILE)

y_true = predictions["actual_cip_resistant"]
y_pred = predictions["predicted_cip_resistant"]
y_prob = predictions["resistance_probability"]


# --------------------------------------------------
# Confusion matrix
# --------------------------------------------------

cm = confusion_matrix(
    y_true,
    y_pred
)

tn, fp, fn, tp = cm.ravel()


plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted phenotype")
plt.ylabel("Actual phenotype")

plt.xticks(
    [0, 1],
    ["Susceptible", "Resistant"]
)

plt.yticks(
    [0, 1],
    ["Susceptible", "Resistant"]
)

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )

plt.tight_layout()

confusion_file = FIGURES_DIR / "random_forest_confusion_matrix.png"

plt.savefig(
    confusion_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# ROC curve
# --------------------------------------------------

fpr, tpr, thresholds = roc_curve(
    y_true,
    y_prob
)

roc_auc = auc(
    fpr,
    tpr
)


plt.figure(figsize=(6, 5))

plt.plot(
    fpr,
    tpr,
    label=f"Random Forest (AUC = {roc_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False positive rate")
plt.ylabel("True positive rate")

plt.title(
    "Random Forest ROC Curve — Independent Test Set"
)

plt.legend(
    loc="lower right"
)

plt.tight_layout()

roc_file = FIGURES_DIR / "random_forest_test_roc_curve.png"

plt.savefig(
    roc_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# Display
# --------------------------------------------------

print("\n" + "=" * 60)
print("FINAL MODEL FIGURES CREATED")
print("=" * 60)

print("\nConfusion matrix:")
print(cm)

print("\nROC-AUC:", round(roc_auc, 4))

print("\nFigures saved to:")

print(confusion_file)
print(roc_file)