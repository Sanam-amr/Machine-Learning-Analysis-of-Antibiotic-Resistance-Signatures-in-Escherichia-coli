import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import joblib

print("==================================================")
print("STAGE 4: BIOLOGICAL VALIDATION & EVALUATION")
print("==================================================")

# 1. Load the saved model and test datasets safely
models_dir = "../models"
model_path = f"{models_dir}/amr_random_forest_model.joblib"

if not os.path.exists(model_path):
    raise FileNotFoundError("Trained model binary not found! Please run Stage 3 first.")

model = joblib.load(model_path)
X_test = pd.read_csv(f"{models_dir}/X_test.csv")
y_test = pd.read_csv(f"{models_dir}/y_test.csv").values.ravel() # Flatten to 1D array

print("Trained Random Forest binary and validation validation cohorts successfully imported.")

# 2. Compute Predictions & Deep Classification Report
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n--- Advanced Performance Metrics ---")
print(classification_report(y_test, y_pred, target_names=["Susceptible", "Resistant"]))
print("-" * 50)

# 3. Extract and Plot Feature Importance (The Biological Drivers)
print("Extracting genomic feature importance ranks...")
importances = model.feature_importances_
feature_names = X_test.columns

# Sort features by importance score
indices = np.argsort(importances)[::-1]
sorted_features = feature_names[indices]
sorted_importances = importances[indices]

# Save figures directory setup
os.makedirs("../figures", exist_ok=True)

# Generate Figure 1: Feature Importance
plt.figure(figsize=(12, 8))
sns.barplot(x=sorted_importances, y=sorted_features, palette="viridis", hue=sorted_features, legend=False)
plt.title("Genomic Features Driving AMR Phenotype Prediction (Feature Importance)", fontsize=12, fontweight="bold", pad=15)
plt.xlabel("Relative Importance Score", fontsize=10)
plt.ylabel("AMR Genetic Determinant / Mutation", fontsize=10)
plt.tight_layout()

fi_output = "../figures/amr_feature_importance.png"
plt.savefig(fi_output, dpi=300)
plt.close()
print(f"  * Figure 1 saved successfully: {fi_output}")

# Generate Figure 2: Receiver Operating Characteristic (ROC) Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 7))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Guess Baseline")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
plt.ylabel("True Positive Rate (Sensitivity)", fontsize=10)
plt.title("Receiver Operating Characteristic (ROC) Diagnostic Performance", fontsize=12, fontweight="bold", pad=15)
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()

roc_output = "../figures/amr_roc_curve.png"
plt.savefig(roc_output, dpi=300)
plt.close()
print(f"  * Figure 2 saved successfully: {roc_output}")
print("\nEvaluation Stage Complete! Your plots are ready for deployment.")