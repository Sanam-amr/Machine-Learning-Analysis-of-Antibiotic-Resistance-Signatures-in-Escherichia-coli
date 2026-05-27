import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

print("==================================================")
print("STAGE 3: MACHINE LEARNING MODEL TRAINING STACK")
print("==================================================")

# 1. Load the dataset generated in Stage 1
data_path = "../data/raw_ecoli_amr_data.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError("Missing dataset! Please run Stage 1 and Stage 2 scripts first.")

df = pd.read_csv(data_path)
print(f"Successfully imported variant matrix for {df.shape[0]} E. coli strains.")

# 2. Segregate Features (Genotypes) and Target Variable (Phenotype)
# Drop metadata 'Isolate_ID' and the target column 'AMR_Phenotype' to isolate features
X = df.drop(columns=["Isolate_ID", "AMR_Phenotype"])
y = df["AMR_Phenotype"]

# 3. Create Reproducible Cohorts (Train/Test Split)
# 80% used to train the machine learning rules, 20% held back to test validity
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"  * Training Set Size (Genotypes): {X_train.shape[0]} isolates")
print(f"  * Validation Set Size (Genotypes): {X_test.shape[0]} isolates")
print(f"  * Number of Variant Features Input: {X_train.shape[1]}")
print("-" * 50)

# 4. Initialize and Configure the Random Forest Ensemble
print("Initializing Random Forest Ensemble Classifier...")
# Using 150 estimators (trees) to balance accuracy and avoid over-fitting
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    random_state=42,
    n_jobs=-1  # Uses all available CPU cores on your laptop for faster computing
)

# 5. Train the Model (Fitting the Genotype-to-Phenotype landscape)
print("Training classifier to map resistance signatures... Please wait.")
model.fit(X_train, y_train)
print("Model training achieved successfully!")

# 6. Generate Baseline In-Sample Evaluation
train_accuracy = model.score(X_train, y_train) * 100
test_accuracy = model.score(X_test, y_test) * 100
print(f"\n--- Preliminary Prediction Metrics ---")
print(f"  * Baseline Training Accuracy: {train_accuracy:.2f}%")
print(f"  * Preliminary Testing Accuracy: {test_accuracy:.2f}%")
print("-" * 50)

# 7. Serialize the Assets (Saving the Model for Client/Web Deployment)
models_dir = "../models"
os.makedirs(models_dir, exist_ok=True)

# Save the trained model binary and the split test data for final evaluation
joblib.dump(model, f"{models_dir}/amr_random_forest_model.joblib")
X_test.to_csv(f"{models_dir}/X_test.csv", index=False)
y_test.to_csv(f"{models_dir}/y_test.csv", index=False)

print(f"Success! Model binary saved to: {models_dir}/amr_random_forest_model.joblib")
print("Stage 3 complete. Ready for advanced statistical validation.")