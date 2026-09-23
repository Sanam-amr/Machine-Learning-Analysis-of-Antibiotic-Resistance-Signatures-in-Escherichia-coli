import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier


# --------------------------------------------------
# Project paths
# --------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"

TRAIN_FILE = DATA_DIR / "ml_train.csv"

OUTPUT_FILE = RESULTS_DIR / "random_forest_feature_importance.csv"

FIGURES_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# Load training data
# --------------------------------------------------

print("Loading training dataset...")

train = pd.read_csv(TRAIN_FILE)

feature_columns = [
    column
    for column in train.columns
    if column not in ["genome_id", "cip_resistant"]
]

X_train = train[feature_columns]
y_train = train["cip_resistant"]


# --------------------------------------------------
# Train Random Forest
# --------------------------------------------------

print("Training Random Forest...")

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


# --------------------------------------------------
# Extract feature importance
# --------------------------------------------------

importance_df = pd.DataFrame({
    "feature": feature_columns,
    "importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    "importance",
    ascending=False
).reset_index(drop=True)


# --------------------------------------------------
# Save all feature importances
# --------------------------------------------------

importance_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# Display top 20
# --------------------------------------------------

print("\n" + "=" * 60)
print("TOP 20 RANDOM FOREST FEATURES")
print("=" * 60)

print(
    importance_df.head(20).to_string(index=False)
)


# --------------------------------------------------
# Plot top 20 features
# --------------------------------------------------

top20 = importance_df.head(20).sort_values(
    "importance",
    ascending=True
)

plt.figure(figsize=(8, 8))

plt.barh(
    top20["feature"],
    top20["importance"]
)

plt.xlabel("Random Forest feature importance")
plt.ylabel("Genomic feature")

plt.title(
    "Top 20 Genomic Features — Random Forest"
)

plt.tight_layout()

figure_file = FIGURES_DIR / "random_forest_top20_features.png"

plt.savefig(
    figure_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# Display file locations
# --------------------------------------------------

print("\nFeature importance table saved to:")
print(OUTPUT_FILE)

print("\nFeature importance figure saved to:")
print(figure_file)