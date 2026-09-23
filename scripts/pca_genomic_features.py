import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# --------------------------------------------------
# Project paths
# --------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

DATA_DIR = PROJECT_DIR / "data"
FIGURES_DIR = PROJECT_DIR / "figures"

TRAIN_FILE = DATA_DIR / "ml_train.csv"

OUTPUT_FILE = FIGURES_DIR / "pca_genomic_features.png"

FIGURES_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# Load training data
# --------------------------------------------------

print("Loading training dataset...")

train = pd.read_csv(
    TRAIN_FILE
)


# --------------------------------------------------
# Extract genomic features
# --------------------------------------------------

feature_columns = [
    column
    for column in train.columns
    if column not in [
        "genome_id",
        "cip_resistant"
    ]
]

X = train[feature_columns]

y = train["cip_resistant"]


print("Number of isolates:", len(train))
print("Number of genomic features:", len(feature_columns))


# --------------------------------------------------
# Standardize features
# --------------------------------------------------

print("\nStandardizing genomic features...")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# --------------------------------------------------
# PCA
# --------------------------------------------------

print("Running PCA...")

pca = PCA(
    n_components=2,
    random_state=42
)

X_pca = pca.fit_transform(
    X_scaled
)


pc1_variance = pca.explained_variance_ratio_[0] * 100
pc2_variance = pca.explained_variance_ratio_[1] * 100


print("\nExplained variance:")
print(
    f"PC1: {pc1_variance:.2f}%"
)

print(
    f"PC2: {pc2_variance:.2f}%"
)

print(
    f"PC1 + PC2: "
    f"{pc1_variance + pc2_variance:.2f}%"
)


# --------------------------------------------------
# Plot
# --------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[y == 0, 0],
    X_pca[y == 0, 1],
    label="Susceptible",
    alpha=0.7
)

plt.scatter(
    X_pca[y == 1, 0],
    X_pca[y == 1, 1],
    label="Resistant",
    alpha=0.7
)

plt.xlabel(
    f"PC1 ({pc1_variance:.2f}% variance)"
)

plt.ylabel(
    f"PC2 ({pc2_variance:.2f}% variance)"
)

plt.title(
    "PCA of E. coli Genomic Features"
)

plt.legend()

plt.tight_layout()


# --------------------------------------------------
# Save
# --------------------------------------------------

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("\nPCA figure saved to:")
print(OUTPUT_FILE)