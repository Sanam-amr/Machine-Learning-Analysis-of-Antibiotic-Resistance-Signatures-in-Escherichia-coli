import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------
# STEP 6: Prepare leakage-safe train/test datasets
# ---------------------------------------------------------

# Project folders
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"

INPUT_FILE = DATA_DIR / "cip_genomic_features.csv"

TRAIN_FILE = DATA_DIR / "ml_train.csv"
TEST_FILE = DATA_DIR / "ml_test.csv"
SPLIT_SUMMARY_FILE = RESULTS_DIR / "ml_split_summary.csv"


# ---------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------

print("Loading genomic feature dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Dataset shape: {df.shape}")


# ---------------------------------------------------------
# 2. Check required columns
# ---------------------------------------------------------

required_columns = ["genome_id", "cip_resistant"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Required column missing: {column}")


# ---------------------------------------------------------
# 3. Check genome IDs
# ---------------------------------------------------------

if df["genome_id"].duplicated().any():
    duplicated_ids = df.loc[
        df["genome_id"].duplicated(), "genome_id"
    ].tolist()

    raise ValueError(
        f"Duplicate genome IDs found. Examples: {duplicated_ids[:10]}"
    )

print("Genome IDs are unique.")


# ---------------------------------------------------------
# 4. Check phenotype
# ---------------------------------------------------------

if df["cip_resistant"].isna().any():
    print("Missing phenotype values found. Removing those rows.")
    df = df.dropna(subset=["cip_resistant"])


# Convert phenotype to integer
df["cip_resistant"] = df["cip_resistant"].astype(int)

unique_labels = sorted(df["cip_resistant"].unique())

print(f"Phenotype classes found: {unique_labels}")

if unique_labels != [0, 1]:
    raise ValueError(
        "cip_resistant must contain exactly two classes: 0 and 1."
    )


# ---------------------------------------------------------
# 5. Separate predictors and target
# ---------------------------------------------------------

feature_columns = [
    column
    for column in df.columns
    if column not in ["genome_id", "cip_resistant"]
]

X = df[feature_columns]
y = df["cip_resistant"]

print(f"Number of genomic features: {len(feature_columns)}")
print(f"Number of isolates: {len(df)}")


# ---------------------------------------------------------
# 6. Train/test split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ---------------------------------------------------------
# 7. Recover genome IDs
# ---------------------------------------------------------

train_ids = df.loc[X_train.index, "genome_id"]
test_ids = df.loc[X_test.index, "genome_id"]


# ---------------------------------------------------------
# 8. Build train and test tables
# ---------------------------------------------------------

train_df = X_train.copy()
train_df.insert(0, "genome_id", train_ids)
train_df["cip_resistant"] = y_train

test_df = X_test.copy()
test_df.insert(0, "genome_id", test_ids)
test_df["cip_resistant"] = y_test


# ---------------------------------------------------------
# 9. Save datasets
# ---------------------------------------------------------

train_df.to_csv(TRAIN_FILE, index=False)
test_df.to_csv(TEST_FILE, index=False)


# ---------------------------------------------------------
# 10. Create split summary
# ---------------------------------------------------------

summary = pd.DataFrame({
    "dataset": ["full", "train", "test"],
    "n_isolates": [
        len(df),
        len(train_df),
        len(test_df)
    ],
    "n_resistant": [
        int(y.sum()),
        int(y_train.sum()),
        int(y_test.sum())
    ],
    "n_susceptible": [
        int((y == 0).sum()),
        int((y_train == 0).sum()),
        int((y_test == 0).sum())
    ]
})

summary["resistant_percent"] = (
    summary["n_resistant"] /
    summary["n_isolates"] * 100
)

summary["susceptible_percent"] = (
    summary["n_susceptible"] /
    summary["n_isolates"] * 100
)

summary.to_csv(SPLIT_SUMMARY_FILE, index=False)


# ---------------------------------------------------------
# 11. Print results
# ---------------------------------------------------------

print("\nTrain/test split completed.")

print("\nSplit summary:")
print(summary.to_string(index=False))

print("\nFiles created:")
print(TRAIN_FILE)
print(TEST_FILE)
print(SPLIT_SUMMARY_FILE)

print("\nIMPORTANT:")
print("The test dataset will remain untouched during model development.")