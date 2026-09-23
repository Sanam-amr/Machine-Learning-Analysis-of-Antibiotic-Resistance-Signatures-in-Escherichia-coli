import pandas as pd
from pathlib import Path


# ---------------------------------------------------------
# STEP 7: Verify train/test datasets
# ---------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

DATA_DIR = PROJECT_DIR / "data"

TRAIN_FILE = DATA_DIR / "ml_train.csv"
TEST_FILE = DATA_DIR / "ml_test.csv"


# ---------------------------------------------------------
# 1. Load datasets
# ---------------------------------------------------------

train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)

print("Training dataset shape:", train.shape)
print("Test dataset shape:", test.shape)


# ---------------------------------------------------------
# 2. Check genome IDs
# ---------------------------------------------------------

print("\nChecking genome IDs...")

train_ids = set(train["genome_id"])
test_ids = set(test["genome_id"])

overlap = train_ids.intersection(test_ids)

print("Duplicate IDs within training:", train["genome_id"].duplicated().sum())
print("Duplicate IDs within test:", test["genome_id"].duplicated().sum())
print("IDs shared between train and test:", len(overlap))

if len(overlap) > 0:
    raise ValueError("ERROR: Some genome IDs occur in both train and test.")


# ---------------------------------------------------------
# 3. Identify genomic features
# ---------------------------------------------------------

train_features = set(
    train.columns
) - {"genome_id", "cip_resistant"}

test_features = set(
    test.columns
) - {"genome_id", "cip_resistant"}

print("\nNumber of training features:", len(train_features))
print("Number of test features:", len(test_features))

if train_features != test_features:
    raise ValueError("ERROR: Train and test feature columns do not match.")


# ---------------------------------------------------------
# 4. Check missing values
# ---------------------------------------------------------

train_missing = train.isna().sum().sum()
test_missing = test.isna().sum().sum()

print("\nMissing values:")
print("Training:", train_missing)
print("Test:", test_missing)

if train_missing > 0 or test_missing > 0:
    raise ValueError("ERROR: Missing values detected.")


# ---------------------------------------------------------
# 5. Check phenotype
# ---------------------------------------------------------

train_labels = sorted(train["cip_resistant"].unique())
test_labels = sorted(test["cip_resistant"].unique())

print("\nTraining phenotype classes:", train_labels)
print("Test phenotype classes:", test_labels)

if train_labels != [0, 1] or test_labels != [0, 1]:
    raise ValueError("ERROR: Phenotype classes are not exactly 0 and 1.")


# ---------------------------------------------------------
# 6. Final message
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("ML TRAIN/TEST SPLIT PASSED ALL CHECKS")
print("=" * 60)
print("\nThe datasets are ready for model development.")