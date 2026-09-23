import pandas as pd
from pathlib import Path

# ============================================================
# STEP 22
# ANALYZE FEATURE CO-OCCURRENCE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

input_file = (
    PROJECT_ROOT
    / "data"
    / "ml_train.csv"
)

output_file = (
    PROJECT_ROOT
    / "results"
    / "key_feature_correlation.csv"
)

print("Loading training dataset...")

df = pd.read_csv(input_file)

# ------------------------------------------------------------
# Key features from Step 19
# ------------------------------------------------------------

key_features = [
    "parC_S80I",
    "gyrA_D87N",
    "gyrA_S83L",
    "parC_E84V",
    "ptsI_V25I",
    "parE_I529L",
    "uhpT_E350Q",
    "parE_S458A",
    "mph(A)",
    "tet(A)",
    "blaCTX-M-15",
    "catB3",
    "aac(6')-Ib-cr5",
    "qacEdelta1",
    "aac(3)-IId",
    "blaOXA-1",
    "sul1",
    "aadA5",
    "dfrA17"
]

features = df[key_features]

# ------------------------------------------------------------
# Pearson correlation for binary features
#
# For binary 0/1 variables this is equivalent to the
# phi coefficient.
# ------------------------------------------------------------

correlation = features.corr(method="pearson")

correlation.to_csv(output_file)

print("\n============================================================")
print("KEY FEATURE CORRELATION")
print("============================================================")

print(correlation.round(3).to_string())

print("\nCorrelation matrix saved to:")
print(output_file)