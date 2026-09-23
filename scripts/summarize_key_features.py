import pandas as pd
from pathlib import Path

# ============================================================
# STEP 19
# SUMMARIZE KEY FEATURES
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

input_file = (
    PROJECT_ROOT
    / "results"
    / "integrated_feature_analysis.csv"
)

output_file = (
    PROJECT_ROOT
    / "results"
    / "key_feature_summary.csv"
)

print("Loading integrated feature results...")

df = pd.read_csv(input_file)

print("\n============================================================")
print("KEY FEATURE SUMMARY")
print("============================================================")

# ------------------------------------------------------------
# Key features identified from the integrated analysis
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

# ------------------------------------------------------------
# Extract key features
# ------------------------------------------------------------

summary = df[
    df["feature"].isin(key_features)
].copy()

# ------------------------------------------------------------
# Keep useful columns
# ------------------------------------------------------------

summary = summary[
    [
        "rf_rank",
        "feature",
        "importance",
        "odds_ratio",
        "fdr_p_value",
        "prevalence_resistant_percent",
        "prevalence_susceptible_percent",
        "evidence_category"
    ]
]

# ------------------------------------------------------------
# Sort according to Random Forest rank
# ------------------------------------------------------------

summary = summary.sort_values(
    "rf_rank"
)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

summary.to_csv(
    output_file,
    index=False
)

# ------------------------------------------------------------
# Display
# ------------------------------------------------------------

print(
    summary.to_string(index=False)
)

print("\nNumber of key features:", len(summary))

print("\nKey feature summary saved to:")
print(output_file)