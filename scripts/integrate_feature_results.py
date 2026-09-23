import pandas as pd
from pathlib import Path


# --------------------------------------------------
# Project paths
# --------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

RESULTS_DIR = PROJECT_DIR / "results"

ASSOCIATION_FILE = RESULTS_DIR / "cip_feature_associations.csv"
IMPORTANCE_FILE = RESULTS_DIR / "random_forest_feature_importance.csv"

OUTPUT_FILE = RESULTS_DIR / "integrated_feature_analysis.csv"


# --------------------------------------------------
# Load results
# --------------------------------------------------

print("Loading statistical association results...")

association = pd.read_csv(
    ASSOCIATION_FILE
)

print("Loading Random Forest feature importance...")

importance = pd.read_csv(
    IMPORTANCE_FILE
)


# --------------------------------------------------
# Select useful association columns
# --------------------------------------------------

association_selected = association[
    [
        "feature",
        "n_present_total",
        "prevalence_total_percent",
        "prevalence_resistant_percent",
        "prevalence_susceptible_percent",
        "odds_ratio",
        "p_value",
        "fdr_p_value",
        "significant_fdr_0.05"
    ]
].copy()


# --------------------------------------------------
# Merge association and ML importance
# --------------------------------------------------

integrated = pd.merge(
    importance,
    association_selected,
    on="feature",
    how="left"
)


# --------------------------------------------------
# Rank features by Random Forest importance
# --------------------------------------------------

integrated["rf_rank"] = (
    integrated["importance"]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)


# --------------------------------------------------
# Rank features by statistical significance
# --------------------------------------------------

integrated["fdr_rank"] = (
    integrated["fdr_p_value"]
    .rank(
        ascending=True,
        method="min"
    )
    .astype(int)
)


# --------------------------------------------------
# Create interpretation category
# --------------------------------------------------

def classify_feature(row):

    rf_high = row["rf_rank"] <= 20
    fdr_significant = row["significant_fdr_0.05"] == True

    if rf_high and fdr_significant:
        return "High ML importance + significant association"

    elif rf_high and not fdr_significant:
        return "High ML importance + not FDR significant"

    elif not rf_high and fdr_significant:
        return "FDR significant + lower ML importance"

    else:
        return "Lower ML importance + not FDR significant"


integrated["evidence_category"] = integrated.apply(
    classify_feature,
    axis=1
)


# --------------------------------------------------
# Sort by Random Forest importance
# --------------------------------------------------

integrated = integrated.sort_values(
    "importance",
    ascending=False
).reset_index(drop=True)


# --------------------------------------------------
# Save
# --------------------------------------------------

integrated.to_csv(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# Display
# --------------------------------------------------

print("\n" + "=" * 90)
print("INTEGRATED STATISTICAL + MACHINE-LEARNING FEATURE ANALYSIS")
print("=" * 90)

display_columns = [
    "rf_rank",
    "feature",
    "importance",
    "odds_ratio",
    "fdr_p_value",
    "prevalence_resistant_percent",
    "prevalence_susceptible_percent",
    "evidence_category"
]

print(
    integrated[
        display_columns
    ].head(30).to_string(index=False)
)


print("\n" + "=" * 90)
print("EVIDENCE CATEGORY COUNTS")
print("=" * 90)

print(
    integrated["evidence_category"]
    .value_counts()
    .to_string()
)


print("\nIntegrated results saved to:")
print(OUTPUT_FILE)