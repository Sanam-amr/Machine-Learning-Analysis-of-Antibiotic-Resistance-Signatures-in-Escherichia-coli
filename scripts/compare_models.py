import pandas as pd
from pathlib import Path


# --------------------------------------------------
# Project paths
# --------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

RESULTS_DIR = PROJECT_DIR / "results"

LOGISTIC_FILE = RESULTS_DIR / "logistic_cv_results.csv"
RF_FILE = RESULTS_DIR / "random_forest_cv_results.csv"

OUTPUT_FILE = RESULTS_DIR / "model_comparison.csv"


# --------------------------------------------------
# Load results
# --------------------------------------------------

print("Loading model results...")

logistic = pd.read_csv(LOGISTIC_FILE)
random_forest = pd.read_csv(RF_FILE)


# --------------------------------------------------
# Keep mean and standard deviation
# --------------------------------------------------

logistic_summary = logistic[["metric", "mean", "std"]].copy()
rf_summary = random_forest[["metric", "mean", "std"]].copy()


# --------------------------------------------------
# Rename columns
# --------------------------------------------------

logistic_summary = logistic_summary.rename(
    columns={
        "mean": "logistic_mean",
        "std": "logistic_std"
    }
)

rf_summary = rf_summary.rename(
    columns={
        "mean": "random_forest_mean",
        "std": "random_forest_std"
    }
)


# --------------------------------------------------
# Merge results
# --------------------------------------------------

comparison = pd.merge(
    logistic_summary,
    rf_summary,
    on="metric",
    how="inner"
)


# --------------------------------------------------
# Calculate difference
# --------------------------------------------------

comparison["difference_rf_minus_logistic"] = (
    comparison["random_forest_mean"]
    - comparison["logistic_mean"]
)


# --------------------------------------------------
# Save comparison
# --------------------------------------------------

comparison.to_csv(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# Display
# --------------------------------------------------

print("\n" + "=" * 75)
print("LOGISTIC REGRESSION VS RANDOM FOREST")
print("=" * 75)

print(
    comparison[
        [
            "metric",
            "logistic_mean",
            "random_forest_mean",
            "difference_rf_minus_logistic"
        ]
    ].to_string(index=False)
)

print("\n" + "=" * 75)
print("INTERPRETATION OF CROSS-VALIDATED RESULTS")
print("=" * 75)

for _, row in comparison.iterrows():

    metric = row["metric"]

    logistic_value = row["logistic_mean"]
    rf_value = row["random_forest_mean"]
    difference = row["difference_rf_minus_logistic"]

    print(
        f"{metric}: "
        f"Logistic={logistic_value:.4f}, "
        f"Random Forest={rf_value:.4f}, "
        f"Difference={difference:+.4f}"
    )

print("\nComparison saved to:")
print(OUTPUT_FILE)