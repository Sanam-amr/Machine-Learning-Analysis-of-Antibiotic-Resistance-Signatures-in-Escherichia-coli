from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"
RESULTS = PROJECT / "results"

RESULTS.mkdir(exist_ok=True)


# ============================================================
# INPUT FILE
# ============================================================

input_file = DATA / "cip_genomic_features.csv"

if not input_file.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{input_file}\n\n"
        "Make sure build_cip_genomic_features.py has been run first."
    )


# ============================================================
# READ DATA
# ============================================================

df = pd.read_csv(input_file)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = ["genome_id", "cip_resistant"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            f"Required column '{column}' is missing from the input file."
        )


# ============================================================
# CHECK PHENOTYPE
# ============================================================

# CIP phenotype should be binary:
# 1 = resistant
# 0 = susceptible

phenotype_values = set(df["cip_resistant"].dropna().unique())

invalid_values = phenotype_values - {0, 1}

if invalid_values:
    raise ValueError(
        "Unexpected values found in 'cip_resistant': "
        f"{sorted(invalid_values)}\n"
        "Expected only 0 and 1."
    )


# Remove rows with missing phenotype
df = df.dropna(subset=["cip_resistant"]).copy()

df["cip_resistant"] = df["cip_resistant"].astype(int)


# ============================================================
# BASIC DATASET INFORMATION
# ============================================================

n_isolates = len(df)

n_resistant = int((df["cip_resistant"] == 1).sum())
n_susceptible = int((df["cip_resistant"] == 0).sum())

print("\nCIP GENOMIC FEATURE ASSOCIATION ANALYSIS")
print("----------------------------------------")
print(f"Total isolates: {n_isolates}")
print(f"CIP-resistant isolates: {n_resistant}")
print(f"CIP-susceptible isolates: {n_susceptible}")


# ============================================================
# IDENTIFY GENOMIC FEATURES
# ============================================================

feature_columns = [
    col
    for col in df.columns
    if col not in ["genome_id", "cip_resistant"]
]

print(f"Total genomic features: {len(feature_columns)}")


# ============================================================
# ANALYZE EACH FEATURE
# ============================================================

results = []

for feature in feature_columns:

    # Convert feature to binary just in case
    feature_values = pd.to_numeric(
        df[feature],
        errors="coerce"
    ).fillna(0)

    feature_values = (feature_values > 0).astype(int)

    # --------------------------------------------------------
    # 2 x 2 contingency table
    #
    #                    CIP R     CIP S
    # Feature present      a         b
    # Feature absent       c         d
    # --------------------------------------------------------

    a = int(((feature_values == 1) & (df["cip_resistant"] == 1)).sum())
    b = int(((feature_values == 1) & (df["cip_resistant"] == 0)).sum())
    c = int(((feature_values == 0) & (df["cip_resistant"] == 1)).sum())
    d = int(((feature_values == 0) & (df["cip_resistant"] == 0)).sum())

    # --------------------------------------------------------
    # Prevalence
    # --------------------------------------------------------

    if n_resistant > 0:
        prevalence_resistant = (a / n_resistant) * 100
    else:
        prevalence_resistant = np.nan

    if n_susceptible > 0:
        prevalence_susceptible = (b / n_susceptible) * 100
    else:
        prevalence_susceptible = np.nan

    total_present = a + b

    prevalence_total = (total_present / n_isolates) * 100


    # --------------------------------------------------------
    # Fisher's exact test
    # --------------------------------------------------------

    contingency_table = [
        [a, b],
        [c, d]
    ]

    odds_ratio, p_value = fisher_exact(
        contingency_table,
        alternative="two-sided"
    )


    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    results.append({
        "feature": feature,

        "n_present_total": total_present,
        "prevalence_total_percent": round(prevalence_total, 2),

        "n_present_resistant": a,
        "n_present_susceptible": b,

        "prevalence_resistant_percent": round(
            prevalence_resistant, 2
        ),

        "prevalence_susceptible_percent": round(
            prevalence_susceptible, 2
        ),

        "odds_ratio": odds_ratio,
        "p_value": p_value,

        "n_absent_resistant": c,
        "n_absent_susceptible": d
    })


# ============================================================
# CREATE RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results)


# ============================================================
# MULTIPLE-TESTING CORRECTION
# ============================================================

# We are testing many genomic features simultaneously.
# Benjamini-Hochberg FDR correction controls the expected
# proportion of false discoveries among significant results.

if len(results_df) > 0:

    reject, adjusted_p, _, _ = multipletests(
        results_df["p_value"],
        method="fdr_bh"
    )

    results_df["fdr_p_value"] = adjusted_p
    results_df["significant_fdr_0.05"] = reject

else:

    results_df["fdr_p_value"] = []
    results_df["significant_fdr_0.05"] = []


# ============================================================
# CLEAN ODDS RATIO VALUES
# ============================================================

# Infinite odds ratios can occur when a feature is present
# only in resistant isolates or only in susceptible isolates.
# We keep them because they contain useful information.

results_df["odds_ratio"] = results_df["odds_ratio"].replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# SORT RESULTS
# ============================================================

# First sort by FDR-adjusted p-value.
# Among similarly significant features, smaller p-values
# appear first.

results_df = results_df.sort_values(
    by=["fdr_p_value", "p_value"],
    ascending=[True, True]
).reset_index(drop=True)


# ============================================================
# SAVE RESULTS
# ============================================================

output_file = RESULTS / "cip_feature_associations.csv"

results_df.to_csv(
    output_file,
    index=False
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\nPhenotype distribution")
print("----------------------")
print(f"Resistant:  {n_resistant}")
print(f"Susceptible: {n_susceptible}")

print("\nTop 20 features by FDR-adjusted p-value")
print("---------------------------------------")

display_columns = [
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

print(
    results_df[display_columns]
    .head(20)
    .to_string(index=False)
)


# ============================================================
# SIGNIFICANT FEATURES
# ============================================================

significant = results_df[
    results_df["significant_fdr_0.05"] == True
]

print("\nSignificant features after FDR correction")
print("------------------------------------------")
print(f"Number of significant features: {len(significant)}")

if len(significant) > 0:

    print("\nTop significant features:")

    print(
        significant[
            [
                "feature",
                "prevalence_resistant_percent",
                "prevalence_susceptible_percent",
                "odds_ratio",
                "fdr_p_value"
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

else:

    print("No features passed FDR < 0.05.")


# ============================================================
# FINISH
# ============================================================

print("\nDONE")
print("----------------------------------------")
print(f"Association results saved to:")
print(output_file)