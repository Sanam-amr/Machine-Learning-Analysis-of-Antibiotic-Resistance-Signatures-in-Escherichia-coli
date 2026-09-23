import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# STEP 20
# PLOT KEY FEATURE PREVALENCE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

input_file = (
    PROJECT_ROOT
    / "results"
    / "key_feature_summary.csv"
)

output_file = (
    PROJECT_ROOT
    / "figures"
    / "key_feature_prevalence.png"
)

print("Loading key feature summary...")

df = pd.read_csv(input_file)

# ------------------------------------------------------------
# Sort by resistant prevalence
# ------------------------------------------------------------

df = df.sort_values(
    "prevalence_resistant_percent",
    ascending=True
)

# ------------------------------------------------------------
# Create figure
# ------------------------------------------------------------

plt.figure(figsize=(10, 8))

plt.barh(
    df["feature"],
    df["prevalence_resistant_percent"],
    label="Ciprofloxacin resistant"
)

plt.barh(
    df["feature"],
    df["prevalence_susceptible_percent"],
    alpha=0.6,
    label="Ciprofloxacin susceptible"
)

plt.xlabel("Feature prevalence (%)")
plt.ylabel("Genomic feature")
plt.title(
    "Prevalence of Key Genomic Features by Ciprofloxacin Phenotype"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nFigure saved to:")
print(output_file)