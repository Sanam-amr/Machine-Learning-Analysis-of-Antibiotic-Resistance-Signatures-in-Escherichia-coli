import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


# --------------------------------------------------
# Project paths
# --------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"

INPUT_FILE = RESULTS_DIR / "integrated_feature_analysis.csv"

OUTPUT_FILE = FIGURES_DIR / "integrated_feature_importance.png"

FIGURES_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# Load integrated results
# --------------------------------------------------

data = pd.read_csv(
    INPUT_FILE
)


# --------------------------------------------------
# Select top 20 ML features
# --------------------------------------------------

top20 = (
    data
    .head(20)
    .sort_values(
        "importance",
        ascending=True
    )
)


# --------------------------------------------------
# Plot
# --------------------------------------------------

plt.figure(figsize=(9, 8))

plt.barh(
    top20["feature"],
    top20["importance"]
)

plt.xlabel(
    "Random Forest feature importance"
)

plt.ylabel(
    "Genomic feature"
)

plt.title(
    "Top 20 Genomic Features and Statistical Evidence"
)

plt.tight_layout()


# --------------------------------------------------
# Save figure
# --------------------------------------------------

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("\nIntegrated feature figure saved to:")
print(OUTPUT_FILE)