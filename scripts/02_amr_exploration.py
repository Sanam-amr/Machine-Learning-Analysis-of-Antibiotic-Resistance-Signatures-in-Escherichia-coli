import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("==================================================")
print("STAGE 2: ADVANCED GENOMIC FEATURE EXPLORATION")
print("==================================================")

# 1. Load the dataset safely
data_path = "../data/raw_ecoli_amr_data.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Missing dataset! Please run 01_download_amr_data.py first.")

df = pd.read_csv(data_path)
print(f"Successfully loaded dataset: {df.shape[0]} isolates across {df.shape[1]} metrics.\n")

# 2. Audit Class Balance of the Target Phenotype
print("--- Target Variable Distribution Audit ---")
phenotype_counts = df["AMR_Phenotype"].value_counts()
phenotype_pct = df["AMR_Phenotype"].value_counts(normalize=True) * 100
for val in phenotype_counts.index:
    label = "Resistant" if val == 1 else "Susceptible"
    print(f"  * {label}: {phenotype_counts[val]} strains ({phenotype_pct[val]:.2f}%)")
print("-" * 42)

# 3. Calculate Epidemiology Epidemiology: Top 5 Most Prevalent AMR Genes
print("\n--- Top 5 Most Prevalent Genomic Determinants ---")
# Drop metadata columns to isolate only gene columns
gene_df = df.drop(columns=["Isolate_ID", "AMR_Phenotype"])
gene_frequencies = gene_df.mean().sort_values(ascending=False) * 100

for i in range(5):
    print(f"  {i+1}. {gene_frequencies.index[i]}: present in {gene_frequencies.iloc[i]:.2f}% of isolates")
print("-" * 42)

# 4. Compute Genomic Co-occurrence / Linkage Disequilibrium Matrix
print("\nComputing pairwise genomic feature co-occurrence matrix...")
co_occurrence_matrix = gene_df.corr(method="pearson")

# 5. Generate Publication-Quality Figure (Nature-Style Formatting)
os.makedirs("../figures", exist_ok=True)

plt.figure(figsize=(14, 11))
# Setting up a professional, diverging color palette for genomic linkage
sns.heatmap(
    co_occurrence_matrix,
    cmap="coolwarm",
    vmin=-1.0,
    vmax=1.0,
    center=0,
    linewidths=0.2,
    cbar_kws={"label": "Linkage Correlation Coefficient (r)"}
)

plt.title("Escherichia coli Genomic AMR Feature Co-occurrence Landscape", fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()

# Save image assets for GitHub markdown rendering
figure_output = "../figures/genomic_cooccurrence_heatmap.png"
plt.savefig(figure_output, dpi=300)
plt.close()

print(f"Success! High-resolution exploratory figure exported to: {figure_output}")
print("Exploration Stage Complete.")