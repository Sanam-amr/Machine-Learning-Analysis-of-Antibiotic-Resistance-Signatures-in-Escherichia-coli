import os
import numpy as np
import pandas as pd

print("==================================================")
print("REALISTIC E. COLI AMR FEATURE MATRIX GENERATOR")
print("==================================================")

# Ensure data directory exists
os.makedirs("../data", exist_ok=True)
output_path = "../data/raw_ecoli_amr_data.csv"

# Setting parameters for a publication-grade simulation dataset
np.random.seed(42)
n_isolates = 1200

# 1. Generate realistic E. coli strain IDs (Isolates)
isolate_ids = [f"EC_clinical_isolate_{i+1:04d}" for i in range(n_isolates)]

# 2. Define real-world E. coli Antimicrobial Resistance (AMR) Genetic Features
genomic_features = [
    "blaCTX-M-15", "blaCTX-M-14", "blaTEM-1", "blaSHV-12",  # Beta-lactamases (ESBL)
    "gyrA_S83L", "gyrA_D87N", "parC_S80I", "parC_E84K",      # Fluoroquinolone resistance target mutations
    "aac(6')-Ib-cr", "aph(3'')-Ib", "aph(6)-Id",             # Aminoglycoside modifying enzymes
    "sul1", "sul2", "dfrA17", "dfrA14",                      # Sulfonamide & Trimethoprim resistance
    "tet(A)", "tet(B)",                                      # Tetracycline efflux pumps
    "mph(A)", "erm(B)",                                      # Macrolide resistance
    "clpA", "ftsI_mutation"                                  # Cell-wall / stress response modifications
]

# 3. Create a realistic binary variant matrix (presence=1, absence=0)
# We add mathematical dependencies so the Machine Learning model can actually learn real signatures!
X_data = {}
for gene in genomic_features:
    # Some genes are highly prevalent in clinical setups, others are rarer
    base_probability = np.random.uniform(0.15, 0.65)
    X_data[gene] = np.random.choice([0, 1], size=n_isolates, p=[1 - base_probability, base_probability])

df = pd.DataFrame(X_data)

# 4. Synthesize a Phenotypic Target Variable (Resistance to Ciprofloxacin/Beta-lactams)
# Strains with combinations of gyrA/parC mutations or ESBL genes are highly likely to be Phenotypically Resistant (1)
score = (
    df["gyrA_S83L"] * 2.5 + 
    df["parC_S80I"] * 2.0 + 
    df["blaCTX-M-15"] * 1.8 + 
    df["tet(A)"] * 0.5 + 
    np.random.normal(0, 1.2, size=n_isolates) # Adding realistic biological noise
)

# Convert continuous score into a binary Phenotype: 1 = Resistant, 0 = Susceptible
df["AMR_Phenotype"] = (score > score.median()).astype(int)

# Insert the Isolate IDs at the very front
df.insert(0, "Isolate_ID", isolate_ids)

# Save to your data directory
df.to_csv(output_path, index=False)

print(f"Success! High-dimensional dataset created locally.")
print(f"Saved to: {output_path}")
print(f"\nDataset Audit:")
print(f"- Total E. coli Strains Analyzed: {df.shape[0]}")
print(f"- Total Antimicrobial Feature Signatures: {df.shape[1] - 2}") # Subtracting ID and Target
print(f"- Target Variable distribution (0=Susceptible, 1=Resistant):\n{df['AMR_Phenotype'].value_counts()}")