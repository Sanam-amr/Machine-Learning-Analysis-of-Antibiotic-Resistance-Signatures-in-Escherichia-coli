from pathlib import Path
import pandas as pd

# Project folders
PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"
ANNOTS = DATA / "annots"

# Read the CIP phenotype table
target = pd.read_csv(DATA / "cip_target.csv")

# Keep only the isolate IDs and phenotype
target = target[["genome_id", "cip_resistant"]].copy()

feature_rows = []

# Read every annotation file
for genome_id in target["genome_id"]:
    file_path = ANNOTS / f"{genome_id}.tsv"

    if not file_path.exists():
        print(f"Warning: {file_path.name} not found")
        feature_rows.append({"genome_id": genome_id})
        continue

    df = pd.read_csv(file_path, sep="\t")

    # We use Gene symbol as the genomic feature.
    # One isolate can contain the same feature more than once,
    # so duplicates are removed.
    if "Gene symbol" not in df.columns:
        print(f"Warning: 'Gene symbol' column missing in {file_path.name}")
        feature_rows.append({"genome_id": genome_id})
        continue

    genes = (
        df["Gene symbol"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    genes = genes[genes.ne("")].unique()

    row = {"genome_id": genome_id}

    # Binary genomic feature: 1 = feature present, 0 = absent
    for gene in genes:
        row[gene] = 1

    feature_rows.append(row)

# Convert to a binary feature matrix
features = pd.DataFrame(feature_rows).fillna(0)

# Make sure all feature columns are numeric 0/1
for col in features.columns:
    if col != "genome_id":
        features[col] = pd.to_numeric(features[col], errors="coerce").fillna(0)
        features[col] = (features[col] > 0).astype(int)

# Add the CIP phenotype
dataset = target.merge(features, on="genome_id", how="left")

# Save result
output = DATA / "cip_genomic_features.csv"
dataset.to_csv(output, index=False)

print("\nDONE")
print(f"Output: {output}")
print(f"Rows (isolates): {dataset.shape[0]}")
print(f"Columns (including genome_id + phenotype): {dataset.shape[1]}")
print(f"Genomic features: {dataset.shape[1] - 2}")
print("\nFirst 5 rows:")
print(dataset.head())
