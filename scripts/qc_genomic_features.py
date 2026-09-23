from pathlib import Path
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"
RESULTS = PROJECT / "results"
RESULTS.mkdir(exist_ok=True)

input_file = DATA / "cip_genomic_features.csv"
df = pd.read_csv(input_file)

feature_columns = [
    col for col in df.columns
    if col not in ["genome_id", "cip_resistant"]
]

features = df[feature_columns]

frequency = pd.DataFrame({
    "feature": feature_columns,
    "n_isolates_present": features.sum(axis=0).astype(int).values
})

frequency["prevalence_percent"] = (
    frequency["n_isolates_present"] / len(df) * 100
).round(2)

frequency["n_isolates_absent"] = (
    len(df) - frequency["n_isolates_present"]
)

frequency = frequency.sort_values(
    "n_isolates_present",
    ascending=False
).reset_index(drop=True)

output_file = RESULTS / "feature_frequency.csv"
frequency.to_csv(output_file, index=False)

print("\nGENOMIC FEATURE QUALITY CONTROL")
print("--------------------------------")
print(f"Isolates analyzed: {len(df)}")
print(f"Genomic features analyzed: {len(feature_columns)}")

print("\nTop 20 features:")
print(frequency.head(20).to_string(index=False))

print("\nRare features:")
print(f"Features present in only 1 isolate: "
      f"{(frequency['n_isolates_present'] == 1).sum()}")
print(f"Features present in <=5 isolates: "
      f"{(frequency['n_isolates_present'] <= 5).sum()}")
print(f"Features present in <=10 isolates: "
      f"{(frequency['n_isolates_present'] <= 10).sum()}")

print("\nDONE")
print(f"QC file saved to: {output_file}")
