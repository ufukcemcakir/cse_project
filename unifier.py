import os
import pandas as pd

# Path to your AL-CPL dataset directory
DATA_PATH = "AL-CPL-dataset-master/data"  # 👈 Change this to match your local path

# Initialize empty list
all_pairs = []

# Loop through all .pairs files (not .preqs)
for fname in os.listdir(DATA_PATH):
    if fname.endswith(".pairs"):
        domain = fname.replace(".pairs", "")
        filepath = os.path.join(DATA_PATH, fname)
        
        # Read the file (2 columns: concept_A, concept_B)
        df = pd.read_csv(filepath, header=None, names=["concept_A", "concept_B"])

        # Create positive label set from corresponding .preqs file
        preqs_file = os.path.join(DATA_PATH, f"{domain}.preqs")
        preqs = set()
        if os.path.exists(preqs_file):
            df_pos = pd.read_csv(preqs_file, header=None, names=["concept_A", "concept_B"])
            preqs = set(tuple(x) for x in df_pos.values)

        # Assign labels based on whether the pair exists in .preqs
        df["label"] = df.apply(lambda row: 1 if (row["concept_A"], row["concept_B"]) in preqs else 0, axis=1)

        # Clean text
        df["concept_A"] = df["concept_A"].str.strip().str.lower()
        df["concept_B"] = df["concept_B"].str.strip().str.lower()

        all_pairs.append(df)

# Combine all domains into one DataFrame
final_df = pd.concat(all_pairs, ignore_index=True)

# Drop duplicates and self-pairs
final_df = final_df[final_df["concept_A"] != final_df["concept_B"]]
final_df = final_df.drop_duplicates()

# Save to CSV
final_df.to_csv("concept_pairs.csv", index=False)
print(f"✅ Saved {len(final_df)} labeled concept pairs to concept_pairs.csv")
