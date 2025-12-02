import pandas as pd
import glob
import random


csv_files = glob.glob("*.csv")

dfs = [pd.read_csv(f) for f in csv_files]
combined_df = pd.concat(dfs, ignore_index=True)

combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

combined_df.to_csv("combined_shuffled.csv", index=False, quoting=1)

print(f"Creato 'combined_shuffled.csv' con {len(combined_df)} righe.")
