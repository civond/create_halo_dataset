import pandas as pd
import numpy as np
import os

read_dir = "./output/"

datasets = []
for file in os.listdir(read_dir):
    if file.endswith(".csv"):
        temp_csv_path = os.path.join(read_dir, file)
        temp_df = pd.read_csv(temp_csv_path)
        datasets.append(temp_df)


combined_dataset = pd.concat(datasets, ignore_index=True)
combined_dataset = combined_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
combined_dataset['fold'] = np.arange(len(combined_dataset)) % 5
combined_dataset = combined_dataset.sort_values('fold').reset_index(drop=True)
#folds = np.array_split(combined_dataset, 5)

print(combined_dataset)
print(combined_dataset['halo_present'].value_counts(dropna=False))
combined_dataset.to_csv('output.csv', index=False)
#print(folds)