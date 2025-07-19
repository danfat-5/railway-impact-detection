# step1.py - in this file I clean the raw data and save it for next steps

import pandas as pd

# I load the 3 CSV files that have sensor features
file1 = pd.read_csv("Trail1_extracted_features_acceleration_m1ai1-1.csv")
file2 = pd.read_csv("Trail2_extracted_features_acceleration_m1ai1.csv")
file3 = pd.read_csv("Trail3_extracted_features_acceleration_m2ai0.csv")

# I put them together in one big table
all_data = pd.concat([file1, file2, file3], ignore_index=True)

# I remove some columns that I don't need in the model
cols_to_remove = ['start_time', 'axle', 'cluster', 'tsne_1', 'tsne_2']
all_data.drop(columns=[c for c in cols_to_remove if c in all_data.columns], inplace=True)

# Now I change event column → normal = 0, other = 1
all_data['event'] = all_data['event'].apply(lambda x: 0 if str(x).strip().lower() == 'normal' else 1)

# I show some rows to make sure it works
print(all_data.head())

# I save this cleaned version for next use
all_data.to_csv("cleaned_data.csv", index=False)
