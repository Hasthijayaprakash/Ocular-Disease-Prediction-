"""
merge_embeddings.py

We are using this Code to merge the embeddings with the patients dataset
"""



import pandas as pd
from pathlib import Path

# Paths to the files
embeddings_parquet = 'E:\\mld\\advanced\\output\\feature_embeddings.parquet'
patients_excel = 'E:\\mld\\advanced\\\data.xlsx'

# Loading the embeddings Parquet file
embeddings_df = pd.read_parquet(embeddings_parquet)

# Loading the patients.csv file
patients_df = pd.read_excel(patients_excel)

patients_melted = patients_df.melt(
    id_vars=['ID', 'Patient Age', 'Patient Sex'],
    value_vars=['Left-Fundus', 'Right-Fundus'],
    var_name='Fundus Side',
    value_name='image_name'
)

# Clean image_name columns 
embeddings_df['image_name'] = embeddings_df['image_name'].str.strip()
patients_melted['image_name'] = patients_melted['image_name'].str.strip()

# Merging embeddings_df with patients_melted on 'image_name'
merged_df = pd.merge(embeddings_df, patients_melted, on='image_name', how='inner')



# Saving the merged DataFrame
output_merged = 'E:\\mld\\advanced\\resnet\\output\\merged_embeddings.parquet'
merged_df.to_parquet(output_merged, index=False)
print(f"Merged embeddings saved to {output_merged}")
