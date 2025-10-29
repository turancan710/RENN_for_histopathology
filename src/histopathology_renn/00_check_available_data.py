import pandas as pd
import numpy as np
#import os
from pathlib import Path


project_root = Path(__file__).resolve().parents[2]
data_dir = project_root / 'data'
csv_path = data_dir / 'tcga_brca_dataset_tile_meta_gene_mki67.csv' 
feat_path = data_dir / 'incv3_frozen_features' / 'features.npz' #extracted 2048 dim feature vectors via inception backbone

df = pd.read_csv(csv_path)
features = np.load(feat_path)

print(f'rows in csv {len(df)}')
print(f'feature vectors in features.npz {len(features.files)}')

df_filtered = df[
    df['MKI67_tpm_unst'].notnull() &
    df['tss_code'].notnull() &
    df['file_id'].notnull()
]

print(f'rows with mki67, tss_code , file_id {len(df_filtered)}')

file_ids_with_features = set(features.files) # get keys
df_filtered = df_filtered[df_filtered['file_id'].isin(file_ids_with_features)]

print(f"final dataset with available mki67, tss_code, file_id, feat_vector {len(df_filtered)}")