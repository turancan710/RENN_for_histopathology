import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
data_dir = project_root / 'data'
meta_path = data_dir / 'tcga_brca_file_ids_metadata.csv'
gene_path = data_dir / 'tcga_brca_gene_expression_tpm_unstr.csv'

df_tile_meta = pd.read_csv(meta_path)
df_genes = pd.read_csv(gene_path)

df_merge = pd.merge(df_tile_meta, df_genes[['case_id', 'MKI67_tpm_unst']], on='case_id', how='left')

out_path = data_dir / 'tcga_brca_dataset_tile_meta_gene_mki67.csv'
df_merge.to_csv(out_path)