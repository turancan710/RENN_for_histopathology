import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


#extracts X=file_ids , y=mki67 expression, groups as indices 0,...q-1 from medical center codes
def load_data(csv_path, features_path):
  df = pd.read_csv(csv_path)

  df = df[df['MKI67_tpm_unst'].notnull()] #y target variable
  df = df[df['tss_code'].notnull()] #groups
  df = df[df['file_id'].notnull()]
  #main_dir = os.path.dirname(csv_path)
  #df['tiles_path'] = df['tiles_path'].apply(lambda rel_path: os.path.normpath(os.path.join(main_dir, rel_path)))
  #df = df[df['tiles_path'].apply(os.path.exists)]
  
  #extracted features
  features = np.load(features_path)
  file_ids_with_features = set(features.files) #get keys of the npz dictionary {file_id : np_features}

  #keep rows with features only
  df = df[df['file_id'].isin(file_ids_with_features)]

  if df.empty:
    raise ValueError('filtered for file_ids with features: no samples')

  group_map = {name: idx for idx, name in enumerate(df['tss_code'].unique())}
  group_indices = torch.tensor( df['tss_code'].map(group_map).astype(int).values , dtype = torch.long)

  #tiles = df['tiles_path'].values.tolist()
  features_list =[features[file_id] for file_id in df['file_id']]
  X = torch.tensor(np.stack(features_list), dtype=torch.float32)
  
  y = torch.tensor( df['MKI67_tpm_unst'].values , dtype = torch.float32)

  return (
    X,
    y,
    group_indices,
    group_map
    )


#### log-MKI67 
def load_data_log(csv_path, features_path):
  df = pd.read_csv(csv_path)

  df = df[df['MKI67_tpm_unst'].notnull()] #y target variable
  df = df[df['tss_code'].notnull()] #groups
  df = df[df['file_id'].notnull()]
  #main_dir = os.path.dirname(csv_path)
  #df['tiles_path'] = df['tiles_path'].apply(lambda rel_path: os.path.normpath(os.path.join(main_dir, rel_path)))
  #df = df[df['tiles_path'].apply(os.path.exists)]
  
  #extracted features
  features = np.load(features_path)
  file_ids_with_features = set(features.files) #get keys of the npz dictionary {file_id : np_features}

  #keep rows with features only
  df = df[df['file_id'].isin(file_ids_with_features)]

  if df.empty:
    raise ValueError('filtered for file_ids with features: no samples')

  group_map = {name: idx for idx, name in enumerate(df['tss_code'].unique())}
  group_indices = torch.tensor( df['tss_code'].map(group_map).astype(int).values , dtype = torch.long)

  #tiles = df['tiles_path'].values.tolist()
  features_list =[features[file_id] for file_id in df['file_id']]
  X = torch.tensor(np.stack(features_list), dtype=torch.float32)
  
  y_raw = df['MKI67_tpm_unst'].values
  y_log = np.log(y_raw)
  y = torch.tensor( y_log , dtype = torch.float32)

  return (
    X,
    y,
    group_indices,
    group_map
    )
                                
# https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
class RandIntDataset(Dataset):
  #full dataset
  def __init__(self, X, y, groups):
    self.X = X
    self.y = y
    self.groups = groups
  
  def __len__(self):
    return len(self.y)
  
  #return samples to the dataloader
  def __getitem__(self, index):
    return (
      self.X[index], 
      self.y[index], 
      self.groups[index]
    )