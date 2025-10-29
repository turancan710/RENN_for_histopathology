#import os
#import numpy as np
import torch
#from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from collections import defaultdict
import math
import random
#from histo_feat_extr_backbone import get_inception_v3_feature_extractor, extract_feat_from_h5
from pathlib import Path

def train_val_split(group_ids, train_frac):

  #create dict group:index lists
  group_to_indices = defaultdict(list)
  for idx, gid in enumerate(group_ids):
    group_to_indices[gid].append(idx)
  
  groups = list(group_to_indices.keys())
  random.shuffle(groups)

  num_train = int(len(groups) * train_frac)
  train_groups = set(groups[:num_train]) #set for faster gid in traingroups check, below!
  val_groups = set(groups[num_train:])

  indices_train , indices_val = [] , []

  for gid , indices in group_to_indices.items():
    if gid in train_groups:
      indices_train.extend(indices)
    else:
      indices_val.extend(indices)

  return indices_train , indices_val
  
### Sampler for 2 batch case (not needed for histopath)
class GroupABSplitSampler(Sampler):
  def __init__(self, group_ids, m_batch_ratio):
    self.m_batch_ratio = m_batch_ratio
    self.group_to_indices = defaultdict(list)

    #create dict group:index lists
    for idx, gid in enumerate(group_ids):
      self.group_to_indices[gid].append(idx) # 1:[0,1,6] 2:[2,7] 3:[3,4,5] = for each class (=key) list of sample indices belonging to that class

    self.groups = list(self.group_to_indices.keys()) # 1, 2, 3

  def __iter__(self):
    random.shuffle(self.groups)
    for group in self.groups:
      indices = self.group_to_indices[group]
      n = len(indices)

      if self.m_batch_ratio in [0,1]:
        yield indices
      else:
        random.shuffle(indices)
        k = max(1, int(math.ceil(self.m_batch_ratio * n)))
        batch_A = indices[:k]
        batch_B = indices[k:]
        if batch_A:
          yield batch_A
        if batch_B:
          yield batch_B

  def __len__(self):
    if self.m_batch_ratio in [0,1]:
      return len(self.groups) #a minibatch per group
    else:
      return len(self.groups) * 2 #2 minibatches per group

### sampler for group exclusive batches with batch size
class GroupMinibatchFixedSizeSampler(Sampler):
  def __init__(self, group_ids, m_batch_size):
    self.m_batch_size = m_batch_size
    self.group_to_indices =  defaultdict(list)
    
    for idx, gid in enumerate(group_ids):
      self.group_to_indices[gid].append(idx) # 1:[0,1,6] 2:[2,7] 3:[3,4,5] = for each class (=key) list of sample indices belonging to that class

    self.groups = list(self.group_to_indices.keys())

  def __iter__(self):
    random.shuffle(self.groups)
    for group in self.groups:
      indices = self.group_to_indices[group]
      random.shuffle(indices)

      for i in range(0, len(indices), self.m_batch_size):
        mb = indices[i:i+self.m_batch_size]
        if mb:
          yield mb
  
  def __len__(self):
    len = sum(math.ceil(len(indices)/self.m_batch_size) 
              for indices in self.group_to_indices.values())
    return len


def save_model(model, path, epoch):
  torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict()
    }, path)

#main training function
def train(model, train_dataloader, val_dataloader, group_sizes_tensor, loss_mode, num_epochs, checkp_epoch, checkp_dir, device, 
          lr_schedule_rate, 
          lr_f, lr_sig, 
          lr_f_weight_decay,
          lr_sig_weight_decay):
  #optimizer = torch.optim.AdamW(model.parameters(), lr=lr , weight_decay=0 ) #weight_decay)
  optimizer = torch.optim.AdamW([
    {
        "params": [p for name, p in model.named_parameters() if "log_sigma2" not in name],
        "lr": lr_f,
        "weight_decay": lr_f_weight_decay
    },
    {
        "params": [p for name, p in model.named_parameters() if "log_sigma2" in name],
        "lr": lr_sig,  # High lr -> faster convergence
        "weight_decay": lr_sig_weight_decay  # No regul. variances
    }
    ])
  
  lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=16 , gamma=lr_schedule_rate)

  #inc_v3_feat_extractor = get_inception_v3_feature_extractor(device) #frozen inception v3 feature extractor
  model.to(device) #lmm_nn 
  group_sizes_tensor.to(device)
  
  train_results = {"train_loss": [],
                   "val_loss": [], 
                   "sigma2_b_est_list": [], 
                   "sigma2_e_est_list": []
                   }
  best_avg_val_loss = math.inf

  for epoch in range(num_epochs):
    
    model.train()
    total_train_loss = 0
    total_train_samples = 0
    for X_batch, y_batch, z_batch in train_dataloader:
      X_batch = X_batch.to(device)
      y_batch = y_batch.to(device)
      z_batch = z_batch.to(device)

      '''# inception v3 feature extractor backbone ---> prior to training, see 00_get_features_from_tiles.py
      features = []
      for h5_path in X_batch:
        f = extract_feat_from_h5(h5_path, inc_v3_feat_extractor, device)
        features.append(f)
      X_batch_features = torch.stack(features).to(device) # m , 2048
      '''
      
      optimizer.zero_grad()
      f_X = model(X_batch)
      
      if loss_mode == 'NLL':
        loss = model.nll(y_batch, f_X)
      elif loss_mode == 'NLL_cor_1':
        gr_id = z_batch[0].item()
        gr_size = group_sizes_tensor[gr_id]
        loss = model.nll_cor_1(y_batch, f_X, gr_size)
      elif loss_mode == 'NLL_cor_2':
        gr_id = z_batch[0].item()
        gr_size = group_sizes_tensor[gr_id]
        loss = model.nll_cor_2(y_batch, f_X, gr_size)
      elif loss_mode == 'NLL_no_RE':
        loss = model.nll_no_re(y_batch, f_X)
      elif loss_mode == 'NLL_cor_det':
        gr_id = z_batch[0].item()
        gr_size = group_sizes_tensor[gr_id]
        loss = model.nll_cor_det(y_batch, f_X, gr_size)
      
      loss.backward()
      optimizer.step()
      total_train_loss += loss.item() #* len(y_batch)
      total_train_samples += y_batch.size(0)
    
    model.eval()
    total_val_loss = 0
    total_val_samples = 0 
    with torch.no_grad():
      for  X_batch, y_batch, z_batch in val_dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        z_batch = z_batch.to(device)
        
        '''
        # inception v3 feature extractor backbone ---> prior to training, see 00_get_features_from_tiles.py
        features = []
        for h5_path in X_batch:
          f = extract_feat_from_h5(h5_path, inc_v3_feat_extractor, device)
          features.append(f)
      
        X_batch_features = torch.stack(features).to(device) # m , 2048
        '''
        
        optimizer.zero_grad()
        f_X = model(X_batch)
        
        if loss_mode == 'NLL':
          loss = model.nll(y_batch, f_X)
        elif loss_mode == 'NLL_cor_1':
          gr_id = z_batch[0].item()
          gr_size = group_sizes_tensor[gr_id]
          loss = model.nll_cor_1(y_batch, f_X, gr_size)
        elif loss_mode == 'NLL_cor_2':
          gr_id = z_batch[0].item()
          gr_size = group_sizes_tensor[gr_id]
          loss = model.nll_cor_2(y_batch, f_X, gr_size)
        elif loss_mode == 'NLL_no_RE':
          loss = model.nll_no_re(y_batch, f_X)
        elif loss_mode == 'NLL_cor_det':
          gr_id = z_batch[0].item()
          gr_size = group_sizes_tensor[gr_id]
          loss = model.nll_cor_det(y_batch, f_X, gr_size)

        total_val_loss += loss.item() #* len(y_batch)
        total_val_samples += y_batch.size(0)
    
    avg_train_loss = total_train_loss / total_train_samples  #len(dataloader.dataset) ?
    avg_val_loss = total_val_loss / total_val_samples
    sigma2_e_est , sigma2_b_est = model.get_var_est()

    train_results['train_loss'].append(avg_train_loss)
    train_results['val_loss'].append(avg_val_loss)
    train_results['sigma2_e_est_list'].append(sigma2_e_est)
    train_results['sigma2_b_est_list'].append(sigma2_b_est)

    print(f"Epoch {epoch+1:02d}: train-{loss_mode}:{avg_train_loss:.4f} ; val-{loss_mode}:{avg_val_loss:.4f}; Var Estimates: sigma2_e: {sigma2_e_est:.4f} ; sigma2_b: {sigma2_b_est:.4f}")

    if (epoch + 1) % checkp_epoch == 0 or (epoch +1) == num_epochs:
     checkp_path = checkp_dir / f"model_epoch_{epoch+1:03d}.pt"
     save_model(model, checkp_path, epoch+1)

    if avg_val_loss < best_avg_val_loss:
      best_avg_val_loss = avg_val_loss
      best_path = checkp_dir / "best_model.pt"
      save_model(model, best_path, epoch+1)

    lr_scheduler.step()
      
  return train_results