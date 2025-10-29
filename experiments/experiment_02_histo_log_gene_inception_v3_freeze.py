#import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from collections import Counter
import json
import time
from pathlib import Path

from histo_dataset import RandIntDataset, load_data, load_data_log
from histo_model import lmm_nn
from histo_train import GroupABSplitSampler , train_val_split, train , GroupMinibatchFixedSizeSampler
from histo_utils import save_data , global_seed , worker_init_fn

#========= CONFIG  =============================================================================================
# here training configurations can be set
#feature extraction has to be done prior (run 00_get_features_from_tiles.py) !!!

#input lmm_nn = output size inception v3 (before class. head)
p = 2048 

#Training
M_BATCH_SIZE = [2, 4 , 8 , 16 , 32 , 150]
LOSS_MODES = ['NLL_cor_det','NLL', 'NLL_no_RE' , 'NLL_cor_1']
TRAIN_DATA_FRAC = 0.8 # training / validaiton set
NUM_EPOCHS = 60

#learning rate for DNN and sigma params + respective weight decay 
lr_f = 0.000025 
lr_f_weight_decay=0.001
lr_sig=0.0015 
lr_sig_weight_decay=0

#lr schedule for both lrs / 16 epochs
lr_schedule_rate = 0.5

EPOCH_CHECKP = 20 #checkpoint for model saving every 20 epochs

NUM_RUNS = 10 #runs per config
RUN_SEEDS = [17, 41, 92, 14, 68, 31, 89, 15, 21, 60] #each run gets seed from RUN_SEEDS list 
init_seed = 62 #here for the datset sampling. 

##directories
project_root = Path(__file__).resolve().parents[1]
data_dir = project_root / 'data'
results_dir = project_root / 'experiments' / 'results'

setting =  "02_exp_log_gene_incepv3_freeze"
exp_dir = results_dir / setting / 'results' # ./simulation/results/00_exp_std_setting
exp_dir.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DEVICE: {device}")

#========= TRAIN =============================================================================================  
global_seed(init_seed)

csv_path = data_dir / 'tcga_brca_dataset_tile_meta_gene_mki67.csv'
feat_path = data_dir / 'incv3_frozen_features' / 'features.npz'
X , y, groups , group_map = load_data_log(csv_path , feat_path)
dataset = RandIntDataset(X , y, groups)

group_ids = dataset.groups.tolist()
group_sizes = Counter(group_ids)
num_groups = len(group_sizes)
group_sizes_tensor = torch.tensor([group_sizes[i] for i in range(num_groups)],  dtype=torch.long)

#training loop
for loss_mode in LOSS_MODES:
    for m in M_BATCH_SIZE:
        print(f" === Backbone: Inception-V3 frozen ")
        if loss_mode == 'NLL':
            print(f" === Training with standard NLL loss , m= {m}===")

        if loss_mode == 'NLL_cor_1':
            print(f" === Training with corrected NLL loss V1 , m= {m} ===")

        if loss_mode == 'NLL_no_RE':
            print(f" === Training with NLL loss NO RE , m= {m} ===")
        
        if loss_mode == 'NLL_cor_det':
            print(f" === Training with DET corr. NLL loss, m= {m} ===")

        for run , seed in enumerate(RUN_SEEDS):
            run_start_time = time.time()

            #reproducability: run seeds
            global_seed(seed)
                
            m_str = f"{int(round(m * 100)):03d}"
            run_dir = exp_dir / f"conf_m{m_str}_{loss_mode}" / f"run_{run}"
            run_dir.mkdir(parents=True, exist_ok=True)
                
            indices_train , indices_val = train_val_split(group_ids, TRAIN_DATA_FRAC)

            train_dataset = RandIntDataset(X[indices_train] , y[indices_train], groups[indices_train])
            val_dataset = RandIntDataset(X[indices_val] , y[indices_val], groups[indices_val])    
            train_group_ids = train_dataset.groups.tolist()
            val_group_ids = val_dataset.groups.tolist()

            train_sampler = GroupMinibatchFixedSizeSampler(train_group_ids, m)
            val_sampler = GroupMinibatchFixedSizeSampler(val_group_ids, m)
            train_loader = DataLoader(train_dataset, batch_sampler= train_sampler, pin_memory=True,  worker_init_fn=worker_init_fn) #num_workers=16,
            val_loader = DataLoader(val_dataset, batch_sampler= val_sampler, pin_memory=True, worker_init_fn=worker_init_fn) #num_workers=8, 

            model = lmm_nn(input_dim=p)
            train_results = train(model, train_loader, val_loader, group_sizes_tensor, loss_mode, NUM_EPOCHS, EPOCH_CHECKP, run_dir, device, 
                                  lr_schedule_rate,
                                  lr_f, 
                                  lr_sig,
                                  lr_f_weight_decay, 
                                  lr_sig_weight_decay)

            run_time = time.time() - run_start_time
            print(f"total training time - run {run}: {run_time} sec.")

            results = {
                "train_results": train_results,
                "run_info": {
                    "run_no": run,
                    "run_time": run_time,
                    "run_seed": seed,
                    "loss_func": loss_mode,
                    "m_batch_ratio": m,
                    "device": str(device)
                    },
                "training_config": {
                    "loss_func": loss_mode,
                    "avl_loss_modes" : LOSS_MODES ,
                    "m_batch_ratio": m ,
                    "train_data_frac": TRAIN_DATA_FRAC,
                    "epochs": NUM_EPOCHS ,
                    "adam_lr_NN_params" : lr_f ,
                    "adam_weight_decay_NN_params": lr_f_weight_decay ,
                    "adam_lr_sigma": lr_sig ,
                    "adam_weight_decay_sigma": lr_sig_weight_decay ,
                    "lr_scheduler_16_epoch": lr_schedule_rate ,
                    "epoch_checkpoint": EPOCH_CHECKP ,
                    "runs_per_config": len(RUN_SEEDS) ,
                    "run_seeds": RUN_SEEDS
                    }
                }

            result_path = run_dir / "training_results.json"
            with open(result_path, "w") as f:
                json.dump(results, f)
                
            print(f"saved results to {result_path}")