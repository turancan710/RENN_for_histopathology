#import os
import json
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

#print("current wd" , os.getcwd())
dir_exp = Path(__file__).resolve().parents[0] #-> 02_exp_log_gene_incepv3_freeze

#with open('./simulation/results/00_exp_std_setting/conf_m050_sigb0100_sige0100_NLL/run_0/training_results.json' , 'r') as f:
#    data = json.load(f)

#print(data['train_results'])

# =========  FLAT DATSET  ===================================================================================================

results_data = []
LOSS_MODES = ['NLL', 'NLL_no_RE' , 'NLL_cor_1', 'NLL_cor_det']
M_BATCH_SIZE = [2, 4 , 8 , 16 , 32 , 150]
num_runs = 10
loss_func_map = {
    'NLL': 'RENN', 
    'NLL_no_RE': 'NN-noRE', 
    'NLL_cor_1': 'RENN-cor',
    'NLL_cor_det': 'RENN-det'
    }

for loss in LOSS_MODES:
    for m in M_BATCH_SIZE:
        for run in range(num_runs):
            m_str= f"{int(round(m*100)):03d}"
            path = dir_exp / 'results'/ f"conf_m{m_str}_{loss}" / f"run_{run}" / "training_results.json"
            if path.exists():
                with open(path, 'r') as f:
                    data = json.load(f)
                    epochs = len(data['train_results']['train_loss'])
                    for epoch in range(epochs):
                        results_data.append({
                            "loss_func": loss,
                            'loss_func_disp': loss_func_map[loss],
                            "m": m , 
                            "run": run , 
                            "epoch": epoch,
                            "train_loss": data['train_results']['train_loss'][epoch],
                            "val_loss": data['train_results']['val_loss'][epoch],
                            "sigma2_b_est": data['train_results']['sigma2_b_est_list'][epoch],
                            "sigma2_e_est": data['train_results']['sigma2_e_est_list'][epoch]
                            })

df = pd.DataFrame(results_data)

out_path = dir_exp / 'results_flat.csv'
df.to_csv(out_path)