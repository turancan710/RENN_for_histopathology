#import os
import pandas as pd
import numpy as np
import torch
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker

import sys
''' old version to add module path to sys (enable import of modules)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))) #= .../app_histo_renn for model module import
'''
project_root = Path(__file__).resolve().parents[3]
module_dir = project_root / 'src' / 'histopathology_renn'
sys.path.append(str(module_dir))

from histo_model import lmm_nn
from histo_dataset import load_data_log

dir_exp = Path(__file__).resolve().parents[0]
res_plot_dir = dir_exp / 'plots_residuals'
res_plot_dir.mkdir(parents=True, exist_ok=True)

palette = {
    'RENN': '#1f77b4',  # blue
    'RENN-cor':'#ff7f0e',  # orange
    'NN-noRE': '#2ca02c',    #green
    'RENN-det': "#9c00aa" #pink
}


# ==== true X features, y, groups from Histo dataset ===========

data_dir = project_root / 'data' 
csv_path = data_dir / 'tcga_brca_dataset_tile_meta_gene_mki67.csv'
features_path = data_dir / 'incv3_frozen_features' / 'features.npz' 

X, y, group_ids, _ = load_data_log(csv_path, features_path) #all torch tensors already

#==== find best model per loss X m config ===========

df = pd.read_csv( dir_exp / 'results_flat.csv' )

df_best_runs= ( df.loc[df.groupby(['loss_func' , 'm', 'run'])['val_loss'].idxmin()] )

#==== predict y_hat per best model ===========
model = lmm_nn(2048) #input dim p=2048 fixed effects -> feature vector size of Inception_V3 

for idx , row in df_best_runs.iterrows():
    loss = row['loss_func']
    loss_disp = row['loss_func_disp']
    m = row['m']
    run = row['run']
    epoch = int(row['epoch'])
    val_loss = row['val_loss']
    sigma2_b_est = row['sigma2_b_est']
    sigma2_e_est = row['sigma2_e_est']

    m_str = f"{int(round(m*100)):03d}"
    model_path = dir_exp / 'results' / f"conf_m{m_str}_{loss}" / f"run_{run}" / "best_model.pt"

    best_model = torch.load(model_path, map_location='cpu')
    
    if int(epoch +1 ) != best_model['epoch']:
        print(f"WARNING:for {loss} , m={m}: Best Val Loss out pf 10 runs in: run={run} , epoch = {int(epoch+1)} \n in run={run} best_model.pt saved in epoch {best_model['epoch']}")
    else:
        print(f"INFO: for {loss} , m={m}: Best Val Loss out pf 10 runs in: run={run} , epoch = {int(epoch+1)} \n in run={run} best_model.pt saved in epoch {best_model['epoch']}")
    
    model.load_state_dict(best_model['model_state_dict'])
    model.eval()

    with torch.no_grad():
        fX = model(X)
     
    fX_np = fX.numpy().flatten()
    y_np = y.numpy().flatten()
    group_ids_np = group_ids.numpy()

    if loss == 'NLL_no_RE':
        mse = np.mean( (y_np - fX_np)**2 )
    
    elif loss == 'NLL':
        df_groups = pd.DataFrame({
            'group_id': group_ids_np,
            'res': y_np - fX_np
        })
    
        group_res_means = df_groups.groupby('group_id')['res'].mean()
        group_sizes = df_groups.groupby('group_id')['res'].size()

        b_hat = (sigma2_b_est / (sigma2_b_est + (sigma2_e_est / group_sizes))) * group_res_means

        y_hat = fX_np + np.array([b_hat[i] for i in group_ids_np])
        mse = np.mean( (y_np - y_hat)**2 )

    elif loss == 'NLL_cor_1':
        df_groups = pd.DataFrame({
            'group_id': group_ids_np,
            'res': y_np - fX_np
        })
    
        group_res_means = df_groups.groupby('group_id')['res'].mean()
        group_sizes = df_groups.groupby('group_id')['res'].size()

        b_hat = (sigma2_b_est / (sigma2_b_est + (sigma2_e_est / group_sizes))) * group_res_means

        y_hat = fX_np + np.array([b_hat[i] for i in group_ids_np])
        mse = np.mean( (y_np - y_hat)**2 )
    
    elif loss == 'NLL_cor_det':
        df_groups = pd.DataFrame({
            'group_id': group_ids_np,
            'res': y_np - fX_np
        })
    
        group_res_means = df_groups.groupby('group_id')['res'].mean()
        group_sizes = df_groups.groupby('group_id')['res'].size()

        b_hat = (sigma2_b_est / (sigma2_b_est + (sigma2_e_est / group_sizes))) * group_res_means

        y_hat = fX_np + np.array([b_hat[i] for i in group_ids_np])
        mse = np.mean( (y_np - y_hat)**2 )

    df_best_runs.loc[idx, 'mse'] = mse

#plt.figure(figsize=(8,5))
sns.set(style='whitegrid')
df_best_runs['m_cat'] = df_best_runs['m'].astype(str)

g = sns.lineplot(
    data = df_best_runs,
    x='m_cat',
    y='mse',
    hue = 'loss_func_disp',
    palette=palette,
    marker='o',
    estimator='mean',
    errorbar=None
)
#g.set(yscale='log')
g.set_title('MSE of Predictions over Batch Size' ) #(Best Model per Run - 10 Runs Average)
g.set_xlabel('Batch Size (m)')
g.set_ylabel(r'MSE: y - $\hat{y}$')
g.legend(title='Model', fontsize='small' , title_fontsize='small')
#g.set_xticks(np.arange(0.0 , 1.01, 0.1))

plt.tight_layout()
out_path = res_plot_dir / f"e02_07_HISTO_mse_pred_m.png"
plt.savefig(out_path, dpi=300, bbox_inches='tight')