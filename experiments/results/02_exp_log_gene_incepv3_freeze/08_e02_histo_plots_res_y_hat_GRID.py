import os
import pandas as pd
import numpy as np
import torch
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker

import sys
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
project_root = Path(__file__).resolve().parents[3]
module_dir = project_root / 'src' / 'histopathology_renn'
sys.path.append(str(module_dir))

from histo_model import lmm_nn
from histo_dataset import load_data_log

dir_exp = Path(__file__).resolve().parents[0]
res_plot_dir = dir_exp / 'plots_residuals'
res_plot_dir.mkdir(parents=True, exist_ok=True)

palette = {
    'RENN': '#1f77b4',  #blue
    'RENN-cor':'#ff7f0e',  # orange
    'NN-noRE': '#2ca02c',    #green
    'RENN-det': "#9c00aa" #pink
    }


'''
Residual= y-y_pred ->Generates Scatterplots, Histograms, Boxplots across Minibatch Sizes
'''

# ==== true X features, y, groups from Gisto dataset ===========
data_dir = project_root / 'data' 
csv_path = data_dir / 'tcga_brca_dataset_tile_meta_gene_mki67.csv'
features_path = data_dir / 'incv3_frozen_features' / 'features.npz'

X, y, group_ids, _ = load_data_log(csv_path, features_path) #all torch tensors already

#==== find best model per loss X m config ===========
df = pd.read_csv( dir_exp / 'results_flat.csv')

df_best_runs= ( df.loc[df.groupby(['loss_func' , 'm'])['val_loss'].idxmin()] )
#print(df_best_runs)

residual_dfs = []
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
        residual = y_np - fX_np #fX = y_hat
        y_hat = fX_np
    
    elif loss =='NLL':
        df_groups = pd.DataFrame({
            'group_id': group_ids_np,
            'res': y_np - fX_np
        })
    
        group_res_means = df_groups.groupby('group_id')['res'].mean()
        group_sizes = df_groups.groupby('group_id')['res'].size()

        b_hat = (sigma2_b_est / (sigma2_b_est + (sigma2_e_est / group_sizes))) * group_res_means

        y_hat = fX_np + np.array([b_hat[i] for i in group_ids_np])
        residual = y_np - y_hat
    
    elif loss =='NLL_cor_1':
        df_groups = pd.DataFrame({
            'group_id': group_ids_np,
            'res': y_np - fX_np
        })
    
        group_res_means = df_groups.groupby('group_id')['res'].mean()
        group_sizes = df_groups.groupby('group_id')['res'].size()

        b_hat = (sigma2_b_est / (sigma2_b_est + (sigma2_e_est / group_sizes))) * group_res_means

        y_hat = fX_np + np.array([b_hat[i] for i in group_ids_np])
        residual = y_np - y_hat

    elif loss =='NLL_cor_det':
        df_groups = pd.DataFrame({
            'group_id': group_ids_np,
            'res': y_np - fX_np
        })
    
        group_res_means = df_groups.groupby('group_id')['res'].mean()
        group_sizes = df_groups.groupby('group_id')['res'].size()

        b_hat = (sigma2_b_est / (sigma2_b_est + (sigma2_e_est / group_sizes))) * group_res_means

        y_hat = fX_np + np.array([b_hat[i] for i in group_ids_np])
        residual = y_np - y_hat

    df_res = pd.DataFrame({
        'y_hat': y_hat,
        'y': y_np,
        'residual': residual,
        'm': m,
        'loss_func': loss,
        'loss_func_disp': loss_disp
    })
    residual_dfs.append(df_res)

df_residuals = pd.concat(residual_dfs, ignore_index=True)
#df_residuals['m_cat']=df_residuals['m'].astype(str)


# === scatter plot Residual Grid &  y_hat vs y Grid ===================================================
for loss_disp in df_residuals['loss_func_disp'].unique():
    df_res_sub = df_residuals[df_residuals['loss_func_disp'] == loss_disp]

    #plt.figure(figsize=(8, 16)) 
    sns.set(style='whitegrid')

    g = sns.relplot(
        data=df_res_sub,
        x='y',
        y='residual',
        col='m',
        col_wrap=3,
        kind='scatter',
        alpha=0.3,
        s=8,
        height=3,
        color=palette[loss_disp]
    )

    g.set_titles("m ={col_name}")
    g.fig.suptitle(rf'Prediction Residuals across Batch Size ({loss_disp})' , y=1.02 ) #: Best Model out of 10 Runs
    g.set_axis_labels('y', r"y - $\hat{y}$") #$\\hat{y}$

    plt.tight_layout()
    out_path = res_plot_dir / f"e02_08_HISTO_{loss_disp}_PRED_residuals_grid_by_m.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')

    f = sns.relplot(
        data=df_res_sub,
        x='y',
        y='y_hat',
        col='m',
        col_wrap=3,
        kind='scatter',
        alpha=0.3,
        s=8,
        height=3,
        color=palette[loss_disp]
    )

    f.set_titles("m ={col_name}")
    f.fig.suptitle(rf'Prediction $\hat{{y}}$ vs. True y across Batch Size ({loss_disp})' , y=1.02 ) # $\hat{y}$#: Best Model out of 10 Runs)
    f.set_axis_labels('y', r"$\hat{y}$")

    plt.tight_layout()
    out_path = res_plot_dir / f"e02_08_HISTO_{loss_disp}_PRED_grid_by_m.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')

# === Histo Grid ===================================================
for loss_disp in df_residuals['loss_func_disp'].unique():
    df_res_sub = df_residuals[df_residuals['loss_func_disp'] == loss_disp]

    g = sns.displot(
        data=df_res_sub,
        x='residual',
        col='m',
        col_wrap=3,
        height=3,
        bins=40,
        kde=True,
        facet_kws={'sharex':True, 'sharey':True},
        color=palette[loss_disp]
    )
    g.set_titles("m ={col_name}")
    g.fig.suptitle(rf'Distributions of Prediction Residuals across Batch Size ({loss_disp})' , y=1.02 ) #: Best Model out of 10 Runs)
    g.set_axis_labels(r'y - $\hat{y}$' , 'Count')

    plt.tight_layout()
    out_path = res_plot_dir / f"e02_08_HISTO_{loss_disp}_PRED_residuals_Histo_grid_by_m.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')

# === boxplots ===================================================

#sns.set(style='whitegrid')
#plt.figure(figsize=(10,5))
sns.set(style='whitegrid',
        context="notebook", 
        font_scale=1.7)

g = sns.catplot(
    data = df_residuals,
    x='m',
    y='residual',
    col='loss_func_disp',
    hue='loss_func_disp',
    kind='box',
    height=5,
    aspect=0.8,
    dodge=True,
    sharey=True,
    palette=palette
)

g.set_titles("{col_name}")
g.fig.suptitle(rf'Prediction Residuals over Batch Size' , y=1.06 ) # (Best Model out of 10 Runs)
g.set_axis_labels( 'Batch Size (m)', r'y - $\hat{y}$')

sns.move_legend(g, 'lower center', title=None,  
             bbox_to_anchor=(0.5, .81),
             ncol=5,
             frameon=False)
g.legend.set_title('Model')
for ax in g.axes.flat:
    ax.tick_params(axis='x', rotation=45) 

plt.tight_layout()
out_path = res_plot_dir / f"e02_08_HISTO_PRED_residuals_Boxplots.png"
plt.savefig(out_path, dpi=300, bbox_inches='tight')