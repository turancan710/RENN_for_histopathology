import os
import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines

dir_exp = Path(__file__).resolve().parents[0]
csv_path = dir_exp / 'results_flat.csv'
df = pd.read_csv(csv_path)

palette = {
    'RENN': '#1f77b4',  # blue
    'RENN-cor':'#ff7f0e',  # orange
    'NN-noRE': '#2ca02c',    #green
    'RENN-det': "#9c00aa" #pink
}

#long dataset (val loss & train loss -> separate row each), make 4 labels as above
df_melted = df.melt(
    id_vars=['loss_func_disp', 'm', 'run', 'epoch'],
    value_vars=['sigma2_b_est' , 'sigma2_e_est'],
    var_name='sigma_type', 
    value_name='sigma_estimate'
)

df_melted = df_melted[~((df_melted['loss_func_disp'] == 'NN-noRE') & (df_melted['sigma_type'] == 'sigma2_b_est' ))]
#df_melted['sig_label'] =  df_melted['sigma_type'] + '_' + df_melted['loss_func_disp']

sigma_type_map = {
    'sigma2_b_est': r'$\hat{\sigma}^2_b$',
    'sigma2_e_est': r'$\hat{\sigma}^2_\epsilon$'
}
df_melted['sigma_type_disp'] = df_melted['sigma_type'].map(sigma_type_map)
#df_melted['sig_label_disp'] = df_melted['sigma_type_disp'] + '-' + df_melted['loss_func_disp']

#subset = df_melted.query('m==0 and loss_label == "NLL_cor_1_train" ')
#print(subset.groupby(['epoch']).loss_value.describe())
df_melted = df_melted.rename(
    columns={
    'loss_func_disp': 'Model', #RENN, RENN-cor, NN-noRE
    'sigma_type_disp': 'Parameter' #train val
    }
    )


#sns.set(style='whitegrid')
sns.set(style='whitegrid',
        context="notebook", 
        font_scale=1.7) 

g = sns.relplot(
    data = df_melted,
    x='epoch',
    y='sigma_estimate',
    hue='Model',
    style='Parameter',
    col='m',
    palette=palette,
    kind='line',
    col_wrap=3,
    #height=3.5,
    estimator='mean',
    errorbar=None ,
    facet_kws={'sharey': True}, #dyn y axis
    linewidth=2
    )


g.set_axis_labels('Epoch', 'Parameter Estimate')
g.set_titles('m = {col_name}')
g.fig.suptitle('Estimated Variances across Batch Size', y=1.03)#(10 Runs Mean)

sns.move_legend(g, title=None, loc='lower center', 
             bbox_to_anchor=(0.5, .93),
             ncol=9,
             frameon=False
             )
#g.legend.set_title('Parameter Estimate & Model')

plt.tight_layout()
out_path = dir_exp / "plots_losses_and_params" / f"e02_02_HISTO_sigma_epoch_grid_by_m.png"
plt.savefig(out_path, dpi=300, bbox_inches='tight')