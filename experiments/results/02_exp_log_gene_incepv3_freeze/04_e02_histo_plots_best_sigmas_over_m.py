#import os
import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

dir_exp = Path(__file__).resolve().parents[0]
csv_path = dir_exp / 'results_flat.csv'
df = pd.read_csv(csv_path)

palette = {
    'RENN': '#1f77b4',  # blue
    'RENN-cor':'#ff7f0e',  # orange
    'NN-noRE': '#2ca02c',    #green
    'RENN-det': "#9c00aa" #pink
}

#select per (loss x m x run) -> best epoch (min Val loss)
df_best_epochs= ( df.loc[df.groupby(['loss_func_disp' , 'm', 'run'])['val_loss'].idxmin()] )
#df_best_epochs.to_csv(os.path.join(dir_exp, 'best_epochs.csv'))


#long dataset (val loss & train loss -> separate row each), make 4 labels as above
df_melted = df_best_epochs.melt(
    id_vars=['loss_func_disp', 'm', 'run' ],
    value_vars=['sigma2_b_est' , 'sigma2_e_est'],
    var_name='sigma_type', 
    value_name='sigma_estimate'
)
df_melted = df_melted[~((df_melted['loss_func_disp'] == 'NN-noRE') & (df_melted['sigma_type'] == 'sigma2_b_est' ))]
#df_melted['sig_label'] = df_melted['sigma_type'] + '_' + df_melted['loss_func_disp']

sigma_type_map = {
    'sigma2_b_est': r'$\hat{\sigma}^2_b$',
    'sigma2_e_est': r'$\hat{\sigma}^2_\epsilon$'
}
df_melted['sigma_type_disp'] = df_melted['sigma_type'].map(sigma_type_map)
df_melted = df_melted.rename(
    columns={
    'loss_func_disp': 'Model', #RENN, RENN-cor, NN-noRE
    'sigma_type_disp': 'Parameter' #train val
    }
    )

sns.set(style='whitegrid')
#plt.figure(figsize=(12, 20))

g = sns.catplot(
    data = df_melted,
    x='m',
    y='sigma_estimate',
    hue='Model',
    col = 'Parameter',
    #col = 'loss_func_disp',
    #row='sigma_type',
    palette=palette,
    kind='box',
    #height=6,
    #width=0.7,  
    aspect=0.8,
    dodge=True,
    sharey=True,#'row',
    sharex=True
    )

#for ax in g.axes.flat:
#    ax.set_yscale('log')

g.set_axis_labels('Batch Size (m)', "Parameter Estimate")
g.set_titles("{col_name}")
g.fig.suptitle('Estimated Variances over Batch Size', y=1.03 )# (Best Epoch per Run over 10 Runs)

g.legend.set_title('Model')
sns.move_legend(g, 'upper center',
         bbox_to_anchor=(0.5 , 1.01),
         ncol=4,
         frameon=False )
                #'lower right', bbox_to_anchor=(1, 0.62))
#g.legend(title=None, 
#         fontsize='small' , 
#         title_fontsize='small',
#         loc='upper center',
#         bbox_to_anchor=(0.9 , 0.6),
#         ncol=3,
#         frameon=False
#         )

plt.tight_layout()
out_path = dir_exp / "plots_losses_and_params" / f"e02_04_HISTO_best_sigmas_m.png"
plt.savefig(out_path, dpi=300, bbox_inches='tight')
