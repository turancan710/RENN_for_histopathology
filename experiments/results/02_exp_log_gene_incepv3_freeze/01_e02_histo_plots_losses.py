#import os
import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker

dir_exp = Path(__file__).resolve().parents[0]
csv_path = dir_exp / 'results_flat.csv'
df = pd.read_csv(csv_path)

palette = {
    'RENN': '#1f77b4',  # blue
    'RENN-cor':'#ff7f0e',  # orange
    'NN-noRE': '#2ca02c',    #green
    'RENN-det': "#9c00aa" #pink
}

df = df[df['epoch']>0]
#long dataset (val loss & train loss -> separate row each), make 4 labels as above
df_melted = df.melt(
    id_vars=['loss_func_disp', 'm', 'run', 'epoch'],
    value_vars=['train_loss' , 'val_loss'],
    var_name='loss_type', 
    value_name='loss_value'
)

#df_melted['loss_label'] = df_melted['loss_func_disp'] + '_' + df_melted['loss_type'].str.replace('_loss', '')
df_melted['loss_type_disp'] = df_melted['loss_type'].str.replace('_loss', '') #i.e. train , val
df_melted = df_melted.rename(
    columns={
    'loss_func_disp': 'Model', #RENN, RENN-cor, NN-noRE
    'loss_type_disp': 'Loss Type' #train val
    }
    )
#subset = df_melted.query('m==0 and loss_label == "NLL_cor_1_train" ')
#print(subset.groupby(['epoch']).loss_value.describe())


#sns.set(style='whitegrid')
sns.set(style='whitegrid',
        context="notebook", 
        font_scale=1.7) 

g = sns.relplot(
    data = df_melted,
    x='epoch',
    y='loss_value',
    hue='Model',
    style='Loss Type',
    col='m',
    palette=palette,
    kind='line',
    col_wrap=3,
    #height=3.5,
    estimator='mean',
    errorbar=None ,
    facet_kws={'sharey': False}, #dyn y axis
    linewidth=2
    )

#g.set(yscale='log')
'''
for ax in g.axes.flat:
    ax.set_yscale('log')
    ticks = np.arange(1.0 , 5.1 , 0.5) # np.concatenate( np.arange(1.0 , 2.0 , 0.1) , np.arange(3.0 , 5.0 , 0.5)) 
    ax.yaxis.set_major_locator(mticker.FixedLocator(ticks))
    ax.tick_params(axis='y', labelsize=7.5 )#, labelrotation=45)
'''


g.set_axis_labels('Epoch', 'Loss')
g.set_titles('m = {col_name}')
g.fig.suptitle('Training & Validation Loss across Batch Size ', y=1.03)#(10 Runs Mean)

sns.move_legend(g, 'lower center', title=None,  
             bbox_to_anchor=(0.5, .93),
             ncol=9,
             frameon=False
             )
#g.legend.set_title(None)

'''
ax_full_gr = [ g.axes.flat[0] , g.axes.flat[5] ]
for ax in ax_full_gr:
    ax.text(0.95, 0.95, "NLL = NLL_cor_1", transform=ax.transAxes, ha="right", va="top", fontsize=8, color='gray')
'''

plt.tight_layout()
out_path = dir_exp / "plots_losses_and_params" / f"e02_01_HISTO_loss_epoch_grid_by_m.png"
plt.savefig(out_path, dpi=300, bbox_inches='tight')
