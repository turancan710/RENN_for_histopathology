import os
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

#select per (loss x m x run) -> best epoch (min Val loss)
df_best_epochs= ( df.loc[df.groupby(['loss_func_disp' , 'm', 'run'])['val_loss'].idxmin()] )


#long dataset (val loss & train loss -> separate row each), make 4 labels as above
df_melted = df_best_epochs.melt(
    id_vars=['loss_func_disp', 'm', 'run' ],
    value_vars=['train_loss' , 'val_loss'],
    var_name='loss_type', 
    value_name='loss_value'
)

#df_melted['loss_label'] = df_melted['loss_func_disp'] + '_' + df_melted['loss_type'].str.replace('_loss', '')
df_melted['loss_type_disp'] = df_melted['loss_type'].str.replace('_loss', '') #i.e. train , val
df_melted['m_cat'] = df_melted['m'].astype(str)

df_melted = df_melted.rename(
    columns={
    'loss_func_disp': 'Model', #RENN, RENN-cor, NN-noRE
    'loss_type_disp': 'Loss Type' #train val
    }
    )

sns.set(style='whitegrid')

g = sns.lineplot(
    data = df_melted,
    x='m_cat',
    y='loss_value',
    hue='Model', #loss_label
    style='Loss Type',
    estimator='mean',
    errorbar=None ,
    palette=palette,
    marker = 'o'
    )

#g.set(yscale='log')
g.set_title('Training & Validation Loss over Batch Size') #(Best Epoch per Run - 10 Runs Average)
g.set_xlabel('Batch Size (m)')
g.set_ylabel('Loss') # (Log Scale)

g.legend(title=None, 
         fontsize='small' , 
         title_fontsize='small',
         loc='upper center',
         bbox_to_anchor=(0.1 , 1.02),
         ncol=1,
         frameon=False
         ) #'Model (Loss Type)'
for text in g.legend_.get_texts():
    text.set_fontsize(8)
#for handle in g.legend_.legend_handles:
#    handle.set_markersize(5)  # smaller marker size
#    handle.set_linewidth(4)

plt.tight_layout()
out_path = dir_exp / "plots_losses_and_params" / f"e02_03_HITSO_best_losses_m.png"
plt.savefig(out_path, dpi=300, bbox_inches='tight')
