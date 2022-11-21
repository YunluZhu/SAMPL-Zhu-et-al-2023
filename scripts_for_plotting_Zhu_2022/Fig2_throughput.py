'''
Plot distribution of parameters in Figure 2 and 3.
Plot parameter correlations for calculating kinetics in Figure 3.
'''

#%%
# import sys
import os
from plot_functions.plt_tools import round_half_up
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.get_bout_features import get_bout_features
from plot_functions.get_bout_kinetics import get_kinetics
from plot_functions.get_IBIangles import get_IBIangles
from plot_functions.plt_tools import (jackknife_mean,set_font_type, defaultPlotting)
from tqdm import tqdm
import matplotlib as mpl

set_font_type()
mpl.rc('figure', max_open_warning = 0)

# %%
# Select data and create figure folder
pick_data = 'all_7dd'
which_ztime = 'day'
root, FRAME_RATE = get_data_dir(pick_data)

folder_name = f'{pick_data} throughput'
folder_dir2 = get_figure_dir('Fig_2')
fig_dir2 = os.path.join(folder_dir2, folder_name)
try:
    os.makedirs(fig_dir2)
except:
    pass

# %% get features
# all_feature_cond, all_cond1, all_cond2 = get_bout_features(root, FRAME_RATE, ztime=which_ztime)
# all_ibi_cond, _, _  = get_IBIangles(root, FRAME_RATE, ztime=which_ztime)
# %% get total box number
folder_dir_list = []
for folder in os.listdir(root):
    if folder.startswith("."):
        pass
    else:
        folder_dir_list.append(os.path.join(root,folder))

metadata_list = []  
menadata_name_list = []
for folder_dir in folder_dir_list:
    for item in os.listdir(folder_dir):
        if item.endswith("metadata.csv"):
            metadata_list.append(os.path.join(folder_dir,item))
            menadata_name_list.append(item)
     
all_metadata = pd.DataFrame()       
for i, metadata_file in enumerate(metadata_list):
    df = pd.read_csv(metadata_file, index_col=0)
    df = df.assign(
        expNum = menadata_name_list[i][0:-13]
    )
    all_metadata = pd.concat([all_metadata,df],ignore_index=True)
    
all_metadata = all_metadata.assign(
    date = [int(filename[0:6]) for filename in all_metadata["filename"]]
) 

bout_number24h = all_metadata.groupby(['date','expNum','box_number']).sum()

bout_by_size = bout_number24h.assign(
    cuvette = pd.cut(bout_number24h['num_fish'],bins=[0,3.1,10],labels=['narrow','standard'])
)

toplt = bout_number24h
feature = 'aligned_bout'
upper = np.percentile(toplt[feature], 99.5)
lower = np.percentile(toplt[feature], 0.5)

plt.figure(figsize=(3,2))
g = sns.histplot(data=toplt, x=feature, 
                    bins = 10, 
                    element="poly",
                    #  kde=True, 
                    stat="probability",
                    pthresh=0.05,
                    binrange=(lower,upper),
                    color='grey'
                    )
g.set_xlabel('bout number/box/24hrs')
sns.despine()
plt.savefig(fig_dir2+f"/{feature} distribution.pdf",format='PDF')
print(f"bouts per box per 24hrs = {toplt[feature].mean():.2f} ± {toplt[feature].std():.2f} (mean ± STD)")
# plt.close()
# %%
bout_by_size = bout_by_size.reset_index(drop=True)

plt.figure(figsize=(3,2))
g = sns.pointplot(data=bout_by_size, y=feature, x='cuvette')
g.set_xlabel('bout number by cuvette/box/24hrs')
sns.despine()
plt.savefig(fig_dir2+f"/{feature} distribution.pdf",format='PDF')

n_mean, s_mean = bout_by_size.groupby('cuvette').mean()['aligned_bout'].values
n_std, s_std = bout_by_size.groupby('cuvette').std()['aligned_bout'].values
print(f"bouts per box per 24hrs (narrow cuvettes) = {n_mean:.2f} ± {n_std:.2f} (mean ± STD)")
print(f"bouts per box per 24hrs (std cuvettes) = {s_mean:.2f} ± {s_std:.2f} (mean ± STD)")

# %%
