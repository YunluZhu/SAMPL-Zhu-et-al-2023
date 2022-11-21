'''
Plot jackknifed kinetics

righting gain
set point
steering gain
correlation of accel & decel rotation
angvel gain (new)

zeitgeber time? Yes
jackknife? Yes
resampled? No
'''

#%%
# import sys
from plot_functions.plt_tools import round_half_up
import os,glob
# import time
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
import math
# from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.plt_tools import (set_font_type, defaultPlotting, day_night_split)
from plot_functions.get_index import get_index
from plot_functions.get_bout_kinetics import get_bout_kinetics
set_font_type()
defaultPlotting()
# %%
pick_data = '7dd_bkg' # all or specific data
# for day night split
which_zeitgeber = 'day' # day night all
SAMPLE_NUM = 1000
# %%
# def main(pick_data):
root, FRAME_RATE = get_data_dir(pick_data)
peak_idx , total_aligned = get_index(FRAME_RATE)
# TSP_THRESHOLD = [-np.Inf,-50,50,np.Inf]
# spd_bins = np.arange(3,24,3)

print("- Figure 7: ZF strains - Steering & Righting")

root, FRAME_RATE = get_data_dir(pick_data)

folder_name = f'{pick_data} Steering and Righting'
folder_dir = get_figure_dir('Fig_7')
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created:{folder_name}')
except:
    print('fig folder already exist')


# %%
all_kinetic_cond, kinetics_jackknife, kinetics_bySpd_jackknife, all_cond1, all_cond2 = get_bout_kinetics(root, FRAME_RATE, ztime=which_zeitgeber, sample=SAMPLE_NUM)
all_cond1.sort()
all_cond2.sort()

# %% Compare by condition
toplt = kinetics_jackknife
cat_cols = ['jackknife_group','condition','expNum','dpf','ztime']
all_features = [c for c in toplt.columns if c not in cat_cols]
# print('plot jackknife data')

for feature_toplt in (all_features):
    p = sns.catplot(
        data = toplt, y=feature_toplt,x='condition',kind='strip',
        color = 'grey',
        edgecolor = None,
        linewidth = 0,
        s=8, 
        alpha=0.3,
        height=3,
        aspect=1,
        zorder=1,
    )
    p.map(sns.pointplot,'condition',feature_toplt,
        markers=['d','d','d'],
        order=all_cond2,
        join=False, 
        ci=None,
        color='black',
        zorder=100,
        data=toplt)
    filename = os.path.join(fig_dir,f"{feature_toplt} .pdf")
    plt.savefig(filename,format='PDF')
    plt.show()
# %%
