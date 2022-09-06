'''

'''

#%%
# import sys
import os,glob
from statistics import mean
# import time
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
import math
# from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.plt_tools import (set_font_type, defaultPlotting, day_night_split)
from plot_functions.get_bout_kinetics import get_bout_kinetics
from plot_functions.get_bout_features import get_bout_features
from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)
from scipy.stats import ttest_rel
from scipy.stats import ttest_ind

set_font_type()
defaultPlotting()

# %%
data_list = ['otog','tan'] # all or specific data
which_zeitgeber = 'day'
folder_name = f'otog TAN kinetics'
folder_dir = get_figure_dir('Fig_4')
fig_dir = os.path.join(folder_dir, folder_name)
try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {folder_name}')
except:
    print('Notes: re-writing old figures')
    
spd_bins = np.arange(5,25,4)

df_features_combined = pd.DataFrame()
df_bySpd_combined = pd.DataFrame()
df_kinetics_combined = pd.DataFrame()
for pick_data in data_list:
    root, FRAME_RATE = get_data_dir(pick_data)
    all_kinetic_cond, kinetics_jackknife, kinetics_bySpd_jackknife, all_cond1, all_cond2 = get_bout_kinetics(root, FRAME_RATE, ztime=which_zeitgeber)
    all_feature_cond, _, _ = get_bout_features(root, FRAME_RATE, ztime=which_zeitgeber)
    all_cond1 = pick_data
    all_cond2.sort()
    kinetics_bySpd_jackknife['dpf'] = pick_data
    kinetics_jackknife['dpf'] = pick_data
    all_feature_cond['dpf'] = pick_data
    df_bySpd_combined = pd.concat([df_bySpd_combined,kinetics_bySpd_jackknife], ignore_index=True)
    df_kinetics_combined = pd.concat([df_kinetics_combined,kinetics_jackknife], ignore_index=True)
    df_features_combined = pd.concat([df_features_combined,all_feature_cond], ignore_index=True)

df_features_combined = df_features_combined.assign(
    speed_bins = pd.cut(df_features_combined['spd_peak'],bins=spd_bins,labels=np.arange(len(spd_bins)-1))
)
df_bySpd_combined.rename(columns={'dpf':'dataset'},inplace=True)
df_kinetics_combined.rename(columns={'dpf':'dataset'},inplace=True)
df_features_combined.rename(columns={'dpf':'dataset'},inplace=True)

# %%
sns.set_style("ticks")

# %%
# by speed bins
toplt = df_bySpd_combined
cat_cols = ['jackknife_group','condition','expNum','dataset','ztime']
# all_features = [c for c in toplt.columns if c not in cat_cols]
all_features = ['steering_gain','righting_gain']

for feature_toplt in (all_features):
    g = sns.relplot(
        data = toplt,
        col = 'dataset',
        hue = 'condition',
        x = 'average_speed',
        y = feature_toplt,
        kind = 'line',
        err_style='bars',
        ci = 95,
        height=3,
    )
    g.set_xlabels("Peak speed (mm/s)", clear_inner=False)
    g.set(xlim=(6, 20))
    filename = os.path.join(fig_dir,f"{feature_toplt}_bySpd.pdf")
    plt.savefig(filename,format='PDF')

# %% Compare by condition
toplt = df_kinetics_combined.reset_index(drop=True)
toplt['condition'] = toplt['condition'].map({'1ctrl':'1ctrl',
                                             '2cond':'2cond',
                                             'hets':'1ctrl',
                                             'otog':'2cond'})
cat_cols = ['jackknife_group','condition','expNum','dataset','ztime']
# all_features = [c for c in toplt.columns if c not in cat_cols]
all_features = ['steering_gain_jack','righting_gain_jack']

for feature_toplt in (all_features):
    g = sns.catplot(
        data = toplt,
        col = 'dataset',
        hue = 'condition',
        x = 'condition',
        y = feature_toplt,
        linestyles = '',
        kind = 'point',
        marker = True,
        aspect=.6,
        height=3,
    )
    g.map(sns.lineplot,'condition',feature_toplt,estimator=None,
      units='jackknife_group',
      data = toplt,
      sort=False,
      color='grey',
      alpha=0.2,)
    g.add_legend()

    sns.despine(offset=10, trim=False)
    filename = os.path.join(fig_dir,f"{feature_toplt}_compare.pdf")
    plt.savefig(filename,format='PDF')

# %%
# check speed distribution
toplt = df_features_combined
toplt['condition'] = toplt['condition'].map({'1ctrl':'1ctrl',
                                             '2cond':'2cond',
                                             'hets':'1ctrl',
                                             'otog':'2cond'})

# check speed
feature_to_plt = 'spd_peak'
upper = np.percentile(toplt[feature_to_plt], 99.5)
lower = np.percentile(toplt[feature_to_plt], 0.5)
g = sns.FacetGrid(data=toplt,
            col="dataset", 
            hue="condition",
            sharey =False,
            sharex =True,
            )
g.map(sns.histplot,feature_to_plt,bins = 10, 
                    element="poly",
                    #  kde=True, 
                    stat="density",
                    pthresh=0.05,
                    fill=False,
                    binrange=(lower,upper),)

g.add_legend()
sns.despine()
plt.savefig(fig_dir+f"/{feature_to_plt} distribution.pdf",format='PDF')# %%

# %%
mean_data = df_features_combined.reset_index(drop=True)
mean_data['condition'] = mean_data['condition'].map({'1ctrl':'1ctrl',
                                             '2cond':'2cond',
                                             'hets':'1ctrl',
                                             'otog':'2cond'})
mean_data = mean_data.groupby(['expNum','dataset','condition']).mean().reset_index()
feature_toplt = 'spd_peak'
toplt = mean_data

g = sns.catplot(
    data = toplt,
    col = 'dataset',
    # hue = 'condition',
    x = 'condition',
    y = feature_toplt,
    linestyles = '',
    kind = 'point',
    marker = True,
    aspect=.6,
    height=3,
)
(g.map(sns.lineplot,
      'condition',
      feature_toplt,
      units=toplt['expNum'],
      estimator=None,
    #   sort=False,
      color='grey',
      alpha=0.2,))
g.add_legend()
sns.despine(offset=10, trim=False)

g.set_ylabels("Peak speed (mm/s)", clear_inner=False)
plt.savefig(fig_dir+f"/{feature_to_plt} compare.pdf",format='PDF')# %%

# %%
multi_comp = MultiComparison(mean_data[feature_toplt], mean_data['dataset']+mean_data['condition'])
print('* attack angles')
print(multi_comp.tukeyhsd().summary())

# %%
for cond in set(mean_data['dataset']):
    this_df = mean_data.loc[mean_data['dataset']==cond]
    ttest_res, ttest_p = ttest_rel(this_df.loc[this_df['condition']=='1ctrl',feature_toplt],
                                    this_df.loc[this_df['condition']=='2cond',feature_toplt])
    print(cond)
    print(f' {ttest_p}')
# %%
