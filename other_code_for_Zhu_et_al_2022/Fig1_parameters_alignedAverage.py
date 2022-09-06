'''
plots parameters of bouts aligned at the time of the peak speed.
Input directory needs to be a folder containing analyzed dlm data.
'''

#%%
from cmath import exp
import os
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.plt_tools import (set_font_type, day_night_split)
from plot_functions.get_index import (get_index)
from plot_functions.get_bout_features import get_bout_features
from plot_functions.get_bout_kinetics import get_kinetics

from tqdm import tqdm

# %%

print('- Figure 1: Bout parameter time series')

# choose the time duration to plot. 
# total aligned duration = [-0.5, 0.4] (sec) around time of peak speed
# [-0.3,0.2] (sec) around peak speed is recommended 

BEFORE_PEAK = 0.3 # s
AFTER_PEAK = 0.2 #s

# %% features for plotting
# below are all the properties can be plotted. 
all_features = {
    'propBoutAligned_speed':'speed (mm*s-1)', 
    'propBoutAligned_linearAccel':'linear accel (mm*s-2)',
    'propBoutAligned_pitch':'pitch (deg)', 
    'propBoutAligned_angVel':'ang vel (deg*s-1)',   # smoothed angular velocity
    # 'propBoutAligned_accel':'ang accel (deg*s-2)',    # angular accel calculated using raw angular vel
    # 'propBoutInflAligned_accel',
    # 'propBoutAligned_instHeading', 
    # 'propBoutAligned_x':'x position (mm)',
    # 'propBoutAligned_y':'y position (mm)', 
    # 'propBoutInflAligned_angVel',
    # 'propBoutInflAligned_speed', 
    # 'propBoutAligned_angVel_hDn',
    # # 'propBoutAligned_speed_hDn', 
    # 'propBoutAligned_pitch_hDn',
    # # 'propBoutAligned_angVel_flat', 
    # # 'propBoutAligned_speed_flat',
    # # 'propBoutAligned_pitch_flat', 
    # 'propBoutAligned_angVel_hUp',
    # 'propBoutAligned_speed_hUp', 
    # 'propBoutAligned_pitch_hUp', 
}
# %%
# Select data and create figure folder
pick_data = 'wt_7dd'
which_ztime = 'day'
root, FRAME_RATE = get_data_dir(pick_data)

folder_name = f'{pick_data} parameter time series'
folder_dir1 = get_figure_dir('Fig_1')
fig_dir1 = os.path.join(folder_dir1, folder_name)

try:
    os.makedirs(fig_dir1)
except:
    pass

# %%

# get the index for the time of peak speed, and total time points for each aligned bout
peak_idx, total_aligned = get_index(FRAME_RATE)
all_conditions = []
folder_paths = []
for folder in os.listdir(root):
    if folder[0] != '.':
        folder_paths.append(root+'/'+folder)
        all_conditions.append(folder)
# calculate indicies
idxRANGE = [peak_idx-int(BEFORE_PEAK*FRAME_RATE),peak_idx+int(AFTER_PEAK*FRAME_RATE)]

for condition_idx, folder in enumerate(folder_paths):
    # enter each condition folder (e.g. 7dd_ctrl)
    for subpath, subdir_list, subfile_list in os.walk(folder):
        # if folder is not empty
        if subdir_list:
            # reset for each condition
            exp_data_all = pd.DataFrame()
            # loop through each sub-folder (experiment) under each condition
            for expNum, exp in enumerate(subdir_list):
                rows = []
                exp_path = os.path.join(subpath, exp)
                # get pitch                
                raw = pd.read_hdf(f"{exp_path}/bout_data.h5", key='prop_bout_aligned')
                exp_data = raw.assign(
                    propBoutAligned_linearAccel = raw['propBoutAligned_speed'].diff()
                )
                exp_data = exp_data.loc[:,all_features.keys()]
                exp_data = exp_data.rename(columns=all_features)
                # assign frame number, total_aligned frames per bout
                exp_data = exp_data.assign(
                    idx = int(len(exp_data)/total_aligned)*list(range(0,total_aligned)),
                    )

                # - get the index of the rows in exp_data to keep
                bout_time = pd.read_hdf(f"{exp_path}/bout_data.h5", key='prop_bout2').loc[:,['aligned_time']]
                # for i in bout_time.index:
                # # if only need day or night bouts:
                for i in day_night_split(bout_time,'aligned_time').index:
                    rows.extend(list(range(i*total_aligned+idxRANGE[0],i*total_aligned+idxRANGE[1])))

                exp_data = exp_data.assign(time_ms = (exp_data['idx']-peak_idx)/FRAME_RATE*1000)
                exp_data_all = pd.concat([exp_data_all,exp_data.loc[rows,:]])
            exp_data_all = exp_data_all.reset_index(drop=True)

# %%
# get bout features
all_feature_cond, all_cond1, all_cond2 = get_bout_features(root, FRAME_RATE, ztime=which_ztime)
one_kinetics = all_feature_cond.groupby(['dpf']).apply(
                        lambda x: get_kinetics(x)
                        ).reset_index()
# assign up and down
set_point = one_kinetics.loc[0,'set_point']
# %%
peak_speed = exp_data_all.loc[exp_data_all.idx==peak_idx,'speed (mm*s-1)']
pitch_pre_bout = exp_data_all.loc[exp_data_all.idx==int(peak_idx - 0.1 * FRAME_RATE),'pitch (deg)']

grp = exp_data_all.groupby(np.arange(len(exp_data_all))//(idxRANGE[1]-idxRANGE[0]))
exp_data_all = exp_data_all.assign(
                                    peak_speed = np.repeat(peak_speed,(idxRANGE[1]-idxRANGE[0])).values,
                                    pitch_pre_bout = np.repeat(pitch_pre_bout,(idxRANGE[1]-idxRANGE[0])).values,
                                    bout_number = grp.ngroup(),
                                )
exp_data_all = exp_data_all.assign(
                                    direction = pd.cut(exp_data_all['pitch_pre_bout'],[-90,set_point,90],labels = ['Nose-down', 'Nose-up'])
                                )
# %%

set_font_type()
for feature_toplt in tqdm(list(all_features.values())):
    p = sns.relplot(
            data = exp_data_all, x = 'time_ms', y = feature_toplt,
            hue='direction',
            kind = 'line',aspect=3, height=2, ci=95
            )
    p.map(
        plt.axvline, x=0, linewidth=1, color=".3", zorder=0
        )
    plt.savefig(os.path.join(fig_dir1, f"{feature_toplt}_timeSeries_up_dn.pdf"),format='PDF')

for feature_toplt in tqdm(list(all_features.values())):
    p = sns.relplot(
            data = exp_data_all, x = 'time_ms', y = feature_toplt,
            kind = 'line',aspect=3, height=2, ci=95
            )
    p.map(
        plt.axvline, x=0, linewidth=1, color=".3", zorder=0
        )
    plt.savefig(os.path.join(fig_dir1, f"{feature_toplt}_timeSeries.pdf"),format='PDF')

# %%
