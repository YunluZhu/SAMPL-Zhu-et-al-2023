'''
Plot basic parameters as a function of time. Modify "all_features" to choose parameters to plot.
This script contains two functions:
    plot_aligned
    plot_raw

plt_timeseries.plot_aligned(dir)
    plots parameters of aligned bouts as a function of time.

        
plt_timeseries.plot_raw(dir) 
    plots raw data from a single epoch that contains one or more bouts as a function of time


This script takes two types of data structures:
1. Input directory being a folder containing analyzed dlm data
    dir/
    ├── all_data.h5
    ├── bout_data.h5
    └── IEI_data.h5
2. Input directory being a folder with subfolders containing dlm data
    dir/
    ├── experiment 1/  
    │   ├── all_data.h5
    │   ├── bout_data.h5
    │   └── IEI_data.h5
    └── experiment 2/
        ├── all_data.h5
        ├── bout_data.h5
        └── IEI_data.h5
'''

# %%
import os
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from plot_functions.plt_tools import round_half_up
from plot_functions.plt_tools import (set_font_type)
from plot_functions.get_data_dir import (get_figure_dir)

from tqdm import tqdm

def Fig2_single_epoch(root):
    # %% features for plotting
    # below are all the properties can be plotted. 
    all_features = {
        'ang':'pitch (deg)',
        # 'absy':'y position (mm)'
        # 'deltaT', 
        'x':'x',
        'y':'y',
        'headx':'head x (mm)',
        'heady':'head y (mm)',
        # 'centeredAng':'centered angle (deg)',
        # 'xvel', 
        # 'yvel', 
        'dist':'distance (mm)',
        # 'displ':'displacement (mm)',
        'angVel':'ang vel (deg*s-1)',
        # 'angVelSmoothed', 
        # 'angAccel':'ang accel (deg*s-2)',
        'swimSpeed':'speed (mm*s-1)',
        'velocity':'velocity (mm*s-1)'
    }
    # %%
    # generate figure folder

    folder_name = f'Single epoch speed and pitch'
    folder_dir2 = get_figure_dir('Fig_2')
    fig_dir2 = os.path.join(folder_dir2, folder_name)

    try:
        os.makedirs(fig_dir2)
    except:
        pass

    # %%
    # for each sub-folder, get the path
    all_dir = [ele[0] for ele in os.walk(root)]
    if len(all_dir) > 1:
        all_dir = all_dir[1:]
        
    # get frame rate
    FRAME_RATE = 166

    epoch_info_all = pd.DataFrame()
    epoch_data_all = pd.DataFrame()
    for expNum, exp_path in enumerate(all_dir):
        # get pitch                
        all_data = pd.read_hdf(f"{exp_path}/all_data.h5", key='grabbed_all')

        exp_data = all_data.loc[:,all_features.keys()]
        exp_data = exp_data.rename(columns=all_features)
        exp_data = exp_data.assign(
            expNum = expNum,
            epochNum = all_data['epochNum'].values,
            deltaT = all_data['deltaT'].values
        )
        
        epoch_info = exp_data.groupby('epochNum').size().reset_index()

        epoch_info = epoch_info.rename(columns={
            'epochNum':'epoch_num',
            0:'frame_num',
        })
        epoch_info.reset_index(drop=True)
        epoch_info = epoch_info.assign(
            idx = np.arange(0,len(epoch_info))+1,
            duration = epoch_info['frame_num']/FRAME_RATE,
            expNum = expNum,
        )
        epoch_info_all = pd.concat([epoch_info_all,epoch_info], ignore_index=True)
        epoch_data_all = pd.concat([epoch_data_all,exp_data], ignore_index=True)
        
    epoch_info_all = epoch_info_all.sort_values(by='duration',ascending=False)
    epoch_info_all = epoch_info_all.reset_index(drop=True)
    # %%
    which_toplt = 179
    toplt = epoch_info_all.loc[epoch_info_all['epoch_num']==which_toplt].squeeze()
    data_toplt = epoch_data_all.loc[(epoch_data_all['expNum']==toplt['expNum']) & (epoch_data_all['epochNum']==int(toplt['epoch_num'])), :]

    start_i = 48
    data_toplt = data_toplt.iloc[start_i:start_i+280]
    data_toplt = data_toplt.assign(
        time_s = np.cumsum(data_toplt['deltaT'])
    )

    set_font_type()
    features_toplot = ['speed (mm*s-1)', 'pitch (deg)']
    for feature_toplt in features_toplot:
        p = sns.relplot(
            data = data_toplt, x = 'time_s', y = feature_toplt,
            kind = 'line',aspect=3, height=2
            )
        plt.savefig(os.path.join(fig_dir2, f"{feature_toplt}_raw.pdf"),format='PDF')


# %%
if __name__ == "__main__":
    # if to use Command Line Inputs
    data_dir = input("- Data directory: where is data folder 'behavior data'? \n")
    root = os.path.join(data_dir,'movie_data_singleEpoch')
    Fig2_single_epoch(root)