'''
Plot distribution of bout parameters. Plot 2D distribution of parameters for kinetics calculation.

This script takes two types of directory:
1. input directory can be a folder containing analyzed dlm data...
    dir/
    ├── all_data.h5
    ├── bout_data.h5
    └── IEI_data.h5
2. or a folder with subfolders containing dlm data
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

#%%
import os
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from plot_functions.plt_tools import round_half_up
from plot_functions.plt_tools import (set_font_type, defaultPlotting, day_night_split)
from plot_functions.plt_v4 import (jackknife_kinetics, extract_bout_features_v4, get_kinetics)
from plot_functions.get_index import (get_index, get_frame_rate)
from plot_functions.plt_tools import (set_font_type, day_night_split)

# %%
def plot_save_histogram(toplt,feature_toplt,xlabel,fig_dir):
    upper = np.nanpercentile(toplt[feature_toplt], 99.5)
    lower = np.nanpercentile(toplt[feature_toplt], 0.5)
    g = sns.histplot(data=toplt, x=feature_toplt, 
                        bins = 20, 
                        element="poly",
                        #  kde=True, 
                        stat="probability",
                        pthresh=0.05,
                        binrange=(lower,upper)
                        )
    g.set_xlabel(xlabel)
    sns.despine()
    filename = os.path.join(fig_dir,f"{feature_toplt}.pdf")
    plt.savefig(filename,format='PDF')
    plt.close()
        
def plot_parameters(root):
    """Plot distribution of bout parameters. Plot 2D distribution of parameters for kinetics calculation.


    Args:
        root (string): directory
    """
    print('\n- Plotting bout parameters')
    # generate figure folder
    folder_name = 'parameters'
    folder_dir = os.getcwd()
    fig_dir = os.path.join(folder_dir, 'figures', folder_name)

    try:
        os.makedirs(fig_dir)
        print(f'fig directory created: {fig_dir}')
    except:
        print('Figure folder already exist! Old figures will be replaced.')
        print(fig_dir)

    # Check if there are subfolders in the current directory
    all_dir = [ele[0] for ele in os.walk(root)]
    if len(all_dir) > 1:
        # if yes, calculate jackknifed std(pitch)
        if_jackknife = True
        all_dir = all_dir[1:]
    else:
        # if no, only calculate one std(pitch) for current experiment
        if_jackknife = False
        
    # get frame rate
    try:
        FRAME_RATE = get_frame_rate(all_dir[0])
    except:
        print("No info file found!\n")
        FRAME_RATE = round_half_up(input("Frame rate? "))

    # get the index for the time of peak speed, and total time points for each aligned bout
    peak_idx, total_aligned = get_index(FRAME_RATE)

    # %%
    # defaultPlotting()
    T_start = -0.3
    T_end = 0.3
    idx_start = round_half_up(peak_idx + T_start * FRAME_RATE)
    idx_end = round_half_up(peak_idx + T_end * FRAME_RATE)
    idxRANGE = [idx_start,idx_end]

    # %%
    # main function

    bout_features = pd.DataFrame()
    all_IBI_data = pd.DataFrame()
    # bout_kinetics = pd.DataFrame()

    # go through each condition folders under the root
    for expNum, exp in enumerate(all_dir):
        # angular velocity (angVel) calculation
        rows = []
        # for each sub-folder, get the path
        exp_path = exp
        # get pitch                
        exp_data = pd.read_hdf(f"{exp_path}/bout_data.h5", key='prop_bout_aligned')
        # assign frame number, total_aligned frames per bout
        exp_data = exp_data.assign(idx=round_half_up(len(exp_data)/total_aligned)*list(range(0,total_aligned)))
        
        # - get the index of the rows in exp_data to keep (for each bout, there are range(0:51) frames. keep range(20:41) frames)
        bout_time = pd.read_hdf(f"{exp_path}/bout_data.h5", key='prop_bout2').loc[:,['aligned_time']]
        # # if only need day or night bouts:
        for i in day_night_split(bout_time,'aligned_time').index:
            rows.extend(list(range(i*total_aligned+round_half_up(idxRANGE[0]),i*total_aligned+round_half_up(idxRANGE[1]))))
        exp_data = exp_data.assign(expNum = exp)
        trunc_day_exp_data = exp_data.loc[rows,:]
        trunc_day_exp_data = trunc_day_exp_data.assign(
            bout_num = trunc_day_exp_data.groupby(np.arange(len(trunc_day_exp_data))//(idxRANGE[1]-idxRANGE[0])).ngroup()
            )
        num_of_bouts = len(trunc_day_exp_data.loc[trunc_day_exp_data['idx'] == peak_idx])
        
        this_exp_features = extract_bout_features_v4(trunc_day_exp_data,peak_idx,FRAME_RATE)
        this_exp_features = this_exp_features.assign(
            exp_num = [expNum]*num_of_bouts,
            )
        this_exp_features = this_exp_features.reset_index(drop=True)
        
        # this_exp_kinetics = get_kinetics(this_exp_features)
        # this_exp_kinetics = this_exp_kinetics.append(pd.Series(data={'exp_num': expNum}))
        
        bout_features = pd.concat([bout_features,this_exp_features])
        # bout_kinetics = pd.concat([bout_kinetics,this_exp_kinetics.to_frame().T], ignore_index=True)
        
        # next, read inter bout interval data
        IBI_data = pd.read_hdf(f"{exp_path}/IEI_data.h5", key='prop_bout_IEI2')               
        IBI_data = IBI_data.loc[:,['propBoutIEI', 'propBoutIEI_pitch', 'propBoutIEItime']]
        IBI_angles = day_night_split(IBI_data,'propBoutIEItime').assign(exp_num=expNum)
        IBI_angles.dropna(inplace=True)
        all_IBI_data = pd.concat([all_IBI_data, IBI_angles[['propBoutIEI', 'propBoutIEI_pitch']]],ignore_index=True)
        
    all_IBI_data.columns = ['IBI','pitch_IBI']
    all_IBI_data = all_IBI_data.reset_index(drop=True)

    defaultPlotting()
    set_font_type()
    sns.set_style("ticks")
    
    bout_features = bout_features.reset_index(drop=True)
    toplt = bout_features
    cat_cols = ['exp_num']


    # plot 1d probability/histogram
    all_features = [c for c in toplt.columns if c not in cat_cols]
    
    mean_val = bout_features[all_features].mean()
    filename = os.path.join(fig_dir,f"parameter mean values.csv")
    mean_val.to_csv(filename)
    print(mean_val)
    
    for feature_toplt in (all_features):
        # let's add unit
        if 'spd' in feature_toplt:
            xlabel = feature_toplt + " (mm*s^-1)"
        elif 'dis' in feature_toplt:
            xlabel = feature_toplt + " (mm)"
        else:
            xlabel = feature_toplt + " (deg)"
        plot_save_histogram(toplt,feature_toplt,xlabel,fig_dir)
    
    toplt = all_IBI_data
    cat_cols = ['exp_num']
    all_features = [c for c in toplt.columns if c not in cat_cols]
    for feature_toplt in (all_features):
        # let's add unit
        if 'pitch' in feature_toplt:
            xlabel = feature_toplt + " (deg)"
        else:
            xlabel = feature_toplt + " (s)"
        plot_save_histogram(toplt,feature_toplt,xlabel,fig_dir)
    
    # %%
    # Plots relational plots and 2D distribution of parameters used to calculate kinetics
    # righting gain
    toplt = bout_features
    g = sns.displot(data=toplt, y='rot_l_decel',x='pitch_pre_bout')
    plt.xlabel('pre-bout pitch (deg)')
    plt.ylabel('deceleration rot (deg)')
    sns.despine()
    filename = os.path.join(fig_dir,f"Righting_decelRot-posture.pdf")
    plt.savefig(filename,format='PDF')
    plt.close()    
    
    # regression
    r = sns.regplot(data=toplt, y='rot_l_decel',x='pitch_pre_bout',x_bins=6)
    r.set_xlabel('pre-bout pitch (deg)')
    r.set_ylabel('deceleration rot (deg)')
    filename = os.path.join(fig_dir,f"Righting_regression.pdf")
    plt.savefig(filename,format='PDF')
    plt.close()   
    
    # steering gain
    g = sns.displot(data=toplt, y='traj_peak',x='pitch_peak')
    plt.xlabel('pitch at 0ms (deg)')
    plt.ylabel('traj at 0ms (deg)')
    sns.despine()
    filename = os.path.join(fig_dir,f"Steering_traj-posture.pdf")
    plt.savefig(filename,format='PDF')
    plt.close()        
    
    r = sns.regplot(data=toplt, y='traj_peak',x='pitch_peak',x_bins=6)
    r.set_xlabel('pitch at 0ms (deg)')
    r.set_ylabel('traj at 0ms (deg)')
    filename = os.path.join(fig_dir,f"Steering_regression.pdf")
    plt.savefig(filename,format='PDF')
    plt.close()   
    
    # set point
    g = sns.displot(data=toplt, y='rot_l_decel',x='pitch_pre_bout')
    r.set_xlabel('pre-bout pitch (deg)')
    r.set_ylabel('deceleration rot (deg)')
    sns.despine()
    filename = os.path.join(fig_dir,f"SetPoint_posture-rot.pdf")
    plt.savefig(filename,format='PDF')
    plt.close()     

    r = sns.regplot(data=toplt, y='rot_l_decel',x='pitch_pre_bout',x_bins=6)
    r.set_xlabel('pre-bout pitch (deg)')
    r.set_ylabel('deceleration rot (deg)')
    filename = os.path.join(fig_dir,f"Set point_regression.pdf")
    plt.savefig(filename,format='PDF')
    plt.close()  
    
    
# %%
if __name__ == "__main__":
    # if to use Command Line Inputs
    root = input("- Data directory? \n")
    plot_parameters(root)
