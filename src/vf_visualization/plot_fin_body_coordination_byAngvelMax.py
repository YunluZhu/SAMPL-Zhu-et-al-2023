'''
Attack angles, rotation, and fin-body ratio
https://elifesciences.org/articles/45839

This script calculates attack angles and rotation and generates the fin-body ratio using a sigmoid fit

This script takes two types of directory:
1. A directory with analyzed dlm data
    dir/
    ├── all_data.h5
    ├── bout_data.h5
    └── IEI_data.h5
2. A directory with subfolders containing dlm data. Jackknife resampling will be applied to calculate errors for coef and sigmoid fit. 
    dir/
    ├── experiment 1/  
    │   ├── all_data.h5
    │   ├── bout_data.h5
    │   └── IEI_data.h5
    └── experiment 2/
        ├── all_data.h5
        ├── bout_data.h5
        └── IEI_data.h5
        
NOTE
Reliable sigmoid regression requires as many bouts as possible. > 6000 bouts is recommended.
User may define the number of bouts sampled from each experimental repeat for jackknifing by defining the argument "sample_bout"
Default is off (sample_bout = -1)
'''

#%%
import os
import pandas as pd # pandas library
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from astropy.stats import jackknife_resampling
from scipy.optimize import curve_fit
from plot_functions.get_index import (get_index, get_frame_rate)
from plot_functions.plt_tools import (set_font_type, day_night_split)
from plot_functions.plt_tools import round_half_up
from plot_functions.plt_v4 import (extract_bout_features_v4)
from scipy.signal import savgol_filter


# %%

def sigmoid_fit(df, x_range_to_fit,func,**kwargs):
    lower_bounds = [0.1,0,-100,1]
    upper_bounds = [10,20,2,100]
    x0=[5, 1, 0, 5]
    p0 = tuple(x0)
    popt, pcov = curve_fit(func, df['rot_to_max_angvel'], df['atk_ang'], 
                        #    maxfev=2000, 
                           p0 = p0,
                           bounds=(lower_bounds,upper_bounds))
    y = func(x_range_to_fit,*popt)
    output_coef = pd.DataFrame(data=popt).transpose()
    output_fitted = pd.DataFrame(data=y).assign(x=x_range_to_fit)
    p_sigma = np.sqrt(np.diag(pcov))
    return output_coef, output_fitted, p_sigma

def sigfunc_4free(x, a, b, c, d):
    y = c + (d)/(1 + np.exp(-(a*(x + b))))
    return y

def distribution_binned_average(df, by_col, bin_col, bin):
    df = df.sort_values(by=by_col)
    bins = pd.cut(df[by_col], list(bin))
    grp = df.groupby(bins)
    df_out = grp[[by_col,bin_col]].mean()
    return df_out


# %%
def plot_fin_body_coordination_byAngvelMax(root, **kwargs):
    """plot fin-body ratio and atk angle vs body rotation with sigmoid fit
    body rotation is calculated as rotation from initial to time of max angvel

    Args:
        root (string): directory
        sample_bout (int): number of bouts to sample from each experimental repeat. default is off (sample_bout = -1)
    """

    if_sample = False
    SAMPLE_N = -1

    for key, value in kwargs.items():
        if key == 'sample_bout':
            SAMPLE_N = value
            
    if SAMPLE_N == -1:
        SAMPLE_N = round_half_up(input("How many bouts to sample from each dataset? ('0' for no sampling): "))
    if SAMPLE_N > 0:
        if_sample = True
    folder_name = 'atk_ang fin_body_ratio rot_by_angvelMax'
    folder_dir = os.getcwd()
    fig_dir = os.path.join(folder_dir, 'figures', folder_name)
    
    print('\n- Plotting atk angle and fin-body ratio (angvel max)')

    try:
        os.makedirs(fig_dir)
        print(f'fig directory created: {fig_dir}')
    except:
        print('Figure folder already exist! Old figures will be replaced:')
        print(fig_dir)

    # %%
    # main function
    # CONSTANTS
    X_RANGE = np.arange(-2,10.01,0.01)
    BIN_WIDTH = 0.8
    AVERAGE_BIN = np.arange(min(X_RANGE),max(X_RANGE),BIN_WIDTH)

    T_start = -0.3
    T_end = 0.2

    # for each sub-folder, get the path
    all_dir = [ele[0] for ele in os.walk(root)]
    if len(all_dir) > 1:
        # if yes, calculate jackknifed std(pitch)
        all_dir = all_dir[1:]
        if_jackknife = True
    else:
        if_jackknife = False
        
    # get frame rate
    try:
        FRAME_RATE = get_frame_rate(all_dir[0])
    except:
        print("No metadata file found!\n")
        FRAME_RATE = round_half_up(input("Frame rate? "))

    peak_idx, total_aligned = get_index(FRAME_RATE)
    idx_start = round_half_up(peak_idx + T_start * FRAME_RATE)
    idx_end = round_half_up(peak_idx + T_end * FRAME_RATE)

    T_INITIAL = -0.25 #s
    T_PRE_BOUT = -0.10 #s
    T_END = 0.2
    T_MID_ACCEL = -0.05

    idx_initial = round_half_up(peak_idx + T_INITIAL * FRAME_RATE)
    idx_pre_bout = round_half_up(peak_idx + T_PRE_BOUT * FRAME_RATE)
    idx_mid_accel = round_half_up(peak_idx + T_MID_ACCEL * FRAME_RATE)
    idx_end = round_half_up(peak_idx + T_END * FRAME_RATE)

    idxRANGE = [idx_start,idx_end]
    # %%
    # initialize results dataframe

    all_for_fit = pd.DataFrame()
    mean_data = pd.DataFrame()
    all_dir.sort()
    exp_data_all = pd.DataFrame()
    for expNum, exp_path in enumerate(all_dir):
        rows = []
        exp_data = pd.read_hdf(f"{exp_path}/bout_data.h5", key='prop_bout_aligned')
        exp_data = exp_data.assign(idx=round_half_up(len(exp_data)/total_aligned)*list(range(0,total_aligned)))

        # - get the index of the rows in exp_data to keep (for each bout, there are range(0:51) frames. keep range(20:41) frames)
        bout_time = pd.read_hdf(f"{exp_path}/bout_data.h5", key='prop_bout2')
        
        # truncate first, just incase some aligned bouts aren't complete
        for i in day_night_split(bout_time,'aligned_time').index:
            rows.extend(list(range(i*total_aligned+idxRANGE[0],i*total_aligned+idxRANGE[1])))
        selected_range = exp_data.loc[rows,:]
        
        # calculate angular speed (smoothed)
        grp = selected_range.groupby(np.arange(len(selected_range))//(idxRANGE[1]-idxRANGE[0]))
        propBoutAligned_angVel = grp['propBoutAligned_pitch'].apply(
            lambda grp_pitch: np.diff(savgol_filter(grp_pitch, 11, 3),prepend=np.array([np.nan]))*FRAME_RATE,
        )
        propBoutAligned_angVel = propBoutAligned_angVel.apply(pd.Series).T.melt()
        # assign angvel and ang speed
        selected_range = selected_range.assign(
            propBoutAligned_angVel_sm = propBoutAligned_angVel['value'].values,
        )
        grp = selected_range.groupby(np.arange(len(selected_range))//(idxRANGE[1]-idxRANGE[0]))
        accel_angvel_mean = grp.apply(
            lambda group: group.loc[(group['idx']>idx_pre_bout)&(group['idx']<idx_mid_accel), 
                                    'propBoutAligned_angVel_sm'].mean()
        )
        adj_by_angvel = accel_angvel_mean/np.absolute(accel_angvel_mean)
        #|||||||||||||||||||||||||
        adj_by_which = adj_by_angvel  #  #
        #|||||||||||||||||||||||||
        
        adj_angvel = selected_range['propBoutAligned_angVel_sm'] * (np.repeat(adj_by_which,(idxRANGE[1]-idxRANGE[0])).values)

        selected_range = selected_range.assign(
            adj_angvel = adj_angvel,
        )

        this_exp_data = selected_range
        this_exp_data = this_exp_data.assign(
            time_ms = (this_exp_data['idx']-peak_idx)/FRAME_RATE*1000,
            expNum = expNum)
        
        exp_data_all = pd.concat([exp_data_all,this_exp_data])
        
    exp_data_all = exp_data_all.reset_index(drop=True)
    # %%
    # calculate mean max angvel time
    mean_angvel = exp_data_all.groupby(['time_ms','expNum'])['adj_angvel'].median().reset_index()
    mean_angvel = mean_angvel.loc[mean_angvel['time_ms']<0]
    idx_mean_max = mean_angvel.groupby(['expNum'])['adj_angvel'].apply(
        lambda y: np.argmax(y)
    )
    time_by_bout_max = ((idx_mean_max/166 + T_start)*1000).reset_index()
    time_by_bout_max.columns = ['expNum','max_angvel_time']
    max_angvel_time = time_by_bout_max['max_angvel_time'].mean()
    
    print(f"Max angvel time = {max_angvel_time}")
    # %%
    T_start = -0.3
    T_end = 0.25
    idx_start = round_half_up(peak_idx + T_start * FRAME_RATE)
    idx_end = round_half_up(peak_idx + T_end * FRAME_RATE)
    idxRANGE = [idx_start,idx_end]
    max_angvel_idx = round_half_up(peak_idx + max_angvel_time/1000 * FRAME_RATE)

    bout_features = pd.DataFrame()

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
            rows.extend(list(range(i*total_aligned+idxRANGE[0],i*total_aligned+idxRANGE[1])))
        exp_data = exp_data.assign(expNum = exp)
        trunc_day_exp_data = exp_data.loc[rows,:]
        trunc_day_exp_data = trunc_day_exp_data.assign(
            bout_num = trunc_day_exp_data.groupby(np.arange(len(trunc_day_exp_data))//(idxRANGE[1]-idxRANGE[0])).ngroup()
            )
        num_of_bouts = len(trunc_day_exp_data.loc[trunc_day_exp_data['idx'] == peak_idx])
        
        this_exp_features = extract_bout_features_v4(trunc_day_exp_data,peak_idx,FRAME_RATE,idx_max_angvel=max_angvel_idx)
        this_exp_features = this_exp_features.assign(
            expNum = [expNum]*num_of_bouts,
            )
        this_exp_features = this_exp_features.loc[this_exp_features['spd_peak']>=7].reset_index(drop=True)
        bout_features = pd.concat([bout_features,this_exp_features], ignore_index=True)
    # %%
    # sample
    all_for_fit = bout_features
    if if_jackknife:
        if if_sample:
            all_for_fit = bout_features.groupby(['expNum']).sample(
                n=SAMPLE_N,
                replace=True
                )
    binned_df = distribution_binned_average(all_for_fit,by_col='rot_to_max_angvel',bin_col='atk_ang',bin=AVERAGE_BIN)
    # %%
    
    # sigmoid fit

    # jeackknife resampling to estimate error
    if if_jackknife:
        jackknife_coef = pd.DataFrame()
        jackknife_y = pd.DataFrame()
        jackknife_idx = jackknife_resampling(np.arange(0,expNum+1))
        for excluded_exp, idx_group in enumerate(jackknife_idx):
            this_data = all_for_fit.loc[all_for_fit['expNum'].isin(idx_group)]
            df = this_data.loc[:,['atk_ang','rot_to_max_angvel']]  # filter for the data you want to use for calculation
            this_coef, this_y, this_sigma = sigmoid_fit(
                df, X_RANGE, func=sigfunc_4free, 
            )
            this_coef = this_coef.assign(
                excluded_exp = excluded_exp
            )
            this_y = this_y.assign(
                excluded_exp = excluded_exp
            )
            jackknife_coef = pd.concat([jackknife_coef,this_coef])
            jackknife_y = pd.concat([jackknife_y,this_y])

        jackknife_coef = jackknife_coef.reset_index(drop=True)
        jackknife_y = jackknife_y.reset_index(drop=True)

        output_par = jackknife_coef.iloc[:,0]*jackknife_coef.iloc[:,3]/4
        output_par.name = 'slope'
        output_par = output_par.to_frame().assign(
            expNum = jackknife_coef['excluded_exp'].values,
            height = jackknife_coef.iloc[:,3]
            )
        g = sns.lineplot(x='x',y=jackknife_y[0],data=jackknife_y,
                        err_style="band", errorbar='sd'
                        )
        g = sns.lineplot(x='rot_to_max_angvel',y='atk_ang',
                            data=binned_df,color='grey')
        g.set_xlabel("Rotation (deg)")
        g.set_ylabel("Attack angle (deg)")

        filename = os.path.join(fig_dir,"attack angle vs rotation (jackknife).pdf")
        plt.savefig(filename,format='PDF')
        plt.close()
        
        print(f"Sigmoid slope = {output_par['slope'].mean()}")
        print(f"Sigmoid height = {output_par['height'].mean()}")
    else:
        df = all_for_fit.loc[:,['atk_ang','rot_to_max_angvel']]
        coef_master, fitted_y_master, sigma_master = sigmoid_fit(
            df, X_RANGE, func=sigfunc_4free
            )
        g = sns.lineplot(x='x',y=fitted_y_master[0],data=fitted_y_master)
        g = sns.lineplot(x='rot_to_max_angvel',y='atk_ang',
                            data=binned_df,
                            color='grey')
        g.set_xlabel("Rotation (deg)")
        g.set_ylabel("Attack angle (deg)")

        filename = os.path.join(fig_dir,"attack angle vs rotation.pdf")
        plt.savefig(filename,format='PDF')
        plt.close()

        fit_height = coef_master[3]
        output_par = coef_master[0]*fit_height/4
        print(f"Sigmoid slope = {output_par.values}")
        print(f"Sigmoid height = {fit_height.values}")

        output_par.name = 'slope'
        output_par = output_par.to_frame().assign(
            expNum = expNum,
            height = fit_height.values,
            )

    # %% 
    # Slope
    p = sns.pointplot(data=output_par,
                    y='slope',
                    hue='expNum')
    p.set_ylabel("Maximal slope of fitted sigmoid")
    filename = os.path.join(fig_dir,"Fin-body ratio.pdf")
    plt.savefig(filename,format='PDF')
    plt.close()

    # Height
    p = sns.pointplot(data=output_par,
                    y='height',
                    hue='expNum')
    p.set_ylabel("Height of fitted sigmoid")
    filename = os.path.join(fig_dir,"Sigmoid height.pdf")
    plt.savefig(filename,format='PDF')
    plt.close()
# %%
if __name__ == "__main__":
    # if to use Command Line Inputs
    root = input("- Data directory? \n")
    plot_fin_body_coordination_byAngvelMax(root)

