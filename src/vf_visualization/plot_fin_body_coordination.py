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
Body rotation is determined by rotation from -250 ms to -50 ms for ease of calculation. If needed, one may determine the rotation by finding time of the peak angular velocity. See scripts for Figure 5 for details.
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

# %%
def sigmoid_fit(df, x_range_to_fit,func,**kwargs):
    lower_bounds = [0.1,-20,-100,1]
    upper_bounds = [5,20,2,100]
    x0=[0.1, 1, -1, 20]
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

def plot_fin_body_coordination(root, **kwargs):
    """plot fin-body ratio and atk angle vs body rotation with sigmoid fit
    body rotation is calculated as rotation from initial to -50 ms

    Args:
        root (string): directory
        sample_bout (int): number of bouts to sample from each experimental repeat. default is off (sample_bout = -1)

    """
    print('\n- Plotting atk angle and fin-body ratio')

    if_sample = False
    SAMPLE_N = -1
    
    for key, value in kwargs.items():
        if key == 'sample_bout':
            SAMPLE_N = int(value)
    if SAMPLE_N == -1:
        SAMPLE_N = round_half_up(input("How many bouts to sample from each dataset? ('0' for no sampling): "))
    if SAMPLE_N > 0:
        if_sample = True
        
    folder_name = 'atk_ang fin_body_ratio'
    folder_dir = os.getcwd()
    fig_dir = os.path.join(folder_dir, 'figures', folder_name)

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
    BIN_WIDTH = 0.4
    AVERAGE_BIN = np.arange(min(X_RANGE),max(X_RANGE),BIN_WIDTH)

    
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

    # %%
# %%
    T_start = -0.3
    T_end = 0.25
    peak_idx, total_aligned = get_index(FRAME_RATE)
    idx_start = round_half_up(peak_idx + T_start * FRAME_RATE)
    idx_end = round_half_up(peak_idx + T_end * FRAME_RATE)
    idxRANGE = [idx_start,idx_end]

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
        
        this_exp_features = extract_bout_features_v4(trunc_day_exp_data,peak_idx,FRAME_RATE)
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
            height = jackknife_coef.iloc[:,3],
            k = jackknife_coef.iloc[:,0],
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
        print(f"Sigmoid k = {output_par['k'].mean()}")

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
    plot_fin_body_coordination(root)