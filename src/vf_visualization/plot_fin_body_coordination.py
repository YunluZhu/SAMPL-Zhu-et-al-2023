'''
Attack angles, pre-bout posture change, and fin-body ratio
https://elifesciences.org/articles/45839

This script calculates attack angles and pre-bout posture change and generates the fin-body ratio using a sigmoid fit

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
Default is off (sample_bout = -1)'''

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
# %%
def sigmoid_fit(df, x_range_to_fit,func,**kwargs):
    lower_bounds = [0.1,-20,-100,1]
    upper_bounds = [5,20,2,100]
    x0=[0.1, 1, -1, 20]
    
    for key, value in kwargs.items():
        if key == 'a':
            x0[0] = value
            lower_bounds[0] = value-0.01
            upper_bounds[0] = value+0.01
        elif key == 'b':
            x0[1] = value
            lower_bounds[1] = value-0.01
            upper_bounds[1] = value+0.01
        elif key == 'c':
            x0[2] = value
            lower_bounds[2] = value-0.01
            upper_bounds[2] = value+0.01
        elif key =='d':
            x0[3] = value
            lower_bounds[3] = value-0.01
            upper_bounds[3] = value+0.01
            
    p0 = tuple(x0)
    popt, pcov = curve_fit(func, df['posture_chg'], df['atk_ang'], 
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
def plot_atk_ang_posture_chg(root, **kwargs):
    print('\n- Plotting atk angle and fin-body ratio')

    if_sample = False
    SAMPLE_N = -1
    
    for key, value in kwargs.items():
        if key == 'sample_bout':
            SAMPLE_N = value
    if SAMPLE_N == -1:
        SAMPLE_N = int(input("How many bouts to sample from each dataset? ('0' for no sampling): "))
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
    BIN_WIDTH = 0.3
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
        FRAME_RATE = int(input("Frame rate? "))

    peak_idx, total_aligned = get_index(FRAME_RATE)
    T_pre_chg_start = 0.25  #s before peak
    T_pre_chg_end = 0.1 #s before peak
    idx_T_pre_rot_start = int(T_pre_chg_start * FRAME_RATE)
    idx_T_pre_rot_end = int(T_pre_chg_end * FRAME_RATE)

    # %%
    # initialize results dataframe

    all_for_fit = pd.DataFrame()
    mean_data = pd.DataFrame()
    all_dir.sort()
    for expNum, exp_path in enumerate(all_dir):
        exp_data = pd.read_hdf(f"{exp_path}/bout_data.h5", key='prop_bout_aligned').loc[:,['propBoutAligned_pitch','propBoutAligned_speed']]
        exp_data = exp_data.assign(idx=int(len(exp_data)/total_aligned)*list(range(0,total_aligned)))
        peak_angles = exp_data.loc[exp_data['idx']==peak_idx]
        bout_attributes = pd.read_hdf(f"{exp_path}/bout_data.h5", key='prop_bout2').loc[:,['aligned_time','epochBouts_trajectory']]
        peak_angles = peak_angles.assign(
            time = bout_attributes['aligned_time'].values,
            heading = bout_attributes['epochBouts_trajectory'].values,
            )  # peak angle
        
        peak_angles_day = day_night_split(peak_angles, 'time')
        # # filter for angles meet the condition
        # peak_angles_day = peak_angles_day.loc[(peak_angles_day['heading']<70) & 
        #                                         (peak_angles_day['heading']>-70)]

        # calculate individual attack angles (heading - pitch)
        atk_ang = peak_angles_day['heading'] - peak_angles_day['propBoutAligned_pitch']

        # get indices of bout peak (for posture change calculation)
        idx_at_peak = peak_angles_day.index
        posture_chg = exp_data.loc[idx_at_peak-idx_T_pre_rot_end, 'propBoutAligned_pitch'].values - exp_data.loc[idx_at_peak-idx_T_pre_rot_start, 'propBoutAligned_pitch']

        for_fit = pd.DataFrame(data={'atk_ang':atk_ang.values, 
                                    'posture_chg':posture_chg.values, 
                                    'heading':peak_angles_day['heading'], 
                                    'pitch':peak_angles_day['propBoutAligned_pitch'],
                                    'speed':peak_angles_day['propBoutAligned_speed'],
                                    'expNum':expNum,
                                })
        if if_sample == True:
            try:
                for_fit = for_fit.sample(n=SAMPLE_N)
            except:
                for_fit = for_fit.sample(n=SAMPLE_N,replace=True)
        all_for_fit = pd.concat([all_for_fit, for_fit], ignore_index=True)

    # clean up
    all_for_fit.drop(all_for_fit[all_for_fit['speed']<7].index, inplace=True)
    # %%
    # sigmoid fit
    df = all_for_fit.loc[:,['atk_ang','posture_chg']]
    coef_master, fitted_y_master, sigma_master = sigmoid_fit(
        df, X_RANGE, func=sigfunc_4free
        )
    g = sns.lineplot(x='x',y=fitted_y_master[0],data=fitted_y_master)
    binned_df = distribution_binned_average(df,by_col='posture_chg',bin_col='atk_ang',bin=AVERAGE_BIN)
    
    g = sns.lineplot(x='posture_chg',y='atk_ang',
                        data=binned_df,color='grey')
    g.set_xlabel("Posture change (deg)")
    g.set_ylabel("Attack angle (deg)")

    filename = os.path.join(fig_dir,"attack angle vs pre-bout rotation.pdf")
    plt.savefig(filename,format='PDF')
    plt.close()

    fit_atk_max = coef_master[3]
    slope = coef_master[0]*fit_atk_max/4
    print(f"Sigmoid slope = {slope.values}")

    if ~if_jackknife:  # if no repeats for jackknife resampling
        slope.name = 'slope'
        slope = slope.to_frame().assign(
            expNum = expNum
            )
    # %%
    # jeackknife resampling to estimate error
    jackknife_coef = pd.DataFrame()
    jackknife_y = pd.DataFrame()
    if if_jackknife:
        jackknife_idx = jackknife_resampling(np.arange(0,expNum+1))
        for excluded_exp, idx_group in enumerate(jackknife_idx):
            this_data = all_for_fit.loc[all_for_fit['expNum'].isin(idx_group)]
            df = this_data.loc[:,['atk_ang','posture_chg']]  # filter for the data you want to use for calculation
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

        slope = jackknife_coef.iloc[:,0]*jackknife_coef.iloc[:,3]/4
        slope.name = 'slope'
        slope = slope.to_frame().assign(
            expNum = jackknife_coef['excluded_exp'].values
            )
        g = sns.lineplot(x='x',y=jackknife_y[0],data=jackknife_y,
                        err_style="band", ci='sd'
                        )
        g = sns.lineplot(x='posture_chg',y='atk_ang',
                            data=binned_df,color='grey')
        g.set_xlabel("Posture change (deg)")
        g.set_ylabel("Attack angle (deg)")

        filename = os.path.join(fig_dir,"attack angle vs pre-bout rotation (jackknife).pdf")
        plt.savefig(filename,format='PDF')
        plt.close()
    
    # %%
    # Attack angle
    mean_data = all_for_fit.groupby('expNum').mean()
    mean_data = mean_data.reset_index()

    p = sns.pointplot(data=mean_data,
                    y='atk_ang',
                    hue='expNum')
    p.set_ylabel("Attack angle")
    filename = os.path.join(fig_dir,"attack angle.pdf")
    plt.savefig(filename,format='PDF')
    plt.close()
    
    # %% 
    # Slope
    p = sns.pointplot(data=slope,
                    y='slope',
                    hue='expNum')
    p.set_ylabel("Maximal slope of fitted sigmoid")
    filename = os.path.join(fig_dir,"Fin-body ratio.pdf")
    plt.savefig(filename,format='PDF')
    plt.close()

# %%
if __name__ == "__main__":
    # if to use Command Line Inputs
    root = input("- Data directory? \n")
    plot_atk_ang_posture_chg(root)