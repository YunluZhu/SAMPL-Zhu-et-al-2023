#%%
import os
from plot_functions.plt_tools import round_half_up
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from astropy.stats import jackknife_resampling
from scipy.optimize import curve_fit
from plot_functions.get_data_dir import (get_figure_dir)
from plot_functions.get_bout_features import get_max_angvel_rot, get_bout_features
from plot_functions.plt_tools import (set_font_type, defaultPlotting, plot_pointplt, distribution_binned_average_nostd)
from statsmodels.stats.multicomp import MultiComparison

    
def sigmoid_fit(df, x_range_to_fit,func,**kwargs):
    lower_bounds = [0.1,0,-100,1]
    upper_bounds = [15,20,2,100]
    x0=[3, 1, 0, 5]
    # for key, value in kwargs.items():
    #     if key == 'a':
    #         x0[0] = value
    #         lower_bounds[0] = value-0.01
    #         upper_bounds[0] = value+0.01
    #     elif key == 'b':
    #         x0[1] = value
    #         lower_bounds[1] = value-0.01
    #         upper_bounds[1] = value+0.01
    #     elif key == 'c':
    #         x0[2] = value
    #         lower_bounds[2] = value-0.01
    #         upper_bounds[2] = value+0.01
    #     elif key =='d':
    #         x0[3] = value
    #         lower_bounds[3] = value-0.01
    #         upper_bounds[3] = value+0.01
            
    p0 = tuple(x0)
    popt, pcov = curve_fit(func, df['rot_to_max_angvel'], df['atk_ang'], 
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

def Fig7_bkg_fin_body(root):
    set_font_type()
    defaultPlotting(size=16)
    # %%
    which_zeitgeber = 'day'
    DAY_RESAMPLE = 0
    NIGHT_RESAMPLE = 500
    # if_use_maxAngvelTime_perCondition = 0 # if to calculate max adjusted angvel time for each condition and selectt range for body rotation differently
    #                                         # or to use -250ms to -50ms for all conditions
    # Select data and create figure folder
    FRAME_RATE = 166
    
    X_RANGE = np.arange(-5,10.01,0.01)
    BIN_WIDTH = 0.6
    AVERAGE_BIN = np.arange(min(X_RANGE),max(X_RANGE),BIN_WIDTH)

    print("- Figure 7: ZF strains - Fin-body coordination")

    folder_name = f'fin-body coordination'
    folder_dir = get_figure_dir('Fig_7')
    fig_dir = os.path.join(folder_dir, folder_name)

    try:
        os.makedirs(fig_dir)
        print(f'fig folder created:{folder_name}')
    except:
        pass

    # %% get max_angvel_time per condition
    which_rotation = 'rot_to_max_angvel'
    which_atk_ang = 'atk_ang' # atk_ang or 'atk_ang_phased'
    # get features

    # if if_use_maxAngvelTime_perCondition:
    #     max_angvel_time, all_cond1, all_cond2 = get_max_angvel_rot(root, FRAME_RATE, ztime = which_zeitgeber)
    #     all_feature_cond, all_cond1, all_cond2 = get_bout_features(root, FRAME_RATE, ztime = which_zeitgeber, max_angvel_time = max_angvel_time)
    # else:
    all_feature_cond, all_cond1, all_cond2 = get_bout_features(root, FRAME_RATE, ztime = which_zeitgeber )


    # %% tidy data
    df_toplt = all_feature_cond.sort_values(by=['condition','expNum']).reset_index(drop=True)
    if FRAME_RATE > 100:
        df_toplt.drop(df_toplt[df_toplt['spd_peak']<7].index, inplace=True)
    elif FRAME_RATE == 40:
        df_toplt.drop(df_toplt[df_toplt['spd_peak']<4].index, inplace=True)

    # df_toplt.drop(df_toplt[df_toplt['tsp_peak'].abs()>25].index, inplace=True)
    # df_toplt.drop(df_toplt[df_toplt['pitch_pre_bout'] <0].index, inplace=True)
    # %%
    angles_day_resampled = pd.DataFrame()
    angles_night_resampled = pd.DataFrame()

    if which_zeitgeber != 'night':
        angles_day_resampled = df_toplt.loc[
            df_toplt['ztime']=='day',:
                ]
        if DAY_RESAMPLE != 0:  # if resampled
            angles_day_resampled = angles_day_resampled.groupby(
                    ['dpf','condition','expNum']
                    ).sample(
                            n=DAY_RESAMPLE,
                            replace=True,
                            random_state=2
                            )
    if which_zeitgeber != 'day':
        angles_night_resampled = df_toplt.loc[
            df_toplt['ztime']=='night',:
                ]
        if NIGHT_RESAMPLE != 0:  # if resampled
            angles_night_resampled = angles_night_resampled.groupby(
                    ['dpf','condition','expNum']
                    ).sample(
                            n=NIGHT_RESAMPLE,
                            replace=True,
                            random_state=2
                            )
    df_toplt = pd.concat([angles_day_resampled,angles_night_resampled],ignore_index=True)
    
    # %% fit sigmoid - master
    all_coef = pd.DataFrame()
    all_y = pd.DataFrame()
    all_binned_average = pd.DataFrame()

    for (cond_abla,cond_dpf,cond_ztime), for_fit in df_toplt.groupby(['condition','dpf','ztime']):
        expNum = for_fit['expNum'].max()
        jackknife_idx = jackknife_resampling(np.array(list(range(expNum+1))))
        for excluded_exp, idx_group in enumerate(jackknife_idx):
            coef, fitted_y, sigma = sigmoid_fit(
                for_fit.loc[for_fit['expNum'].isin(idx_group)], X_RANGE, func=sigfunc_4free
            )
            slope = coef.iloc[0,0]*(coef.iloc[0,3]) / 4
            fitted_y.columns = ['Attack angle (deg)','Rotation (deg)']
            all_y = pd.concat([all_y, fitted_y.assign(
                dpf=cond_dpf,
                condition=cond_abla,
                excluded_exp = excluded_exp,
                ztime=cond_ztime,
                )])
            all_coef = pd.concat([all_coef, coef.assign(
                slope=slope,
                dpf=cond_dpf,
                condition=cond_abla,
                excluded_exp = excluded_exp,
                ztime=cond_ztime,
                )])
        binned_df = distribution_binned_average_nostd(for_fit,by_col=which_rotation,bin_col=which_atk_ang,bin=AVERAGE_BIN)
        binned_df.columns=['Rotation (deg)',which_atk_ang]
        all_binned_average = pd.concat([all_binned_average,binned_df.assign(
            dpf=cond_dpf,
            condition=cond_abla,
            ztime=cond_ztime,
            )],ignore_index=True)
        
    all_y = all_y.reset_index(drop=True)
    all_coef = all_coef.reset_index(drop=True)
    all_coef.columns=['k','xval','min','height',
                    'slope','dpf','condition','excluded_exp','ztime']
    all_ztime = list(set(all_coef['ztime']))
    all_ztime.sort()
    # %%
    # plot bout frequency vs IBI pitch and fit with parabola
    defaultPlotting(size=12)

    plt.figure()

    g = sns.relplot(x='Rotation (deg)',y='Attack angle (deg)', data=all_y, 
                    kind='line',
                    col='dpf', col_order=all_cond1,
                    row = 'ztime', row_order=all_ztime,
                    hue='condition', hue_order = all_cond2,errorbar='sd',
                    )
    for i , g_row in enumerate(g.axes):
        for j, ax in enumerate(g_row):
            sns.lineplot(data=all_binned_average.loc[
                (all_binned_average['dpf']==all_cond1[j]) & (all_binned_average['ztime']==all_ztime[i]),:
                    ], 
                        x='Rotation (deg)', y=which_atk_ang, 
                        hue='condition',alpha=0.5,legend=False,
                        ax=ax)
    upper = np.percentile(df_toplt[which_atk_ang], 80)
    lower = np.percentile(df_toplt[which_atk_ang], 24)
    g.set(ylim=(lower, upper))
    g.set(xlim=(-2, 6))

    filename = os.path.join(fig_dir,"attack angle vs rot to angvel max.pdf")
    plt.savefig(filename,format='PDF')

    # plot coef
    for coef_name in ['k','xval','min','height','slope']:
        plot_pointplt(all_coef,coef_name,all_cond2)
        filename = os.path.join(fig_dir,f"{coef_name}.pdf")
        plt.savefig(filename,format='PDF')
        
    
    # multiple comparison
    
    multi_comp = MultiComparison(all_coef['slope'], all_coef['condition'])
    print(multi_comp.tukeyhsd().summary())

# %%
if __name__ == "__main__":
    # if to use Command Line Inputs
    root = input("- Data directory: where is folder 'DD_7dpf_bkg'? \n")
    Fig7_bkg_fin_body(root)