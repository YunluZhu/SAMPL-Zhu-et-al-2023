#%%
import os
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from astropy.stats import jackknife_resampling
from scipy.optimize import curve_fit
from plot_functions.get_data_dir import (get_figure_dir)
from plot_functions.plt_tools import (set_font_type, defaultPlotting, day_night_split)
from plot_functions.get_IBIangles import get_IBIangles
import scipy.stats as st

def distribution_binned_average(df, bin_width):
    '''
    bins raw pitch data using fixed bin width. Return binned mean of pitch and bout frequency.
    '''
    df = df.sort_values(by='propBoutIEI_pitch')
    # df = df.assign(bout_freq = 1/df['propBoutIEI'])
    bins = pd.cut(df['propBoutIEI_pitch'], list(np.arange(-90,90,bin_width)))
    grp = df.groupby(bins)
    df_out = grp[['propBoutIEI_pitch','bout_freq']].mean()
    return df_out
    
def ffunc1(x, a, b, c):
    # parabola function
    return a*((x-b)**2)+c

def parabola_fit1(df, X_RANGE_to_fit):
    '''
    fit bout probability - pitch to parabola
    May need to adjust bounds
    '''
    popt, pcov = curve_fit(ffunc1, df['propBoutIEI_pitch'], df['bout_freq'], 
                           p0=(0.005,3,0.5) , 
                           bounds=((0, -5, 0),(10, 15, 10)))
    # output = pd.DataFrame(data=popt,columns=['sensitivity','x_inter','y_inter'])
    # output = output.assign(condition=condition)
    y = []
    for x in X_RANGE_to_fit:
        y.append(ffunc1(x,*popt))
    output_coef = pd.DataFrame(data=popt).transpose()
    output_fitted = pd.DataFrame(data=y).assign(x=X_RANGE_to_fit)
    p_sigma = np.sqrt(np.diag(pcov))

    return output_coef, output_fitted, p_sigma


def Fig3_bout_timing(root):
    set_font_type()
    defaultPlotting()
    # %%
    which_ztime = 'day'
    FRAME_RATE = 166
    if_sample = True
    SAMPLE_N = 1000

    folder_dir = os.getcwd()
    folder_name = f'Bout timing'
    folder_dir3 = get_figure_dir('Fig_3')
    fig_dir = os.path.join(folder_dir3, folder_name)

    try:
        os.makedirs(fig_dir)
    except:
        pass

    print('- Figure 3: Bout timing')

    # %%
    # CONSTANTS
    # SAMPLES_PER_BIN = 70  # this adjusts the density of raw data points on the fitted parabola
    BIN_WIDTH = 3  # this adjusts the density of raw data points on the fitted parabola
    X_RANGE_FULL = range(-30,41,1)
    frequency_th = 3 / 40 * FRAME_RATE

    # %%
    IBI_angles, cond1_all, cond2_all= get_IBIangles(root, FRAME_RATE, ztime=which_ztime)
    IBI_angles = IBI_angles.assign(bout_freq=1/IBI_angles['propBoutIEI'])

    # %% distribution of IBI
    toplt = IBI_angles
    all_features = ['propBoutIEI_pitch','propBoutIEI']
    for feature_toplt in (all_features):
        # let's add unit
        if 'pitch' in feature_toplt:
            xlabel = "IBI pitch (deg)"
        else:
            xlabel = "Inter-bout interval (s)"
        plt.figure(figsize=(3,2))
        upper = np.nanpercentile(toplt[feature_toplt], 99.5)
        lower = np.nanpercentile(toplt[feature_toplt], 0.5)
        
        g = sns.histplot(data=toplt, x=feature_toplt, 
                            bins = 20, 
                            element="poly",
                            #  kde=True, 
                            stat="probability",
                            pthresh=0.05,
                            binrange=(lower,upper),
                            color='grey'
                            )
        g.set_xlabel(xlabel)
        sns.despine()
        plt.savefig(fig_dir+f"/{feature_toplt} distribution.pdf",format='PDF')



    # %%
    # Distribution plot, bout frequency vs IBI pitch

    toplt = IBI_angles[['bout_freq','propBoutIEI_pitch']]
    toplt.columns=['Bout frequency (Hz)','IBI pitch (deg)']
    plt.figure()
    g = sns.displot(data=toplt,
                    x='IBI pitch (deg)',y='Bout frequency (Hz)',
                    aspect=0.8,
                    cbar=True)
    g.set(xlim=(-25, 45),
        ylim=(None,3.5)
        )

    filename = os.path.join(fig_dir,f"IBI timing distribution.pdf")
    plt.savefig(filename,format='PDF')

    # %%
    jackknifed_coef = pd.DataFrame()
    jackknifed_y = pd.DataFrame()
    binned_angles = pd.DataFrame()
    cat_cols = ['condition','dpf','ztime']

    IBI_sampled = IBI_angles
    if SAMPLE_N !=0:
        IBI_sampled = IBI_sampled.groupby(['condition','dpf','ztime','exp']).sample(
            n=SAMPLE_N,
            replace=True,
            )
    for (this_cond, this_dpf, this_ztime), group in IBI_sampled.groupby(cat_cols):
        jackknife_idx = jackknife_resampling(np.array(list(range(group['expNum'].max()+1))))
        for excluded_exp, idx_group in enumerate(jackknife_idx):
            this_df_toFit = group.loc[group['expNum'].isin(idx_group),['propBoutIEI_pitch','bout_freq','propBoutIEI']].reset_index(drop=True)
            this_df_toFit.dropna(inplace=True)
            coef, fitted_y, p_sigma = parabola_fit1(this_df_toFit, X_RANGE_FULL)
            jackknifed_coef = pd.concat([jackknifed_coef, coef.assign(dpf=this_dpf,
                                                                    condition=this_cond,
                                                                    excluded_exp=excluded_exp,
                                                                    ztime=this_ztime)])
            jackknifed_y = pd.concat([jackknifed_y, fitted_y.assign(dpf=this_dpf,
                                                                    condition=this_cond,
                                                                    excluded_exp=excluded_exp,
                                                                    ztime=this_ztime)])
            
        this_binned_angles = distribution_binned_average(this_df_toFit, BIN_WIDTH)
        this_binned_angles = this_binned_angles.assign(dpf=this_dpf,
                                                        condition=this_cond,
                                                        ztime=this_ztime)
        binned_angles = pd.concat([binned_angles, this_binned_angles],ignore_index=True)

    jackknifed_y.columns = ['Bout frequency','IBI pitch','dpf','condition','jackknife num','ztime']
    jackknifed_y = jackknifed_y.reset_index(drop=True)
    coef_columns = ['Sensitivity (mHz/deg^2)','Baseline posture (deg)','Base bout rate (Hz)']
    coef_names = ['Sensitivity','Baseline posture','Base bout rate']
    jackknifed_coef.columns = coef_columns + ['dpf','condition','jackknife num','ztime']
    jackknifed_coef = jackknifed_coef.reset_index(drop=True)

    binned_angles = binned_angles.reset_index(drop=True)

    all_ztime = list(set(jackknifed_coef['ztime']))
    all_ztime.sort()

    jackknifed_coef['Sensitivity (mHz/deg^2)'] = jackknifed_coef['Sensitivity (mHz/deg^2)']*1000

    print("Mean of coefficients for fitted parabola:")
    print(jackknifed_coef[coef_columns].mean())
    print("Std. of coefficients for fitted parabola:")
    print(jackknifed_coef[coef_columns].std())
    # %% plot bout frequency vs IBI pitch, fit with parabola

    g = sns.relplot(x='IBI pitch',y='Bout frequency', data=jackknifed_y, 
                    kind='line',
                    col='dpf', col_order=cond1_all,
                    hue='condition', hue_order = cond2_all,errorbar='sd',
                    aspect=0.8
                    )
    for i , g_row in enumerate(g.axes):
        for j, ax in enumerate(g_row):
            sns.scatterplot(data=binned_angles.loc[
                (binned_angles['dpf']==cond1_all[j]) & (binned_angles['ztime']==all_ztime[i]),:
                    ], 
                        x='propBoutIEI_pitch', y='bout_freq', 
                        hue='condition',alpha=0.2,
                        ax=ax)
    g.set(xlim=(-25, 45),
        ylim=(None,2)
        )
    g._legend.remove()
        
    filename = os.path.join(fig_dir,f"IBI timing parabola fit.pdf")
    plt.savefig(filename,format='PDF')

    # %%
    # plot all coefs

    plt.close()

    for i, coef_col_name in enumerate(coef_columns):
        p = sns.catplot(
            data = jackknifed_coef, y=coef_col_name,x='dpf',kind='point',join=False,
            col_order=cond1_all,errorbar='sd',
            hue='condition', dodge=True,
            hue_order = cond2_all,sharey=False
        
        )
        p.map(sns.lineplot,'dpf',coef_col_name,estimator=None,
            units='jackknife num',
            hue='condition',
            alpha=0.2,
            data=jackknifed_coef)
        filename = os.path.join(fig_dir,f"IBI {coef_names[i]} sample{SAMPLE_N} ± SD.pdf")
        plt.savefig(filename,format='PDF')

if __name__ == "__main__":
    # if to use Command Line Inputs
    root = input("- Data directory: where is folder 'DD_7dpf_combined'? \n")
    Fig3_bout_timing(root)