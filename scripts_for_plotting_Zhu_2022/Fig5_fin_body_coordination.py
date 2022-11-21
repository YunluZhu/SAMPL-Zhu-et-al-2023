#%%
import os
from plot_functions.plt_tools import round_half_up
import pandas as pd # pandas library
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from astropy.stats import jackknife_resampling
from scipy.optimize import curve_fit
from plot_functions.get_data_dir import ( get_figure_dir)
from plot_functions.get_bout_features import get_bout_features,get_max_angvel_rot
from plot_functions.plt_tools import (set_font_type, defaultPlotting,distribution_binned_average_nostd)
from scipy import stats

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


# %%
def Fig5_fin_body_coordination(root, root_fin):
    set_font_type()
    which_ztime = 'day'
    FRAME_RATE = 166
    DAY_RESAMPLE = 0
    folder_name = f'Atk_ang fin_body_ratio'
    folder_dir5 = get_figure_dir('Fig_5')
    fig_dir = os.path.join(folder_dir5, folder_name)

    try:
        os.makedirs(fig_dir)
        print(f'fig directory created: {fig_dir}')
    except:
        print('Figure folder already exist! Old figures will be replaced:')
        print(fig_dir)

    # %%
    # main function
    # CONSTANTS
    X_RANGE = np.arange(-2,8.01,0.01)
    BIN_WIDTH = 0.4
    AVERAGE_BIN = np.arange(min(X_RANGE),max(X_RANGE),BIN_WIDTH)
    max_angvel_time, _, _ = get_max_angvel_rot(root, FRAME_RATE)
    all_feature_cond, all_cond1, all_cond2 = get_bout_features(root, FRAME_RATE, max_angvel_time=max_angvel_time)
    all_feature_cond = all_feature_cond.reset_index(drop=True)
    # %%
    print("- Figure 5: Distribution of early body rotation and attack angle")
    feature_to_plt = ['rot_to_max_angvel','atk_ang']
    toplt = all_feature_cond

    for feature in feature_to_plt:
        plt.figure(figsize=(3,2))
        upper = np.percentile(toplt[feature], 99.5)
        lower = np.percentile(toplt[feature], 1)
        if feature == 'atk_ang':
            atk_ctrl_upper = upper
            atk_ctrl_lower = lower
        xlabel = feature + " (deg)"
        
        g = sns.histplot(data=toplt, x=feature, 
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
        plt.savefig(fig_dir+f"/{feature} distribution.pdf",format='PDF')
        # plt.close()
    # %% tidy data
    all_feature_cond = all_feature_cond.sort_values(by=['condition','expNum']).reset_index(drop=True)
    all_feature_cond.drop(all_feature_cond[all_feature_cond['spd_peak']<7].index, inplace=True)
    # %% 5D
    print("- Figure 5: correlation of attack angle with rotation and rotation residual")
    toplt = all_feature_cond
    plt_dict = {
        'early_rotation vs atk_ang':['rot_to_max_angvel','atk_ang'],
        'late_rotation vs atk_ang':['rot_residual','atk_ang'],
    }

    # %%
    for which_to_plot in plt_dict:
        [x,y] = plt_dict[which_to_plot]
        upper = np.percentile(toplt[x], 99)
        lower = np.percentile(toplt[x], 2)
        BIN_WIDTH = 0.5
        AVERAGE_BIN = np.arange(round_half_up(lower),round_half_up(upper),BIN_WIDTH)
        binned_df = toplt.groupby(['condition','dpf']).apply(
            lambda group: distribution_binned_average_nostd(group,by_col=x,bin_col=y,bin=AVERAGE_BIN)
        )
        binned_df.columns=[x,y]
        binned_df = binned_df.reset_index(level=['dpf','condition'])
        binned_df = binned_df.reset_index(drop=True)

        # xlabel = "Relative pitch change (deg)"
        # ylabel = 'Trajectory deviation (deg)'
    
        g = sns.relplot(
            kind='scatter',
            data = toplt.sample(n=4000),
            row='condition',
            col = 'dpf',
            col_order = all_cond1,
            row_order = all_cond2,
            x = x,
            y = y,
            alpha=0.1,
            linewidth = 0,
            color = 'grey',
            height=3,
            aspect=2/2,
            legend=False
            )
        for i , g_row in enumerate(g.axes):
            for j, ax in enumerate(g_row):
                sns.lineplot(data=binned_df.loc[(binned_df['dpf']==all_cond1[j]) & 
                                                (binned_df['condition']==all_cond2[i])], 
                            x=x, y=y, 
                            hue='condition',alpha=1,
                            legend=False,
                            ax=ax)
        
        g.set(ylim=(-12,16))
        g.set(xlim=(lower,upper))
        g.set(xlabel=x+" (deg)")
        g.set(ylabel=y+" (deg)")

        # g.set_axis_labels(x_var = xlabel, y_var = ylabel)
        sns.despine()
        plt.savefig(fig_dir+f"/{x} {y} correlation.pdf",format='PDF')
        r_val = stats.pearsonr(toplt[x],toplt[y])[0]
        print(f"pearson's r = {r_val}")
        
        
    # %% fit sigmoid 
    angles_day_resampled = pd.DataFrame()
    angles_night_resampled = pd.DataFrame()

    if which_ztime != 'night':
        angles_day_resampled = all_feature_cond.loc[
            all_feature_cond['ztime']=='day',:
                ]
        if DAY_RESAMPLE != 0:  # if resampled
            angles_day_resampled = angles_day_resampled.groupby(
                    ['dpf','condition','expNum']
                    ).sample(
                            n=DAY_RESAMPLE,
                            replace=True
                            )
    df_toplt = angles_day_resampled.reset_index(drop=True)
    # %% fit sigmoid - master
    all_coef = pd.DataFrame()
    all_y = pd.DataFrame()
    all_binned_average = pd.DataFrame()

    upper = np.percentile(df_toplt['rot_to_max_angvel'], 99)
    lower = np.percentile(df_toplt['rot_to_max_angvel'], 1)
    BIN_WIDTH = 0.6
    AVERAGE_BIN = np.arange(round_half_up(lower),round_half_up(upper),BIN_WIDTH)

    for (cond_abla,cond_dpf,cond_ztime), for_fit in df_toplt.groupby(['condition','dpf','ztime']):

        expNum = for_fit['expNum'].max()
        jackknife_idx = jackknife_resampling(np.array(list(range(expNum+1))))
        for excluded_exp, idx_group in enumerate(jackknife_idx):
            coef, fitted_y, sigma = sigmoid_fit(
                for_fit.loc[for_fit['expNum'].isin(idx_group)], X_RANGE, func=sigfunc_4free
            )
            slope = coef.iloc[0,0]*(coef.iloc[0,3]) / 4
            fitted_y.columns = ['Attack angle (deg)','rotation (deg)']
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
        binned_df = distribution_binned_average_nostd(for_fit,by_col='rot_to_max_angvel',bin_col='atk_ang',bin=AVERAGE_BIN)
        binned_df.columns=['rotation (deg)','atk_ang']
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

    defaultPlotting(size=12)

    plt.figure()

    g = sns.relplot(x='rotation (deg)',y='Attack angle (deg)', data=all_y, 
                    kind='line',
                    col='dpf', 
                    row = 'ztime', row_order=all_ztime,
                    errorbar='sd',
                    )
    for i , g_row in enumerate(g.axes):
        for j, ax in enumerate(g_row):
            sns.lineplot(data=all_binned_average.loc[
                (all_binned_average['dpf']==all_cond1[j]) & (all_binned_average['ztime']==all_ztime[i]),:
                    ], 
                        x='rotation (deg)', y='atk_ang', alpha=0.5,
                        ax=ax)
    upper = np.percentile(df_toplt['atk_ang'], 80)
    lower = np.percentile(df_toplt['atk_ang'], 20)
    g.set(ylim=(-1, 4.5))
    g.set(xlim=(-2, 7))

    filename = os.path.join(fig_dir,"fin-body coordination.pdf")
    plt.savefig(filename,format='PDF')

    # %%
    # plot 
    # plt.close()
    defaultPlotting(size=12)
    plt.figure()
    p = sns.catplot(
        data = all_coef, y='slope',x='dpf',kind='point',join=False,
        col_order=all_cond1,errorbar='sd',
        row = 'ztime', row_order=all_ztime,
        # units=excluded_exp,
        hue='condition', dodge=True,
        hue_order = all_cond2,
    )
    p.map(sns.lineplot,'dpf','slope',estimator=None,
        units='excluded_exp',
        hue='condition',
        alpha=0.2,
        data=all_coef)
    filename = os.path.join(fig_dir,"slope_together.pdf")
    plt.savefig(filename,format='PDF')

    # %%

    mean_val = all_coef['slope'].mean()
    std_val = all_coef['slope'].std()
    print(f"maximal slope: {mean_val:.3f}±{std_val:.3f}")

    mean_val = (all_coef['height']+all_coef['min']).mean()
    std_val = (all_coef['height']+all_coef['min']).std()
    print(f"upper asymptote: {mean_val:.3f}±{std_val:.3f}")

    # %%

    # plot finless fish data
    X_RANGE = np.arange(-2,8.01,0.01)
    BIN_WIDTH = 0.4
    AVERAGE_BIN = np.arange(min(X_RANGE),max(X_RANGE),BIN_WIDTH)
    all_feature_finless, fin_cond1, fin_cond2 = get_bout_features(root_fin, FRAME_RATE)
    all_feature_finless = all_feature_finless.reset_index(drop=True)

    feature = 'atk_ang'
    if 'spd' in feature:
        xlabel = feature + " (mm*s^-1)"
    elif 'dis' in feature:
        xlabel = feature + " (mm)"
    else:
        xlabel = feature + " (deg)"
    plt.figure(figsize=(3,2))
    upper = atk_ctrl_upper
    lower =atk_ctrl_lower

    g = sns.histplot(data=all_feature_finless, 
                    x=feature, 
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
    plt.savefig(os.path.join(fig_dir,f"{feature} distribution finless fish.pdf"),format='PDF')
# %%
if __name__ == "__main__":
    # if to use Command Line Inputs
    root = input("- Data directory: where is folder 'DD_7dpf_combined'? \n")
    root_finless = input("- Data directory: where is folder 'DD_finless'? \n")
    Fig5_fin_body_coordination(root, root_finless)