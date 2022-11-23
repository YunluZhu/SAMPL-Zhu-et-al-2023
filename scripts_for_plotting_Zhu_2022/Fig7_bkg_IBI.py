#%%
import sys
from plot_functions.plt_tools import round_half_up
import os
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from astropy.stats import jackknife_resampling
from plot_functions.get_data_dir import ( get_figure_dir)
from plot_functions.plt_tools import (defaultPlotting, set_font_type, plot_pointplt)
from plot_functions.get_IBIangles import get_IBIangles



def Fig7_bkg_IBI(root):
    defaultPlotting()
    set_font_type()
    # Paste root directory here
    which_zeitgeber = 'day'
    DAY_RESAMPLE = 0

    # %%

    print("- Figure 7: ZF strains - IBI")

    FRAME_RATE = 166
    
    folder_name = f'IBI features'
    folder_dir = get_figure_dir('Fig_7')
    fig_dir = os.path.join(folder_dir, folder_name)

    try:
        os.makedirs(fig_dir)
        print(f'fig folder created:{folder_name}')
    except:
        pass

    # %%
    # main function
    all_conditions = []
    folder_paths = []
    # get the name of all folders under root
    for folder in os.listdir(root):
        if folder[0] != '.':
            folder_paths.append(root+'/'+folder)
            all_conditions.append(folder)

    IBI_angles, cond1, cond2 = get_IBIangles(root, FRAME_RATE, ztime=which_zeitgeber)
    IBI_angles_cond = IBI_angles.loc[:,['propBoutIEI_pitch','propBoutIEI','ztime','expNum','dpf','condition','exp']]
    IBI_angles_cond.columns = ['IBI_pitch','IBI','ztime','expNum','dpf','condition','exp']
    IBI_angles_cond.reset_index(drop=True,inplace=True)
    cond_cols = ['ztime','dpf','condition']
    all_ztime = list(set(IBI_angles_cond.ztime))
    all_ztime.sort()
    
    # %% jackknife for day bouts

    jackknifed_night_std = pd.DataFrame()
    jackknifed_day_std = pd.DataFrame()

    if which_zeitgeber != 'night':
        IBI_angles_day_resampled = IBI_angles_cond.loc[
            IBI_angles_cond['ztime']=='day',:
                ]
        if DAY_RESAMPLE != 0:  # if resampled
            IBI_angles_day_resampled = IBI_angles_day_resampled.groupby(
                    ['dpf','condition','exp']
                    ).sample(
                            n=DAY_RESAMPLE,
                            replace=True
                            )
        cat_cols = ['condition','dpf','ztime']
        for (this_cond, this_dpf, this_ztime), group in IBI_angles_day_resampled.groupby(cat_cols):
            jackknife_idx = jackknife_resampling(np.array(list(range(group['expNum'].max()+1))))
            for excluded_exp, idx_group in enumerate(jackknife_idx):
                this_std = group.loc[group['expNum'].isin(idx_group),['IBI_pitch']].std().to_frame(name='jackknifed_std')
                this_mean = group.loc[group['expNum'].isin(idx_group),['IBI_pitch']].mean()
                this_IBI = group.loc[group['expNum'].isin(idx_group),['IBI']].mean()
                jackknifed_day_std = pd.concat([jackknifed_day_std, this_std.assign(dpf=this_dpf,
                                                                        condition=this_cond,
                                                                        excluded_exp=excluded_exp,
                                                                        ztime=this_ztime,
                                                                        jackknifed_mean=this_mean.values,
                                                                        jackknife_IBI = this_IBI.values)])
        jackknifed_day_std = jackknifed_day_std.reset_index(drop=True)


    # if which_zeitgeber != 'day':
    #     IBI_angles_night_resampled = IBI_angles_cond.loc[
    #         IBI_angles_cond['ztime']=='night',:
    #             ]
    #     if NIGHT_RESAMPLE != 0:  # if resampled
    #         IBI_angles_night_resampled = IBI_angles_night_resampled.groupby(
    #                 ['dpf','condition','exp']
    #                 ).sample(
    #                         n=NIGHT_RESAMPLE,
    #                         replace=True
    #                         )
    #     cat_cols = ['condition','dpf','ztime']
    #     for (this_cond, this_dpf, this_ztime), group in IBI_angles_night_resampled.groupby(cat_cols):
    #         jackknife_idx = jackknife_resampling(np.array(list(range(group['expNum'].max()+1))))
    #         for excluded_exp, idx_group in enumerate(jackknife_idx):
    #             this_std = group.loc[group['expNum'].isin(idx_group),['IBI_pitch']].std().to_frame(name='jackknifed_std')
    #             this_mean = group.loc[group['expNum'].isin(idx_group),['IBI_pitch']].mean()
    #             jackknifed_night_std = pd.concat([jackknifed_night_std, this_std.assign(dpf=this_dpf,
    #                                                                     condition=this_cond,
    #                                                                     excluded_exp=excluded_exp,
    #                                                                     ztime=this_ztime,
    #                                                                     jackknifed_mean=this_mean)])
    #     jackknifed_night_std = jackknifed_night_std.reset_index(drop=True)

    jackknifed_std = pd.concat([jackknifed_day_std,jackknifed_night_std]).reset_index(drop=True)
    # IBI_std_cond = IBI_angles_cond.groupby(['ztime','dpf','condition','exp','expNum']).std().reset_index()
    # IBI_std_day_resampled = IBI_angles_day_resampled.groupby(['ztime','dpf','condition','expNum']).std().reset_index()

    coef_columns = ['IBI pitch std (deg)', 'IBI pitch (deg)', 'IBI duration (s)']
    jackknifed_std = jackknifed_std.loc[:,['jackknifed_std','jackknifed_mean','jackknife_IBI','condition']]
    jackknifed_std.columns = coef_columns + ['condition']
    
    # plot IBI features

    for i, coef_col_name in enumerate(coef_columns):
        plot_pointplt(jackknifed_std,coef_col_name,cond2)
        filename = os.path.join(fig_dir,f"{coef_col_name} .pdf")
        plt.savefig(filename,format='PDF')


# %%
if __name__ == "__main__":
    # if to use Command Line Inputs
    root = input("- Data directory: where is folder 'DD_7dpf_bkg'? \n")
    Fig7_bkg_IBI(root)