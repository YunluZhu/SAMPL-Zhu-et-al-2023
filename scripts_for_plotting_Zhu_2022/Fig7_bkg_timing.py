'''
plot mean IBI bout frequency vs. IBI pitch and fit with a parabola
UP DN separated

zeitgeber time? Yes
Jackknife? Yes
Sampled? Yes - ONE sample number for day and night
- change the var RESAMPLE to select the number of bouts sampled per condition per repeat. 
- to disable sampling, change it to 0 
- If ztime == all, day and night count as 2 conditions
- for the pd.sample function, replace = True
'''

#%%
import os
from plot_functions.plt_tools import round_half_up
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from astropy.stats import jackknife_resampling
from scipy.stats import ttest_rel
from scipy.optimize import curve_fit
# from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)
from plot_functions.get_data_dir import (get_data_dir,get_figure_dir)
from plot_functions.plt_tools import (set_font_type, defaultPlotting, day_night_split)
from plot_functions.get_IBIangles import get_IBIangles

set_font_type()
defaultPlotting()
# %%
pick_data = '7dd_bkg'
which_ztime = 'day'
root, FRAME_RATE = get_data_dir(pick_data)
if_sample = True
SAMPLE_N = 1000

folder_dir = os.getcwd()
folder_name = f'{pick_data} bout_timing'
folder_dir = get_figure_dir('Fig_7')
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
except:
    pass

print('- Figure 7: Bout timing')

# %%
# CONSTANTS
# SAMPLES_PER_BIN = 70  # this adjusts the density of raw data points on the fitted parabola
BIN_WIDTH = 3  # this adjusts the density of raw data points on the fitted parabola
X_RANGE_FULL = range(-30,41,1)
frequency_th = 3 / 40 * FRAME_RATE

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

def parabola_fit1(df, X_RANGE_to_fit = X_RANGE_FULL):
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
    return output_coef, output_fitted

# %%
IBI_angles, cond1_all, cond2_all= get_IBIangles(root, FRAME_RATE, ztime=which_ztime)
IBI_angles = IBI_angles.assign(bout_freq=1/IBI_angles['propBoutIEI'])

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
        coef, fitted_y = parabola_fit1(this_df_toFit, X_RANGE_FULL)
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

jackknifed_y.columns = ['Bout frequency (Hz)','IBI pitch (deg)','dpf','condition','jackknife num','ztime']
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

g = sns.relplot(x='IBI pitch (deg)',y='Bout frequency (Hz)', data=jackknifed_y, 
                kind='line',
                col='dpf', col_order=cond1_all,
                hue='condition', hue_order = cond2_all,errorbar='sd',
                aspect=0.9 , height = 3,
                )
for i , g_row in enumerate(g.axes):
    for j, ax in enumerate(g_row):
        sns.scatterplot(data=binned_angles.loc[
            (binned_angles['dpf']==cond1_all[j]) & (binned_angles['ztime']==all_ztime[i]),:
                ], 
                    x='propBoutIEI_pitch', y='bout_freq', 
                    hue='condition',alpha=0.2,legend=False,
                    ax=ax)
g.set(xlim=(-25, 45),
      ylim=(None,2)
      )
leg = g._legend
leg.set_bbox_to_anchor([1,0.7])
filename = os.path.join(fig_dir,f"IBI timing parabola fit.pdf")
plt.savefig(filename,format='PDF')

# %%
# plot all coefs

plt.close()
# %%
# plot all coef
for i, coef_col_name in enumerate(coef_columns):
    p = sns.catplot(
        data = jackknifed_coef, y=coef_col_name,x='condition',kind='strip',
        color = 'grey',
        edgecolor = None,
        linewidth = 0,
        s=8, 
        alpha=0.3,
        height=3,
        zorder=1,
        aspect=1
    )
    p.map(sns.pointplot,'condition',coef_col_name,
        markers=['d','d','d'],
        order=cond2_all,
        join=False, 
        ci=None,
        color='black',
        zorder=100,
        data=jackknifed_coef)
    filename = os.path.join(fig_dir,f"IBI {coef_names[i]} sample{SAMPLE_N} ± SD.pdf")
    plt.savefig(filename,format='PDF')


# %%
