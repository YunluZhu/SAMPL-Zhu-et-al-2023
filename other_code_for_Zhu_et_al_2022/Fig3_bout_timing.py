'''
plot mean IBI bout frequency vs. IBI pitch and fit with a parabola
UP DN separated

zeitgeber time? Yes
Jackknife? Yes
Sampled? Yes - ONE sample number for day and night
- change the var SAMPLE_N to select the number of bouts sampled per condition per repeat. 
- to disable sampling, change it to 0 
- If ztime == all, day and night count as 2 conditions
- for the pd.sample function, replace = True
'''

#%%
import os
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from astropy.stats import jackknife_resampling
from scipy.stats import ttest_rel
from scipy.optimize import curve_fit
from plot_functions.get_data_dir import (get_data_dir,get_figure_dir)
from plot_functions.plt_tools import (set_font_type, defaultPlotting, day_night_split)
from plot_functions.get_IBIangles import get_IBIangles
import scipy.stats as st

set_font_type()
defaultPlotting()
# %%
pick_data = 'all_7dd'
which_ztime = 'day'
root, FRAME_RATE = get_data_dir(pick_data)
if_sample = True
SAMPLE_N = 1000

folder_dir = os.getcwd()
folder_name = f'{pick_data} bout_timing'
folder_dir3 = get_figure_dir('Fig_3')
fig_dir = os.path.join(folder_dir3, folder_name)
ci_fig = get_figure_dir('SupFig_CI')

try:
    os.makedirs(fig_dir)
except:
    pass
try:
    os.makedirs(ci_fig)
except:
    pass
print('- Figure 3: Bout timing')

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
    p_sigma = np.sqrt(np.diag(pcov))

    return output_coef, output_fitted, p_sigma

# %%
IBI_angles, cond1_all, cond2_all= get_IBIangles(root, FRAME_RATE, ztime=which_ztime)
IBI_angles = IBI_angles.assign(bout_freq=1/IBI_angles['propBoutIEI'])
# IBI_angles = IBI_angles.loc[IBI_angles['bout_freq']<frequency_th]
# IBI_angles = IBI_angles.loc[IBI_angles['propBoutIEI_angVel'].abs()<30]
# IBI_angles = IBI_angles.loc[IBI_angles['propBoutIEI_pitch'].abs()<65]

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
                hue='condition', hue_order = cond2_all,ci='sd',
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
        col_order=cond1_all,ci='sd',
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

# %%
# plot CI of slope
print("Figure supp - CI width vs sample size - bout timing sensitivity")

list_of_sample_N = np.arange(1000,len(IBI_angles),500)
repeated_res = pd.DataFrame()
num_of_repeats = 20
rep = 0

while rep < num_of_repeats:
    list_of_ci_width = []
    for sample_N in list_of_sample_N:
        sample_for_fit = IBI_angles.sample(n=sample_N)[['propBoutIEI_pitch','bout_freq','propBoutIEI']]
        sample_for_fit.dropna(inplace=True)
        coef, fitted_y, sigma = parabola_fit1(sample_for_fit, X_RANGE_FULL)
        E_sensitivity = coef.iloc[0,0] * 1000
        sigma_sensitivity = sigma[0] * 1000
                
        (ci_low, ci_high) = st.norm.interval(0.95, loc=E_sensitivity, scale=sigma_sensitivity)
        ci_width = ci_high - ci_low
        list_of_ci_width.append(ci_width)
    res = pd.DataFrame(
        data = {
            'sample':list_of_sample_N,
            'CI width': list_of_ci_width,
        }
    )
    repeated_res = pd.concat([repeated_res,res],ignore_index=True)
    rep+=1

plt.figure(figsize=(5,4))
g = sns.lineplot(
    data = repeated_res,
    x = 'sample',
    y = 'CI width'
)
filename = os.path.join(ci_fig,"bout timing sensitivity CI width.pdf")
plt.savefig(filename,format='PDF')
# %%
