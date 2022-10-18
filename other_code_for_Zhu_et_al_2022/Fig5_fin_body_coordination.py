'''

'''

#%%
import os
import pandas as pd # pandas library
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from astropy.stats import jackknife_resampling
from scipy.optimize import curve_fit
from plot_functions.get_index import (get_index)
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.get_bout_features import get_bout_features
from plot_functions.plt_tools import (set_font_type, defaultPlotting,distribution_binned_average,distribution_binned_average_nostd)
import scipy.stats as st
from scipy import stats


# %%
def sigmoid_fit(df, x_range_to_fit,func,**kwargs):
    lower_bounds = [0.1,-20,-100,1]
    upper_bounds = [5,20,2,100]
    x0=[0.1, 1, -1, 20]
    
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
    popt, pcov = curve_fit(func, df['rot_early'], df['atk_ang'], 
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
pick_data = 'all_7dd'
which_ztime = 'day'
root, FRAME_RATE = get_data_dir(pick_data)
if_sample = True
SAMPLE_N = 1000

folder_name = f'{pick_data} atk_ang fin_body_ratio'
folder_dir5 = get_figure_dir('Fig_5')
fig_dir = os.path.join(folder_dir5, folder_name)
ci_fig = get_figure_dir('SupFig_CI')

try:
    os.makedirs(fig_dir)
    print(f'fig directory created: {fig_dir}')
except:
    print('Figure folder already exist! Old figures will be replaced:')
    print(fig_dir)

try:
    os.makedirs(ci_fig)
except:
    pass
# %%
# main function
# CONSTANTS
X_RANGE = np.arange(-2,8.01,0.01)
BIN_WIDTH = 0.4
AVERAGE_BIN = np.arange(min(X_RANGE),max(X_RANGE),BIN_WIDTH)
all_feature_cond, all_cond1, all_cond2 = get_bout_features(root, FRAME_RATE)
all_feature_cond = all_feature_cond.reset_index(drop=True)
# %%
print("- Figure 5: Distribution of early body rotation and attack angle")
feature_to_plt = ['rot_early','atk_ang']
toplt = all_feature_cond

for feature in feature_to_plt:
    # let's add unit
    if 'spd' in feature:
        xlabel = feature + " (mm*s^-1)"
    elif 'dis' in feature:
        xlabel = feature + " (mm)"
    else:
        xlabel = feature + " (deg)"
    plt.figure(figsize=(3,2))
    upper = np.percentile(toplt[feature], 99.5)
    lower = np.percentile(toplt[feature], 1)
    
    if feature == 'atk_ang':
        atk_ctrl_upper = upper
        atk_ctrl_lower = lower
    
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
if FRAME_RATE > 100:
    all_feature_cond.drop(all_feature_cond[all_feature_cond['spd_peak']<7].index, inplace=True)
elif FRAME_RATE == 40:
    all_feature_cond.drop(all_feature_cond[all_feature_cond['spd_peak']<4].index, inplace=True)

# %% 5D
print("- Figure 5: correlation of attack angle with rotation and rotation residual")
toplt = all_feature_cond
plt_dict = {
    'early_rotation vs atk_ang':['rot_early','atk_ang'],
    'late_rotation vs atk_ang':['rot_late_accel','atk_ang'],
}

for which_to_plot in plt_dict:
    [x,y] = plt_dict[which_to_plot]

    upper = np.percentile(toplt[x], 99)
    lower = np.percentile(toplt[x], 1)
    BIN_WIDTH = 1
    AVERAGE_BIN = np.arange(int(lower),int(upper),BIN_WIDTH)
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
        data = toplt.sample(frac=0.5),
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
        )
    for i , g_row in enumerate(g.axes):
        for j, ax in enumerate(g_row):
            sns.lineplot(data=binned_df.loc[(binned_df['dpf']==all_cond1[j]) & 
                                            (binned_df['condition']==all_cond2[i])], 
                        x=x, y=y, 
                        hue='condition',alpha=1,
                        ax=ax)
    
    g.set(ylim=(-15,20))
    g.set(xlim=(lower,upper))
    
    # g.set_axis_labels(x_var = xlabel, y_var = ylabel)
    sns.despine()
    plt.savefig(fig_dir+f"/{x} {y} correlation.pdf",format='PDF')
    r_val = stats.pearsonr(toplt[x],toplt[y])[0]
    print(f"pearson's r = {r_val}")
    
    
# %% fit sigmoid - master
print("- Figure 5: Fin-body ratio")

all_coef = pd.DataFrame()
all_y = pd.DataFrame()
all_binned_average = pd.DataFrame()

for (cond_abla,cond_dpf,cond_ztime), for_fit in all_feature_cond.groupby(['condition','dpf','ztime']):
    expNum = for_fit['expNum'].max()
    jackknife_idx = jackknife_resampling(np.array(list(range(expNum+1))))
    for excluded_exp, idx_group in enumerate(jackknife_idx):
        coef, fitted_y, sigma = sigmoid_fit(
            for_fit.loc[for_fit['expNum'].isin(idx_group)], X_RANGE, func=sigfunc_4free
        )
        slope = coef.iloc[0,0]*(coef.iloc[0,3]) / 4
        fitted_y.columns = ['Attack angle','Rotation']
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
    binned_df, _ = distribution_binned_average(for_fit,by_col='rot_early',bin_col='atk_ang',bin=AVERAGE_BIN)
    binned_df.columns=['Rotation','atk_ang']
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
# plt.close()
defaultPlotting(size=12)
set_font_type()
plt.figure()
g = sns.relplot(x='Rotation',y='Attack angle', data=all_y, 
                kind='line',
                col='dpf', col_order=all_cond1,
                row = 'ztime', row_order=all_ztime,
                hue='condition', hue_order = all_cond2,ci='sd',
                )
for i , g_row in enumerate(g.axes):
    for j, ax in enumerate(g_row):
        sns.lineplot(data=all_binned_average.loc[
            (all_binned_average['dpf']==all_cond1[j]) & (all_binned_average['ztime']==all_ztime[i]),:
                ], 
                    x='Rotation', y='atk_ang', 
                    hue='condition',alpha=0.5,
                    ax=ax)
g.set_xlabels('Rotation (deg)')
g.set_ylabels('Attack angle (deg)')
filename = os.path.join(fig_dir,"Fin-body Coordination.pdf")
plt.savefig(filename,format='PDF')

# plt.show()

# %%
# plot slope
# plt.close()
defaultPlotting(size=12)
plt.figure()
p = sns.catplot(
    data = all_coef, y='slope',x='dpf',kind='point',join=False,
    col_order=all_cond1,ci='sd',
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
filename = os.path.join(fig_dir,"fin-body ratio.pdf")
plt.savefig(filename,format='PDF')

# %%
# if to plot other coefs

# defaultPlotting(size=12)
# for coef_name in ['k','xval','min','height','slope']:
#     plt.figure()
#     p = sns.catplot(
#         data = all_coef, y=coef_name,x='condition',kind='point',join=False,
#         col='dpf',col_order=all_cond1,
#         ci='sd',
#         row = 'ztime', row_order=all_ztime,
#         # units=excluded_exp,
#         hue='condition', dodge=True,
#         hue_order = all_cond2,
#         sharey=False,
#         aspect=.6,
#     )
#     p.map(sns.lineplot,'condition',coef_name,estimator=None,
#         units='excluded_exp',
#         hue='condition',
#         alpha=0.2,
#         data=all_coef)
#     sns.despine(offset=10)
#     filename = os.path.join(fig_dir,f"{coef_name} by cond1.pdf")
    
#     plt.savefig(filename,format='PDF')
# %%
# plot CI of slope
print("- Figure supp - CI width vs sample size - max slope of fin-body ratio")
list_of_sample_N = np.arange(1000,len(all_feature_cond),1000)
repeated_res = pd.DataFrame()
num_of_repeats = 20
rep = 0

while rep < num_of_repeats:
    list_of_ci_width = []
    for sample_N in list_of_sample_N:
        sample_for_fit = all_feature_cond.sample(n=sample_N)
        coef, fitted_y, sigma = sigmoid_fit(
            sample_for_fit, X_RANGE, func=sigfunc_4free
        )
        E_height = coef.iloc[0,3]
        E_k = coef.iloc[0,0]
        V_height = sigma[3]**2
        V_k = sigma[0]**2
        
        mean_formSample = E_k * E_height / 4
        
        slope_var = (V_height*V_k + V_height*(E_k**2) +  V_k*(E_height**2)) * (1/4)**2
        sigma_formSample = np.sqrt(slope_var)
        
        (ci_low, ci_high) = st.norm.interval(0.95, loc=mean_formSample, scale=sigma_formSample)
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
    y = 'CI width',
    ci='sd',
)
filename = os.path.join(ci_fig,"fin-body ratio slope CI width.pdf")
plt.savefig(filename,format='PDF')
# %%
# plot finless fish data
pick_data = 'finless'
which_ztime = 'day'
root, FRAME_RATE = get_data_dir(pick_data)

X_RANGE = np.arange(-2,8.01,0.01)
BIN_WIDTH = 0.4
AVERAGE_BIN = np.arange(min(X_RANGE),max(X_RANGE),BIN_WIDTH)
all_feature_finless, all_cond1, all_cond2 = get_bout_features(root, FRAME_RATE)
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