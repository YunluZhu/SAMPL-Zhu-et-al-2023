'''
Plot distribution of parameters in Figure 2 and 3.
Plot parameter correlations for calculating kinetics in Figure 3.
'''

#%%
# import sys
import os
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.get_bout_features import get_bout_features
from plot_functions.get_bout_kinetics import get_kinetics
from plot_functions.get_IBIangles import get_IBIangles
from plot_functions.plt_tools import (jackknife_mean,set_font_type, defaultPlotting)
from tqdm import tqdm
import matplotlib as mpl
from scipy.stats import linregress

set_font_type()
mpl.rc('figure', max_open_warning = 0)

# %%
# Select data and create figure folder
pick_data = 'wt_7dd'
which_ztime = 'day'
root, FRAME_RATE = get_data_dir(pick_data)

folder_name = f'{pick_data} parameter distribution'
folder_dir1 = get_figure_dir('Fig_1')
fig_dir2 = os.path.join(folder_dir1, folder_name)

folder_name = f'{pick_data} kinetics'
folder_dir2 = get_figure_dir('Fig_2')
fig_dir3 = os.path.join(folder_dir2, folder_name)

try:
    os.makedirs(fig_dir2)
except:
    pass
try:
    os.makedirs(fig_dir3)
except:
    pass
# %% get features
all_feature_cond, all_cond1, all_cond2 = get_bout_features(root, FRAME_RATE, ztime=which_ztime)
all_ibi_cond, _, _  = get_IBIangles(root, FRAME_RATE, ztime=which_ztime)
# %% tidy data

all_feature_cond = all_feature_cond.sort_values(by=['condition','expNum']).reset_index(drop=True)
all_ibi_cond = all_ibi_cond.sort_values(by=['condition','expNum']).reset_index(drop=True)

all_ibi_cond = all_ibi_cond.assign(y_boutFreq=1/all_ibi_cond['propBoutIEI'])

all_feature_UD = all_feature_cond

# %%
# Plot parameter distribution
print("Figure 2: Distribution of parameters")
toplt = all_feature_UD
feature_to_plt = ['pitch_initial','pitch_end','spd_peak','rot_total','bout_traj','bout_displ']

for feature in feature_to_plt:
    # let's add unit
    if 'spd' in feature:
        xlabel = feature + " (mm*s^-1)"
    elif 'dis' in feature:
        xlabel = feature + " (mm)"
    else:
        xlabel = feature + " (deg)"
    
    upper = np.percentile(toplt[feature], 99.5)
    lower = np.percentile(toplt[feature], 0.5)
    
    plt.figure(figsize=(3,2))
    g = sns.histplot(data=toplt, x=feature, 
                        bins = 20, 
                        element="poly",
                        #  kde=True, 
                        stat="density",
                        pthresh=0.05,
                        binrange=(lower,upper),
                        color='grey'
                        )
    g.set_xlabel(xlabel)
    sns.despine()
    plt.savefig(fig_dir2+f"/{feature} distribution.pdf",format='PDF')
    plt.close()
    
# inter bout interval data
toplt = all_ibi_cond
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
                        stat="density",
                        pthresh=0.05,
                        binrange=(lower,upper),
                        color='grey'
                        )
    g.set_xlabel(xlabel)
    sns.despine()
    plt.savefig(fig_dir2+f"/{feature_toplt} distribution.pdf",format='PDF')
    plt.close()

print("Figure 3: Distribution of pre_bout rotation and attack angle")
feature_to_plt = ['rot_pre_bout','atk_ang']
toplt = all_feature_UD

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
    
    g = sns.histplot(data=toplt, x=feature, 
                        bins = 20, 
                        element="poly",
                        #  kde=True, 
                        stat="density",
                        pthresh=0.05,
                        binrange=(lower,upper),
                        color='grey'
                        )
    g.set_xlabel(xlabel)
    sns.despine()
    plt.savefig(fig_dir3+f"/{feature} distribution.pdf",format='PDF')
    plt.close()
# %%
# parameter distribution
toplt = all_feature_UD

def linReg_sampleSatter_plot(data,xcol,ycol,xmin,xmax,color):
    xdata = data[xcol] 
    ydata = data[ycol]
    model_par = linregress(xdata, ydata)
    slope, intercept, r_value, p_value, std_err = model_par
    x = np.linspace(xmin,xmax,100)
    y = slope*x+intercept
    plt.figure(figsize=(4,4))
    g = sns.scatterplot(x=xcol, 
                        y=ycol, 
                        data=data.sample(2000), 
                        # marker='+',
                        alpha = 0.1,
                        color='grey',
                        edgecolor="none",
                        )
    plt.plot(x, y, color=color)
    return g, slope, intercept, r_value, p_value, std_err

# %%
print("Figure 3: Linear regression for kinetics")
# steering
print("- Steering Gain")
xcol = 'pitch_peak'
ycol = 'traj_peak'
xmin = -30
xmax = 50
color = 'darkgreen'
g, slope, intercept, r_value, p_value, std_err = linReg_sampleSatter_plot(toplt,xcol,ycol,xmin,xmax,color)

print(f"Pearson's correlation coefficient = {r_value}")
print(f"Slope = {slope}")
print(f"Steering gain = {slope}")

g.set_xlabel(xcol+' (deg)')
g.set_ylabel(ycol+' (deg)')
g.set(
    ylim=(-30,50),
    xlim=(xmin,xmax)
    )
plt.savefig(fig_dir3+"/Steering fit.pdf",format='PDF')
plt.close()

# righting
print("- Righting Gain")
xcol = 'pitch_pre_bout'
ycol = 'rot_l_decel'
xmin = -30
xmax = 50
color = 'darkred'
g, slope, intercept, r_value, p_value, std_err = linReg_sampleSatter_plot(toplt,xcol,ycol,xmin,xmax,color)

print(f"Pearson's correlation coefficient = {r_value}")
print(f"Slope = {slope}")
print(f"Righting gain = {-1*slope}")

g.set_xlabel(xcol+' (deg)')
g.set_ylabel(ycol+' (deg)')
g.set(
    ylim=(-6,10),
    xlim=(xmin,xmax)
    )
plt.savefig(fig_dir3+"/Righting fit.pdf",format='PDF')
plt.close()

# set point
print("- Set Point")
xcol = 'pitch_pre_bout'
ycol = 'rot_l_decel'
xmin = -30
xmax = 50
color = 'black'
g, slope, intercept, r_value, p_value, std_err = linReg_sampleSatter_plot(toplt,xcol,ycol,xmin,xmax,color)

x_intercept = -1*intercept/slope
print(f"Pearson's correlation coefficient = {r_value}")
print(f"Set point: {x_intercept}")

g.set_xlabel(xcol+' (deg)')
g.set_ylabel(ycol+' (deg)')
g.set(
    ylim=(-6,10),
    xlim=(xmin,xmax)
    )
plt.hlines(y=0,xmin=-30,xmax=x_intercept,colors='darkgrey')
plt.vlines(x=x_intercept,ymin=-6,ymax=0,colors='blue')
plt.savefig(fig_dir3+"/Set point.pdf",format='PDF')
plt.close()

# %%
