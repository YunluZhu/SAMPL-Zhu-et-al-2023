'''
find ways to illustrate righting rotation vs initial pitch
'''

#%%
import os
from plot_functions.plt_tools import round_half_up
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from astropy.stats import jackknife_resampling
from scipy.optimize import curve_fit
from plot_functions.get_data_dir import (get_data_dir,get_figure_dir)
from plot_functions.get_bout_features import get_bout_features
from plot_functions.plt_tools import (jackknife_mean, set_font_type, defaultPlotting, distribution_binned_average_nostd)
from plot_functions.get_bout_kinetics import get_bout_kinetics
from scipy import stats

set_font_type()
# defaultPlotting()v
# %%
pick_data = 'all_7dd'
which_ztime = 'day'
root, FRAME_RATE = get_data_dir(pick_data)

folder_name = f'{pick_data} righting correlation'

# fig_dir4 = os.path.join(get_figure_dir('Fig_4'), folder_name)
fig_dir = os.path.join(get_figure_dir('Fig_6'), folder_name)
DAY_RESAMPLE = 1000

# try:
#     os.makedirs(fig_dir4)
# except:
#     pass
try:
    os.makedirs(fig_dir)
except:
    pass

# %% get features
all_feature_cond, all_cond1, all_cond2 = get_bout_features(root, FRAME_RATE, ztime = which_ztime)
all_kinetic_cond, kinetics_jackknife, kinetics_bySpd_jackknife, all_cond1, all_cond2 = get_bout_kinetics(root, FRAME_RATE, ztime=which_ztime, sample=DAY_RESAMPLE)

all_feature_cond = all_feature_cond.sort_values(by=['condition','expNum']).reset_index(drop=True)
# all_feature_cond.drop(all_feature_cond[all_feature_cond['spd_peak']<7].index, inplace=True)
all_cond1.sort()
all_cond2.sort()
# %%
set_point = kinetics_jackknife['set_point_jack'].mean()
# find bouts that rotate past set point
bouts_passing_setPoint = all_feature_cond.query('(pitch_max_angvel - @set_point)*(pitch_initial - @set_point) < 0')
# bouts_passing_setPoint = bouts_passing_setPoint.loc[(bouts_passing_setPoint['pitch_initial'] - bouts_passing_setPoint['pitch_end']).abs() > 5]
bouts_passing_setPoint = bouts_passing_setPoint.assign(
    types = 'goes_up'
)
bouts_passing_setPoint.loc[bouts_passing_setPoint['pitch_initial']>bouts_passing_setPoint['pitch_max_angvel'],'types'] = 'goes_down'
bouts_passing_setPoint = bouts_passing_setPoint.assign(
    initialDeviationFromSet = np.abs(set_point  - bouts_passing_setPoint['pitch_initial']),
    absRightingRot = np.abs(bouts_passing_setPoint['rot_l_decel']),
    peakDeviationFromSet = np.abs(set_point  - bouts_passing_setPoint['pitch_peak']),
    maxAngvelDeviationFromSet = np.abs(set_point  - bouts_passing_setPoint['pitch_max_angvel']),
)

# %%
# WARNING - these plots don't take into consideration of different condition and dpf

x = 'pitch_max_angvel'
y = 'rot_l_decel'
upper = np.percentile(bouts_passing_setPoint[x], 99)
lower = np.percentile(bouts_passing_setPoint[x], 1)
BIN_NUM = 15
BIN_WIDTH = (upper - lower)/BIN_NUM
AVERAGE_BIN = np.arange(round_half_up(lower),round_half_up(upper),BIN_WIDTH)
binned_df_byPeak = bouts_passing_setPoint.groupby(['condition','dpf']).apply(
    lambda group: distribution_binned_average_nostd(group,by_col=x,bin_col=y,bin=AVERAGE_BIN)
)

binned_df_byPeak = binned_df_byPeak.reset_index(level=['condition','dpf'])
binned_df_byPeak = binned_df_byPeak.reset_index(drop=True)

g = sns.relplot(
    kind = 'line',
    data = binned_df_byPeak,
    x = x,
    y = y,
    col = 'condition',
    color='black',
    height=3,
)
g.map(
    sns.scatterplot,
    data = bouts_passing_setPoint,
    x = x,
    y = y,
    alpha=0.15,
    color='grey',
    style = 'types',
    markers = ['^','v'],
    s=60,
    linewidths = 0,
)
plt.axvline(set_point, -3,3)
g.axes[0,0].set_xlabel(f"{x} (deg)")
g.axes[0,0].set_ylabel(f"righting rot (deg)")

g.set(ylim=(-3,3))
g.set(xlim=(lower,upper))
sns.despine()
plt.savefig(os.path.join(fig_dir,f"bouts_passing_setpoint_{x}_{y}.pdf"),format='PDF')
r_val = stats.pearsonr(bouts_passing_setPoint[x],bouts_passing_setPoint[y])[0]
print(f"{x} vs {y} pearson's r = {r_val}")
# plt.show()



x = 'pitch_initial'
y = 'rot_l_decel'
upper = np.percentile(bouts_passing_setPoint[x], 99)
lower = np.percentile(bouts_passing_setPoint[x], 1)
AVERAGE_BIN = np.arange(round_half_up(lower),round_half_up(upper),BIN_WIDTH)
binned_df_byInitial = bouts_passing_setPoint.groupby(['condition','dpf']).apply(
    lambda group: distribution_binned_average_nostd(group,by_col=x,bin_col=y,bin=AVERAGE_BIN)
)
binned_df_byInitial = binned_df_byInitial.reset_index(level=['condition','dpf'])
binned_df_byInitial = binned_df_byInitial.reset_index(drop=True)


p = sns.relplot(
    kind = 'line',
    data = binned_df_byInitial,
    x = x,
    y = y,
    col = 'condition',
    color = 'black',
    height=3,
)
p.map(
    sns.scatterplot,
    data = bouts_passing_setPoint,
    x = x,
    y = y,
    alpha=0.15,
    color='grey',
    style = 'types',
    markers = ['^','v'],
    s=60,
    linewidths = 0,

)
plt.axvline(set_point, -3,3)
p.axes[0,0].set_xlabel(f"{x} (deg)")
p.axes[0,0].set_ylabel(f"righting rot (deg)")

p.set(ylim=(-3,3))
p.set(xlim=(lower,upper))
sns.despine()
plt.savefig(os.path.join(fig_dir,f"bouts_passing_setpoint_{x}_{y}.pdf"),format='PDF')
r_val = stats.pearsonr(bouts_passing_setPoint[x],bouts_passing_setPoint[y])[0]
print(f"{x} vs {y} pearson's r = {r_val}")
# plt.show()
# %%

# %%
# bouts_passing_setPoint = all_feature_cond.query('(pitch_initial - @set_point)/(pitch_max_angvel - @set_point) < 0.5')
# bouts_passing_setPoint = bouts_passing_setPoint.assign(
#     types = 'goes_up'
# )
# bouts_passing_setPoint.loc[bouts_passing_setPoint['pitch_initial']>bouts_passing_setPoint['pitch_peak'],'types'] = 'goes_down'
# bouts_passing_setPoint = bouts_passing_setPoint.assign(
#     initialDeviationFromSet = np.abs(set_point  - bouts_passing_setPoint['pitch_initial']),
#     absRightingRot = np.abs(bouts_passing_setPoint['rot_l_decel']),
#     peakDeviationFromSet = np.abs(set_point  - bouts_passing_setPoint['pitch_peak']),
#     maxAngvelDeviationFromSet = np.abs(set_point  - bouts_passing_setPoint['pitch_max_angvel']),

# )


# x = 'peakDeviationFromSet'
# y = 'absRightingRot'
# upper = np.percentile(bouts_passing_setPoint[x], 99)
# lower = np.percentile(bouts_passing_setPoint[x], 1)
# BIN_NUM = 12
# BIN_WIDTH = (upper - lower)/BIN_NUM
# AVERAGE_BIN = np.arange(round_half_up(lower),round_half_up(upper),BIN_WIDTH)
# binned_df_byPeak = bouts_passing_setPoint.groupby(['condition','dpf']).apply(
#     lambda group: distribution_binned_average_nostd(group,by_col=x,bin_col=y,bin=AVERAGE_BIN)
# )

# binned_df_byPeak = binned_df_byPeak.reset_index(level=['condition','dpf'])
# binned_df_byPeak = binned_df_byPeak.reset_index(drop=True)

# g = sns.relplot(
#     kind = 'line',
#     data = binned_df_byPeak,
#     x = x,
#     y = y,
#     col = 'condition',
#     color='black',
#     height=3,
# )
# g.map(
#     sns.scatterplot,
#     data = bouts_passing_setPoint,
#     x = x,
#     y = y,
#     alpha=0.15,
#     color='grey',
#     # style = 'types',
#     # markers = ['^','v'],
#     s=60,
#     linewidths = 0,
# )
# plt.axvline(set_point, -3,3)
# g.axes[0,0].set_xlabel(f"{x} (deg)")
# g.axes[0,0].set_ylabel(f"righting rot (deg)")

# # g.set(ylim=(-3,3))
# g.set(xlim=(lower,upper))
# sns.despine()
# plt.savefig(os.path.join(fig_dir,f"bouts_passing_setpoint_{x}_{y}.pdf"),format='PDF')
# r_val = stats.pearsonr(bouts_passing_setPoint[x],bouts_passing_setPoint[y])[0]
# print(f"{x} vs {y} pearson's r = {r_val}")
# plt.show()



# x = 'initialDeviationFromSet'
# y = 'absRightingRot'
# upper = np.percentile(bouts_passing_setPoint[x], 99)
# lower = np.percentile(bouts_passing_setPoint[x], 1)
# AVERAGE_BIN = np.arange(round_half_up(lower),round_half_up(upper),BIN_WIDTH)
# binned_df_byInitial = bouts_passing_setPoint.groupby(['condition','dpf']).apply(
#     lambda group: distribution_binned_average_nostd(group,by_col=x,bin_col=y,bin=AVERAGE_BIN)
# )
# binned_df_byInitial = binned_df_byInitial.reset_index(level=['condition','dpf'])
# binned_df_byInitial = binned_df_byInitial.reset_index(drop=True)


# p = sns.relplot(
#     kind = 'line',
#     data = binned_df_byInitial,
#     x = x,
#     y = y,
#     col = 'condition',
#     color = 'black',
#     height=3,
# )
# p.map(
#     sns.scatterplot,
#     data = bouts_passing_setPoint,
#     x = x,
#     y = y,
#     alpha=0.15,
#     color='grey',
#     # style = 'types',
#     # markers = ['^','v'],
#     s=60,
#     linewidths = 0,

# )
# plt.axvline(set_point, -3,3)
# p.axes[0,0].set_xlabel(f"{x} (deg)")
# p.axes[0,0].set_ylabel(f"righting rot (deg)")

# # p.set(ylim=(-3,3))
# p.set(xlim=(lower,upper))
# sns.despine()
# plt.savefig(os.path.join(fig_dir,f"bouts_passing_setpoint_{x}_{y}.pdf"),format='PDF')
# r_val = stats.pearsonr(bouts_passing_setPoint[x],bouts_passing_setPoint[y])[0]
# print(f"{x} vs {y} pearson's r = {r_val}")
# plt.show()
# # plt.show()



