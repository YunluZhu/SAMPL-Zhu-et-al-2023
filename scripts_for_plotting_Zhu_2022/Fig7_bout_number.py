'''
Plot jackknifed kinetics

righting gain
set point
steering gain
correlation of accel & decel rotation
angvel gain (new)

zeitgeber time? Yes
jackknife? Yes
resampled? No
'''

#%%
# import sys
from plot_functions.plt_tools import round_half_up
import os,glob
# import time
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.plt_tools import (set_font_type, defaultPlotting, day_night_split)
from plot_functions.get_index import get_index
from plot_functions.get_bout_kinetics import get_bout_kinetics
from plot_functions.get_IBIangles import get_IBIangles
from plot_functions.get_bout_features import get_bout_features

set_font_type()
defaultPlotting()
# %%
pick_data = '7dd_bkg' # all or specific data
# for day night split
which_zeitgeber = 'day' # day night all
SAMPLE_NUM = 1000
# %%
root, FRAME_RATE = get_data_dir(pick_data)
peak_idx , total_aligned = get_index(FRAME_RATE)

root, FRAME_RATE = get_data_dir(pick_data)

# folder_name = f'{pick_data} Steering and Righting'
# folder_dir = get_figure_dir('Fig_7')
# fig_dir = os.path.join(folder_dir, folder_name)

# try:
#     os.makedirs(fig_dir)
#     print(f'fig folder created:{folder_name}')
# except:
#     print('fig folder already exist')


# %%
# all_kinetic_cond, kinetics_jackknife, kinetics_bySpd_jackknife, all_cond1, all_cond2 = get_bout_kinetics(root, FRAME_RATE, ztime=which_zeitgeber, sample=SAMPLE_NUM)
IBI_angles, cond1, cond2 = get_IBIangles(root, FRAME_RATE, ztime=which_zeitgeber)
all_feature_cond, all_cond1, all_cond2 = get_bout_features(root, FRAME_RATE, ztime = which_zeitgeber )#, max_angvel_time = max_angvel_time)

# %%
print("Bout number:")
print(all_feature_cond.groupby('condition').size())
print("\nIBI number:")
print(IBI_angles.groupby('condition').size())
# %%
