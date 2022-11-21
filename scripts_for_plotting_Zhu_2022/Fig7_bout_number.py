#%%
# import sys
from plot_functions.plt_tools import round_half_up
from plot_functions.plt_tools import (set_font_type, defaultPlotting)
from plot_functions.get_IBIangles import get_IBIangles
from plot_functions.get_bout_features import get_bout_features

def Fig7_bout_number(root):
    set_font_type()
    defaultPlotting()
    # for day night split
    which_zeitgeber = 'day' # day night all
    FRAME_RATE = 166
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
if __name__ == "__main__":
    # if to use Command Line Inputs
    root = input("- Data directory: where is folder 'DD_7dpf_bkg'? \n")
    Fig7_bout_number(root)