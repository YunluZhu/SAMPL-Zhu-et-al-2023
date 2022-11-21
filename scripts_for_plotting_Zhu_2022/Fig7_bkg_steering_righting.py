#%%
import os
import seaborn as sns
import matplotlib.pyplot as plt
from plot_functions.get_data_dir import ( get_figure_dir)
from plot_functions.plt_tools import (set_font_type, defaultPlotting)
from plot_functions.get_bout_kinetics import get_bout_kinetics

def Fig7_bkg_steering_righting(root):
    set_font_type()
    defaultPlotting()
    # %%
    # for day night split
    which_zeitgeber = 'day' # day night all
    SAMPLE_NUM = 1000
    print("- Figure 7: ZF strains - Steering & Righting")

    FRAME_RATE = 166
    folder_name = f'Steering and Righting'
    folder_dir = get_figure_dir('Fig_7')
    fig_dir = os.path.join(folder_dir, folder_name)

    try:
        os.makedirs(fig_dir)
        print(f'fig folder created:{folder_name}')
    except:
        pass


    # %%
    all_kinetic_cond, kinetics_jackknife, kinetics_bySpd_jackknife, all_cond1, all_cond2 = get_bout_kinetics(root, FRAME_RATE, ztime=which_zeitgeber, sample=SAMPLE_NUM)
    all_cond1.sort()
    all_cond2.sort()

    # %% Compare by condition
    toplt = kinetics_jackknife
    cat_cols = ['jackknife_group','condition','expNum','dpf','ztime']
    all_features = [c for c in toplt.columns if c not in cat_cols]
    # print('plot jackknife data')

    for feature_toplt in (all_features):
        p = sns.catplot(
            data = toplt, y=feature_toplt,x='condition',kind='strip',
            color = 'grey',
            edgecolor = None,
            linewidth = 0,
            s=8, 
            alpha=0.3,
            height=3,
            aspect=1,
        )
        p.map(sns.pointplot,'condition',feature_toplt,
            markers=['d','d','d'],
            order=all_cond2,
            join=False, 
            
            color='black',
            data=toplt)
        filename = os.path.join(fig_dir,f"{feature_toplt} .pdf")
        plt.savefig(filename,format='PDF')
        
# %%
if __name__ == "__main__":
    # if to use Command Line Inputs
    root = input("- Data directory: where is folder 'DD_7dpf_bkg'? \n")
    Fig7_bkg_steering_righting(root)