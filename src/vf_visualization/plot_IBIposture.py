'''
Plot Inter Bout Interval (IBI or IEI) posture distribution and standard deviation

This scripts looks for "prop_Bout_IEI2" in the "prop_bout_IEI_pitch" data which includes mean of body angles during IEI

This script takes two types of directory:
1. if input directory is a folder containing analyzed dlm data, IBI posture of the current experiment will be plotted
    dir/
    ├── all_data.h5
    ├── bout_data.h5
    └── IEI_data.h5
2. if input directory is a folder with subfolders containing dlm data, IBI posture of all the experiments and jackknifed standard deviation will be plotted. 
    dir/
    ├── experiment 1/  
    │   ├── all_data.h5
    │   ├── bout_data.h5
    │   └── IEI_data.h5
    └── experiment 2/
        ├── all_data.h5
        ├── bout_data.h5
        └── IEI_data.h5
NOTE
User may define the number of bouts sampled from each experimental repeat for jackknifing by defining the argument "sample_bout"
Default is off (sample_bout = -1)'''

# %%
import os
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from astropy.stats import jackknife_resampling
from plot_functions.plt_tools import (set_font_type, day_night_split, defaultPlotting)

def plot_IBIposture(root, **kwargs):
    print('\n- Plotting inter-bout-interval posture')
    # generate figure folder
    folder_name = 'IBI posture'
    folder_dir = os.getcwd()
    fig_dir = os.path.join(folder_dir, 'figures', folder_name)
    
    if_sample = False
    SAMPLE_N = -1
    
    for key, value in kwargs.items():
        if key == 'sample_bout':
            SAMPLE_N = value
    if SAMPLE_N == -1:
        SAMPLE_N = int(input("How many bouts to sample from each dataset? ('0' for no sampling): "))
    if SAMPLE_N > 0:
        if_sample = True

    try:
        os.makedirs(fig_dir)
        print(f'fig directory created: {fig_dir}')
    except:
        print('Figure folder already exist! Old figures will be replaced.')
        print(fig_dir)

    bins = list(range(-90,94,4))

    # %%
    # main function

    # Check if there are subfolders in the current directory
    all_dir = [ele[0] for ele in os.walk(root)]
    if len(all_dir) > 1:
        # if yes, calculate jackknifed std(pitch)
        if_jackknife = True
        all_dir = all_dir[1:]
    else:
        # if no, only calculate one std(pitch) for current experiment
        if_jackknife = False
        
    # %%
    all_angles = pd.DataFrame()
    ang_std = []
    
    all_dir.sort()
    # go through each condition folders under the root
    for expNum, exp_path in enumerate(all_dir):
        # for each sub-folder, get the path
        df = pd.read_hdf(f"{exp_path}/IEI_data.h5", key='prop_bout_IEI2')
        df = day_night_split(df,'propBoutIEItime')

        # get pitch
        body_angles = df.loc[:,['propBoutIEI_pitch']].assign(exp_num = expNum)
        if if_jackknife == True:
            if if_sample == True:
                try:
                    body_angles = body_angles.sample(n=SAMPLE_N)
                except:
                    body_angles = body_angles.sample(n=SAMPLE_N,replace=True)
        all_angles = pd.concat([all_angles, body_angles],axis=0)

    if if_jackknife:
        # jackknife for the index
        jackknife_idx = jackknife_resampling(np.arange(0,expNum+1))
        # get the distribution of every jackknifed sample for the current condition
        ang_distribution = pd.concat(
            [pd.DataFrame(
                np.histogram(all_angles.loc[all_angles['exp_num'].isin(idx_group),'propBoutIEI_pitch'].to_numpy().flatten(), bins=bins, density=True)
                ) for idx_group in jackknife_idx], axis=1
            ).transpose()
        ang_std = [np.nanstd(all_angles.loc[all_angles['exp_num'].isin(idx_group),'propBoutIEI_pitch'].to_numpy().flatten()) for idx_group in jackknife_idx]
        ang_std = pd.DataFrame(data={'Std of posture':ang_std, 
                                    'exp jackknifed':np.arange(0,expNum+1)})
    else:
        ang_distribution = pd.DataFrame(np.histogram(all_angles['propBoutIEI_pitch'].to_numpy().flatten(), bins=bins, density=True)).T
        ang_std = np.nanstd(all_angles['propBoutIEI_pitch'])
        
    ang_distribution.columns = ['Probability','Posture (deg)']
    ang_distribution.reset_index(drop=True,inplace=True)

    
    # %%
    # Plot posture distribution and its standard deviation
    # defaultPlotting()
    # set_font_type()

    # plot distribution
    g = sns.lineplot(x='Posture (deg)',
                     y='Probability', 
                     data=ang_distribution, 
                     ci='sd', err_style='band')
    g.set_xticks(np.arange(-90,135,45))  # adjust x ticks
    filename = os.path.join(fig_dir, "inter bout interval pitch distribution.pdf")
    plt.savefig(filename,format='PDF')
    
    if if_jackknife:
        # plot jackknifed paired data
        # plot mean data
        # g = sns.pointplot(y='Std of posture',
        #                   data=ang_std, 
        #                   ci=95,
        #                     # linewidth=0,
        #                     alpha=0.9,
        #                     markers='d',
        #                 )
        g = sns.catplot(data = ang_std,y = 'Std of posture',
                        height=4, aspect=0.8, kind='point',
                        markers='d',sharey=False,
                        zorder=10
                        )
        g = sns.despine(trim=False)
        filename = os.path.join(fig_dir, "inter-bout interval std(pitch).pdf")
        plt.savefig(filename,format='PDF')
    else:
        print(f"Standard deviation of posture is: {ang_std}")    
    

if __name__ == "__main__":
    # if to use Command Line Inputs
    root = input("- Data directory? \n")
    plot_IBIposture(root)

# %%
