import os
from plot_IBIposture import plot_IBIposture
from plot_bout_timing import plot_bout_timing
from plot_kinematics import plot_kinematics
from plot_fin_body_coordination_byAngvelMax import plot_fin_body_coordination_byAngvelMax
from plot_parameters import plot_parameters
import matplotlib.pyplot as plt

def main(root, **kwargs):
    """Script to generate parameters reported in Table 2. Requires folder "behavior data/DD_4_7_14_dpf" to plot.

    Args:
        root (str): root directory containing multiple experiment/condition folders
        ---kwargs---
        sample_bout (int): number of bouts to sample from each experimental repeat. default is off
        figure_dir (str): directory to save figures. If not defined, figures will be saved to folder "figures"

    """
    sample = 0
    figure_dir = False
    
    for key, value in kwargs.items():
        if key == 'sample_bout':
            sample = value
        if key == "figure_dir":
            figure_dir = value
            
    for age in ["04dpf", "07dpf", "14dpf"]:
        dir_for_age = os.path.join(root,f"DD_{age}")
        fig_dir = os.path.join(figure_dir, age)
        print(f"----- Plotting {age} data -----")
        print(f"In {dir_for_age}")
        
        plt.close('all')
        plot_bout_timing(dir_for_age, sample_bout=sample, figure_dir=fig_dir)
        plt.close('all')
        plot_IBIposture(dir_for_age, sample_bout=sample, figure_dir=fig_dir)
        plt.close('all')
        plot_parameters(dir_for_age, figure_dir=fig_dir)
        plt.close('all')
        plot_kinematics(dir_for_age, sample_bout=sample, figure_dir=fig_dir)
        plt.close('all')
        plot_fin_body_coordination_byAngvelMax(dir_for_age, sample_bout=sample, figure_dir=fig_dir)
    
if __name__ == "__main__":
    cur_dir = os.getcwd()
    folder_dir = os.path.join(cur_dir, 'Manuscript figures','Table3')
    data_dir = input("- Data directory: where is data folder 'behavior data'? \n")
    root = os.path.join(data_dir,"DD_4_7_14_dpf")
    main(root, figure_dir = folder_dir)

        
# %%
