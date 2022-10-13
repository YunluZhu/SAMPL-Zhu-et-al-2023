from plot_IBIposture import plot_IBIposture
from plot_timeseries import (plot_raw, plot_aligned)
from plot_bout_timing import plot_bout_timing
from plot_kinetics import plot_kinetics
from plot_fin_body_coordination import plot_atk_ang_rotation
from plot_parameters import plot_parameters
import matplotlib.pyplot as plt

def main(root, sample):
    plt.close('all')
    plot_bout_timing(root, sample_bout=sample)
    
    plt.close('all')
    plot_IBIposture(root, sample_bout=sample)
    
    plt.close('all')
    plot_parameters(root)
    
    plt.close('all')
    plot_kinetics(root, sample_bout=sample)
    
    # Timeseries for aligned bouts may take long to plot for large dataset (>10GB)
    plt.close('all')
    plot_aligned(root)
    
    plt.close('all')
    plot_atk_ang_rotation(root, sample_bout=sample)
    
    plt.close('all')
    plot_raw(root)
    
if __name__ == "__main__":
    root_dir = input("- Which data to plot? \n")
    sample = input("- How many bouts to sample from each dataset? ('0' for no sampling): ")
    main(root_dir, sample)