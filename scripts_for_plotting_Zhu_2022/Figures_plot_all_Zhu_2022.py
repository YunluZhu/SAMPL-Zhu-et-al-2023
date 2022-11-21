from Fig2_parameter_distribution_prob import Fig2_parameter_distribution_prob
from Fig2_parameters_alignedAverage import Fig2_parameters_alignedAverage
from Fig2_throughput import Fig2_throughput
from Fig3_bout_timing import Fig3_bout_timing
from Fig4_steering_fit import Fig4_steering_fit
from Fig4_trajDeviation_pitchChg import Fig4_trajDeviation_pitchChg
from Fig5_fin_body_coordination import Fig5_fin_body_coordination
from Fig5_time_of_maxAngvel import Fig5_time_of_maxAngvel
from Fig6_pitch_timeseries import Fig6_pitch_timeseries
from Fig6_righting_fit import Fig6_righting_fit
from Fig7_bkg_IBI import Fig7_bkg_IBI
from Fig7_bkg_timing import Fig7_bkg_timing
from Fig7_bkg_steering_righting import Fig7_bkg_steering_righting
from Fig7_bkg_fin_body import Fig7_bkg_fin_body
from Fig7_bout_number import Fig7_bout_number
from Fig8_CI_width import Fig8_CI_width
from Fig8_modeling_stats import Fig8_modeling_stats
from Fig8_sims_effectSize import Fig8_sims_effectSize
import matplotlib.pyplot as plt

def data_dir():
    root_7dd_combined = input("- Getting data directory: where is folder 'DD_7dpf_combined'? \n")
    root_7dd_finless = input("- Getting data directory: where is folder 'DD_finless'? \n")
    root_7dd_bkg = input("- Getting data directory: where is folder 'DD_7dpf_bkg'? \n")
    data_directory_dict = {
        '7dd_combined': root_7dd_combined, # directory for folder "DD_7dpf_combined"
        'finless': root_7dd_finless, # directory for folder "DD_finless"
        '7dd_bkg': root_7dd_bkg, # directory for folder "DD_7dpf_bkg"
    }
    return data_directory_dict



root_dict = {
        '7dd_combined': '/Volumes/LabData/manuscript data/2022-09 VF method v2/behavior data/DD_7dpf_combined', # directory for folder "DD_7dpf_combined"
        'finless': '/Volumes/LabData/manuscript data/2022-09 VF method v2/behavior data/DD_finless', # directory for folder "DD_finless"
        '7dd_bkg': '/Volumes/LabData/manuscript data/2022-09 VF method v2/behavior data/DD_7dpf_bkg', # directory for folder "DD_7dpf_bkg"
    }

# root_dict = data_dir()

# ––– Figure 2: swim parameter distribution and timeseries

# Fig2_parameter_distribution_prob(root_dict['7dd_combined'])
# plt.close('all')
# Fig2_parameters_alignedAverage(root_dict['7dd_combined'])
# plt.close('all')
# Fig2_throughput(root_dict['7dd_combined'])
# plt.close('all')

# ––– Figure 3: bout timing and sensitivity

# Fig3_bout_timing(root_dict['7dd_combined'])
# plt.close('all')

# ––– Figure 4: steering gain

# Fig4_trajDeviation_pitchChg(root_dict['7dd_combined'])
# plt.close('all')
# Fig4_steering_fit(root_dict['7dd_combined'])
# plt.close('all')

# ––– Figure 5: fin-body coordination

# Fig5_time_of_maxAngvel(root_dict['7dd_combined'])
# plt.close('all')
# Fig5_fin_body_coordination(root_dict['7dd_combined'], root_dict['finless'])
# plt.close('all')

# ––– Figure 6: righting gain

# Fig6_pitch_timeseries(root_dict['7dd_combined'])
# plt.close('all')
# Fig6_righting_fit(root_dict['7dd_combined'])
# plt.close('all')

# ––– Figure 7: measurements of different zebrafish strains

# Fig7_bkg_IBI(root_dict['7dd_bkg'])
# plt.close('all')
# Fig7_bkg_timing(root_dict['7dd_bkg'])
# plt.close('all')
# Fig7_bkg_steering_righting(root_dict['7dd_bkg'])
# plt.close('all')
Fig7_bkg_fin_body(root_dict['7dd_bkg'])
plt.close('all')

# ––– Figure 8: statistics of kinetic regression
# NOTE slow to plot

Fig8_CI_width(root_dict['7dd_combined'])
plt.close('all')
Fig8_modeling_stats(root_dict['7dd_combined'])
plt.close('all')
Fig8_sims_effectSize(root_dict['7dd_combined'])
plt.close('all')


    


