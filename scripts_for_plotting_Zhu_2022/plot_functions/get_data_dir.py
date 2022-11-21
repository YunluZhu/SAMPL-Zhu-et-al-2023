import os

def get_figure_dir(which_figure):
    folder_dir = os.getcwd()
    fig_dir = os.path.join(folder_dir, 'Manuscript figures',which_figure)
    return fig_dir

def get_data_dir(pick_data):
    data_directory_dict = {
        '7dd_combined': '/Volumes/LabData/manuscript data/2022-09 VF method v2/behavior data/DD_7dpf_combined', # paste the directory for folder "DD_7dpf_combined"
        'finless': '/Volumes/LabData/manuscript data/2022-09 VF method v2/behavior data/DD_finless', # paste the directory for folder "DD_finless"
        '7dd_bkg': '/Volumes/LabData/manuscript data/2022-09 VF method v2/behavior data/DD_7dpf_bkg', # paste the directory for folder "DD_7dpf_bkg"
    }
    root = data_directory_dict[pick_data]
    fr = 166
    return root, fr
