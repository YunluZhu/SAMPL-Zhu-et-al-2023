import os

def get_figure_dir(which_figure):
    folder_dir = os.getcwd()
    fig_dir = os.path.join(folder_dir, 'Manuscript figures',which_figure)
    return fig_dir

def get_data_dir(pick_data):
    # folder_dir = os.getcwd()
    # fig_dir = os.path.join(folder_dir, 'figures', 'Manuscript figures')
    if pick_data == 'all_7dd':
        root = "/Volumes/LabData/manuscript data/2022-09 VF method v2/behavior data/DD_7dpf_combined"
        fr = 166    
    elif pick_data == 'finless':
        root = "/Volumes/LabData/manuscript data/2022-09 VF method v2/behavior data/DD_finless"
        fr = 166  
    elif pick_data == '7dd_bkg':
        root = "/Volumes/LabData/manuscript data/2022-09 VF method v2/behavior data/DD_7dpf_bkg"
        fr = 166
    elif pick_data == 'tmp':
        root = input("Dir? ")
        fr = int(input("Frame rate? "))
        
    return root, fr