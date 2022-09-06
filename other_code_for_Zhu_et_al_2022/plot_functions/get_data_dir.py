import os

def get_figure_dir(which_figure):
    folder_dir = os.getcwd()
    fig_dir = os.path.join(folder_dir, 'Manuscript figures',which_figure)
    return fig_dir

def get_data_dir(pick_data):
    # folder_dir = os.getcwd()
    # fig_dir = os.path.join(folder_dir, 'figures', 'Manuscript figures')
    if pick_data == 'tan':
        root = "/Volumes/LabData/manuscript data/2022-07 VF method/behavior data/partial_data_for_ana/TAN_lesion"
        fr = 40    
    elif pick_data == 'otog':
        root = "/Volumes/LabData/manuscript data/2022-07 VF method/behavior data/partial_data_for_ana/otog"
        fr = 166
    elif pick_data == '7dd':
        root = "/Volumes/LabData/manuscript data/2022-07 VF method/behavior data/partial_data_for_ana/DD_07dpf"
        fr = 166  
    elif pick_data == 'wt_7dd':
        root = "/Volumes/LabData/manuscript data/2022-07 VF method/behavior data/WT_DD_7dpf"
        fr = 166
    elif pick_data == 'tmp':
        root = input("Dir? ")
        fr = int(input("Frame rate? "))
        
    return root, fr