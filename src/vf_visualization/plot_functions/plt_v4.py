import pandas as pd # pandas library
import numpy as np # numpy
from plot_functions.plt_tools import jackknife_list


def extract_bout_features_v4(bout_data,PEAK_IDX, FRAME_RATE):
    """_summary_

    Args:
        bout_data (dataFrame): bout data read from ('bout_data.h5', key='prop_bout_aligned')
        PEAK_IDX (numeric): index of the frame at time of peak speed
        FRAME_RATE (int): frame rate

    Returns:
        this_exp_features: extracted bout features
    """
    T_INITIAL = -0.25 #s
    T_PRE_BOUT = -0.10 #s
    T_POST_BOUT = 0.1 #s
    T_END = 0.2
    T_MID_ACCEL = -0.05
    T_MID_DECEL = 0.05
    
    idx_initial = int(PEAK_IDX + T_INITIAL * FRAME_RATE)
    idx_pre_bout = int(PEAK_IDX + T_PRE_BOUT * FRAME_RATE)
    idx_post_bout = int(PEAK_IDX + T_POST_BOUT * FRAME_RATE)
    idx_mid_accel = int(PEAK_IDX + T_MID_ACCEL * FRAME_RATE)
    idx_mid_decel = int(PEAK_IDX + T_MID_DECEL * FRAME_RATE)
    idx_end = int(PEAK_IDX + T_END * FRAME_RATE)
    
    this_exp_features = pd.DataFrame(data={
        'pitch_initial':bout_data.loc[bout_data['idx']==idx_initial,'propBoutAligned_pitch'].values, 
        'pitch_pre_bout':bout_data.loc[bout_data['idx']==idx_pre_bout,'propBoutAligned_pitch'].values, 
        'pitch_peak':bout_data.loc[bout_data['idx']==PEAK_IDX,'propBoutAligned_pitch'].values, 
        'pitch_post_bout':bout_data.loc[bout_data['idx']==idx_post_bout,'propBoutAligned_pitch'].values, 
        'pitch_end': bout_data.loc[bout_data['idx']==idx_end,'propBoutAligned_pitch'].values, 

        # 'traj_initial':bout_data.loc[bout_data['idx']==idx_initial,'propBoutAligned_instHeading'].values, 
        # 'traj_pre_bout':bout_data.loc[bout_data['idx']==idx_pre_bout,'propBoutAligned_instHeading'].values, 
        'traj_peak':bout_data.loc[bout_data['idx']==PEAK_IDX,'propBoutAligned_instHeading'].values, 
        # 'traj_post_bout':bout_data.loc[bout_data['idx']==idx_post_bout,'propBoutAligned_instHeading'].values, 
        # 'traj_end':bout_data.loc[bout_data['idx']==idx_end,'propBoutAligned_instHeading'].values, 

        'spd_peak':bout_data.loc[bout_data['idx']==PEAK_IDX,'propBoutAligned_speed'].values, 
        # 'spd_mid_decel':bout_data.loc[bout_data['idx']==idx_mid_accel,'propBoutAligned_speed'].values, 
        # 'bout_num':bout_data.loc[bout_data['idx']==PEAK_IDX,'bout_num'].values, 
        
    })
    # calculate attack angles
    # bout trajectory is the same as (bout_data.h5, key='prop_bout2')['epochBouts_trajectory']
    yy = (bout_data.loc[bout_data['idx']==idx_post_bout,'propBoutAligned_y'].values - bout_data.loc[bout_data['idx']==idx_pre_bout,'propBoutAligned_y'].values)
    absxx = np.absolute((bout_data.loc[bout_data['idx']==idx_post_bout,'propBoutAligned_x'].values - bout_data.loc[bout_data['idx']==idx_pre_bout,'propBoutAligned_x'].values))
    epochBouts_trajectory = np.degrees(np.arctan(yy/absxx)) # direction of the bout, -90:90
    
    displ = np.sqrt(np.square(yy) + np.square(absxx))
    pitch_mid_accel = bout_data.loc[bout_data['idx']==idx_mid_accel,'propBoutAligned_pitch'].reset_index(drop=True)
    pitch_mid_decel = bout_data.loc[bout_data['idx']==idx_mid_decel,'propBoutAligned_pitch'].reset_index(drop=True)
    
    this_exp_features = this_exp_features.assign(rot_total=this_exp_features['pitch_post_bout']-this_exp_features['pitch_initial'],
                                                rot_pre_bout=this_exp_features['pitch_pre_bout']-this_exp_features['pitch_initial'],
                                                rot_l_accel=this_exp_features['pitch_peak']-this_exp_features['pitch_pre_bout'],
                                                rot_l_decel=this_exp_features['pitch_post_bout']-this_exp_features['pitch_peak'],
                                                rot_early_accel = pitch_mid_accel-this_exp_features['pitch_pre_bout'],
                                                rot_late_accel = this_exp_features['pitch_peak'] - pitch_mid_accel,
                                                rot_early_decel = pitch_mid_decel-this_exp_features['pitch_peak'],
                                                rot_late_decel = this_exp_features['pitch_post_bout'] - pitch_mid_decel,
                                                bout_traj = epochBouts_trajectory,
                                                bout_displ = displ,
                                                atk_ang = epochBouts_trajectory - this_exp_features['pitch_pre_bout'],
                                                )  
    return this_exp_features


def get_kinetics(df):
    righting_fit = np.polyfit(x=df['pitch_pre_bout'], y=df['rot_l_decel'], deg=1)
    steering_fit = np.polyfit(x=df['pitch_peak'], y=df['traj_peak'], deg=1)
    set_point = np.polyfit(x=df['rot_l_decel'], y=df['pitch_pre_bout'], deg=1)
    kinetics = pd.Series(data={
        'righting_gain': -1 * righting_fit[0],
        'steering_gain': steering_fit[0],
        'set_point':set_point[1],
    })
    return kinetics


def jackknife_kinetics(df,col):
    """_summary_

    Args:
        df (datFrame): dataframe with bout features
        col (string): name of the column for jackknife calculation
    Returns:
        output: jackknife'd kinetics
    """
    exp_df = df.groupby(col).size()
    jackknife_exp_matrix = jackknife_list(list(exp_df.index))
    output = pd.DataFrame()
    for j, exp_group in enumerate(jackknife_exp_matrix):
        this_group_data = df.loc[df[col].isin(exp_group),:]
        this_group_kinetics = get_kinetics(this_group_data)
        this_group_kinetics = this_group_kinetics.append(pd.Series(data={
            'jackknife_group':j
        }))
        output = pd.concat([output,this_group_kinetics],axis=1)
    output = output.T.reset_index(drop=True)
    return output

