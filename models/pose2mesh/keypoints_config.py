"""
This file contains definitions of useful data stuctures and the paths
for the datasets and data files necessary to run the code.

Adapted from opensource project GraphCMR (https://github.com/nkolot/GraphCMR/) and Pose2Mesh (https://github.com/hongsukchoi/Pose2Mesh_RELEASE)

"""

from os.path import join
folder_path = 'models/'
JOINT_REGRESSOR_TRAIN_EXTRA = folder_path + 'data/J_regressor_extra.npy'
JOINT_REGRESSOR_H36M_correct = folder_path + 'data/J_regressor_h36m_correct.npy'

J24_NAME = ('R_Ankle', 'R_Knee', 'R_Hip', 'L_Hip', 'L_Knee', 'L_Ankle', 'R_Wrist', 'R_Elbow', 'R_Shoulder', 'L_Shoulder',
'L_Elbow','L_Wrist','Neck','Top_of_Head','Pelvis','Thorax','Spine','Jaw','Head','Nose','L_Eye','R_Eye','L_Ear','R_Ear')
H36M_J17_NAME = ( 'Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head',
                  'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')
COLLECTION_15_NAME = ( 'pelvis', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
    'neck', 'head', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist')
J24_TO_J14 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18]
H36M_J17_TO_J14 = [3, 2, 1, 4, 5, 6, 16, 15, 14, 11, 12, 13, 8, 10]
H36M_J17_TO_J15 = [0, 4, 1, 5, 2, 6, 3, 8, 10, 11, 14, 12, 15, 13, 16]

joints_15_dict = {
            'pelvis':0,
            'left_hip':1,
            'right_hip':2,
            'left_knee':3,
            'right_knee':4,
            'left_ankle':5,
            'right_ankle':6,
            'neck':7,
            'head':8,
            'left_shoulder':9,
            'right_shoulder':10,
            'left_elbow':11,
            'right_elbow':12,
            'left_wrist':13,
            'right_wrist':14
            }
flip_pairs_15 = ((2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13))
skeleton_15 = ((0, 7), (7, 8), (7, 9), (7, 10), (9, 11), (10, 12), (11, 13), (12, 14), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6))
