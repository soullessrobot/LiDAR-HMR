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
kps15_flip_pairs = ((2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13))
kps15_bone = ((0, 7), (7, 8), (7, 9), (7, 10), (9, 11), (10, 12), (11, 13), (12, 14), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6))
kps15_bone_ratio = ((4,6), (5,7), (10,12), (11,13), (4,10), (5,11), (6,12), (7,13))

# kps15_flip_pairs = ((2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13))
# kps15_bone = ((0, 7), (7, 8), (7, 9), (7, 10), (9, 11), (10, 12), (11, 13), (12, 14), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6))
# kps15_bone_ratio = ((4,6), (5,7), (10,12), (11,13), (4,10), (5,11), (6,12), (7,13))
