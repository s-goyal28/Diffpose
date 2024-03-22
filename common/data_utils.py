from __future__ import absolute_import, division

import numpy as np

from .camera import world_to_camera, normalize_screen_coordinates, image_coordinates, project_to_2d
from .utils import wrap

camera_dict = {
    '54138969': [2.2901, 2.2876, 0.0251, 0.0289],
    '55011271': [2.2994, 2.2952, 0.0177, 0.0161],
    '58860488': [2.2983, 2.2976, 0.0396, 0.0028],
    '60457274': [2.2910, 2.2895, 0.0299, 0.0018],
}


mapping = {'S1': {('1', '1'): '_ALL 1', ('1', '2'): '_ALL', ('2', '1'): 'Directions 1', ('2', '2'): 'Directions', ('3', '1'): 'Discussion 1', ('3', '2'): 'Discussion', ('4', '1'): 'Eating 2', ('4', '2'): 'Eating', ('5', '1'): 'Greeting 1', ('5', '2'): 'Greeting', ('6', '1'): 'Phoning 1', ('6', '2'): 'Phoning', ('7', '1'): 'Posing 1', ('7', '2'): 'Posing', ('8', '1'): 'Purchases 1', ('8', '2'): 'Purchases', ('9', '1'): 'Sitting 1', ('9', '2'): 'Sitting 2', ('10', '1'): 'SittingDown 2', ('10', '2'): 'SittingDown', ('11', '1'): 'Smoking 1', ('11', '2'): 'Smoking', ('12', '1'): 'TakingPhoto 1', ('12', '2'): 'TakingPhoto', ('13', '1'): 'Waiting 1', ('13', '2'): 'Waiting', ('14', '1'): 'Walking 1', ('14', '2'): 'Walking', ('15', '1'): 'WalkingDog 1', ('15', '2'): 'WalkingDog', ('16', '1'): 'WalkTogether 1', ('16', '2'): 'WalkTogether'}, 'S2': {('1', '1'): '_ALL 2', ('1', '2'): '_ALL 1', ('2', '1'): 'Directions 1', ('2', '2'): 'Directions', ('3', '1'): 'Discussion 1', ('3', '2'): 'Discussion', ('4', '1'): 'Eating 1', ('4', '2'): 'Eating 2', ('5', '1'): 'Greeting 1', ('5', '2'): 'Greeting', ('6', '1'): 'Phoning 1', ('6', '2'): 'Phoning', ('7', '1'): 'Posing 1', ('7', '2'): 'Posing', ('8', '1'): 'Purchases 1', ('8', '2'): 'Purchases', ('9', '1'): 'Sitting 1', ('9', '2'): 'Sitting', ('10', '1'): 'SittingDown 2', ('10', '2'): 'SittingDown 3', ('11', '1'): 'Smoking 1', ('11', '2'): 'Smoking', ('12', '1'): 'Photo 1', ('12', '2'): 'Photo', ('13', '1'): 'Waiting 1', ('13', '2'): 'Waiting', ('14', '1'): 'Walking 1', ('14', '2'): 'Walking', ('15', '1'): 'WalkDog 1', ('15', '2'): 'WalkDog', ('16', '1'): 'WalkTogether 1', ('16', '2'): 'WalkTogether'}, 'S3': {('1', '1'): '_ALL 1', ('1', '2'): '_ALL', ('2', '1'): 'Directions 1', ('2', '2'): 'Directions', ('3', '1'): 'Discussion 1', ('3', '2'): 'Discussion', ('4', '1'): 'Eating 1', ('4', '2'): 'Eating 2', ('5', '1'): 'Greeting 1', ('5', '2'): 'Greeting', ('6', '1'): 'Phoning 1', ('6', '2'): 'Phoning', ('7', '1'): 'Posing 1', ('7', '2'): 'Posing 2', ('8', '1'): 'Purchases 1', ('8', '2'): 'Purchases', ('9', '1'): 'Sitting 1', ('9', '2'): 'Sitting', ('10', '1'): 'SittingDown 1', ('10', '2'): 'SittingDown', ('11', '1'): 'Smoking 1', ('11', '2'): 'Smoking', ('12', '1'): 'Photo 1', ('12', '2'): 'Photo', ('13', '1'): 'Waiting 1', ('13', '2'): 'Waiting', ('14', '1'): 'Walking 1', ('14', '2'): 'Walking 2', ('15', '1'): 'WalkDog 1', ('15', '2'): 'WalkDog', ('16', '1'): 'WalkTogether 1', ('16', '2'): 'WalkTogether'}, 'S4': {('1', '1'): '_ALL 1', ('1', '2'): '_ALL', ('2', '1'): 'Directions 1', ('2', '2'): 'Directions', ('3', '1'): 'Discussion 1', ('3', '2'): 'Discussion', ('4', '1'): 'Eating 1', ('4', '2'): 'Eating', ('5', '1'): 'Greeting 1', ('5', '2'): 'Greeting', ('6', '1'): 'Phoning 1', ('6', '2'): 'Phoning', ('7', '1'): 'Posing 1', ('7', '2'): 'Posing', ('8', '1'): 'Purchases 1', ('8', '2'): 'Purchases', ('9', '1'): 'Sitting 1', ('9', '2'): 'Sitting', ('10', '1'): 'SittingDown 1', ('10', '2'): 'SittingDown 2', ('11', '1'): 'Smoking 1', ('11', '2'): 'Smoking', ('12', '1'): 'Photo 1', ('12', '2'): 'Photo', ('13', '1'): 'Waiting 1', ('13', '2'): 'Waiting', ('14', '1'): 'Walking 1', ('14', '2'): 'Walking', ('15', '1'): 'WalkDog 1', ('15', '2'): 'WalkDog', ('16', '1'): 'WalkTogether 2', ('16', '2'): 'WalkTogether 3'}, 'S5': {('1', '1'): '_ALL 1', ('1', '2'): '_ALL', ('2', '1'): 'Directions 1', ('2', '2'): 'Directions 2', ('3', '1'): 'Discussion 2', ('3', '2'): 'Discussion 3', ('4', '1'): 'Eating 1', ('4', '2'): 'Eating', ('5', '1'): 'Greeting 1', ('5', '2'): 'Greeting 2', ('6', '1'): 'Phoning 1', ('6', '2'): 'Phoning', ('7', '1'): 'Posing 1', ('7', '2'): 'Posing', ('8', '1'): 'Purchases 1', ('8', '2'): 'Purchases', ('9', '1'): 'Sitting 1', ('9', '2'): 'Sitting', ('10', '1'): 'SittingDown', ('10', '2'): 'SittingDown 1', ('11', '1'): 'Smoking 1', ('11', '2'): 'Smoking', ('12', '1'): 'Photo', ('12', '2'): 'Photo 2', ('13', '1'): 'Waiting 1', ('13', '2'): 'Waiting 2', ('14', '1'): 'Walking 1', ('14', '2'): 'Walking', ('15', '1'): 'WalkDog 1', ('15', '2'): 'WalkDog', ('16', '1'): 'WalkTogether 1', ('16', '2'): 'WalkTogether'}, 'S6': {('1', '1'): '_ALL 1', ('1', '2'): '_ALL', ('2', '1'): 'Directions 1', ('2', '2'): 'Directions', ('3', '1'): 'Discussion 1', ('3', '2'): 'Discussion', ('4', '1'): 'Eating 1', ('4', '2'): 'Eating 2', ('5', '1'): 'Greeting 1', ('5', '2'): 'Greeting', ('6', '1'): 'Phoning 1', ('6', '2'): 'Phoning', ('7', '1'): 'Posing 2', ('7', '2'): 'Posing', ('8', '1'): 'Purchases 1', ('8', '2'): 'Purchases', ('9', '1'): 'Sitting 1', ('9', '2'): 'Sitting 2', ('10', '1'): 'SittingDown 1', ('10', '2'): 'SittingDown', ('11', '1'): 'Smoking 1', ('11', '2'): 'Smoking', ('12', '1'): 'Photo', ('12', '2'): 'Photo 1', ('13', '1'): 'Waiting 3', ('13', '2'): 'Waiting', ('14', '1'): 'Walking 1', ('14', '2'): 'Walking', ('15', '1'): 'WalkDog 1', ('15', '2'): 'WalkDog', ('16', '1'): 'WalkTogether 1', ('16', '2'): 'WalkTogether'}, 'S7': {('1', '1'): '_ALL 1', ('1', '2'): '_ALL', ('2', '1'): 'Directions 1', ('2', '2'): 'Directions', ('3', '1'): 'Discussion 1', ('3', '2'): 'Discussion', ('4', '1'): 'Eating 1', ('4', '2'): 'Eating', ('5', '1'): 'Greeting 1', ('5', '2'): 'Greeting', ('6', '1'): 'Phoning 2', ('6', '2'): 'Phoning', ('7', '1'): 'Posing 1', ('7', '2'): 'Posing', ('8', '1'): 'Purchases 1', ('8', '2'): 'Purchases', ('9', '1'): 'Sitting 1', ('9', '2'): 'Sitting', ('10', '1'): 'SittingDown', ('10', '2'): 'SittingDown 1', ('11', '1'): 'Smoking 1', ('11', '2'): 'Smoking', ('12', '1'): 'Photo', ('12', '2'): 'Photo 1', ('13', '1'): 'Waiting 1', ('13', '2'): 'Waiting 2', ('14', '1'): 'Walking 1', ('14', '2'): 'Walking 2', ('15', '1'): 'WalkDog 1', ('15', '2'): 'WalkDog', ('16', '1'): 'WalkTogether 1', ('16', '2'): 'WalkTogether'}, 'S8': {('1', '1'): '_ALL 1', ('1', '2'): '_ALL', ('2', '1'): 'Directions 1', ('2', '2'): 'Directions', ('3', '1'): 'Discussion 1', ('3', '2'): 'Discussion', ('4', '1'): 'Eating 1', ('4', '2'): 'Eating', ('5', '1'): 'Greeting 1', ('5', '2'): 'Greeting', ('6', '1'): 'Phoning 1', ('6', '2'): 'Phoning', ('7', '1'): 'Posing 1', ('7', '2'): 'Posing', ('8', '1'): 'Purchases 1', ('8', '2'): 'Purchases', ('9', '1'): 'Sitting 1', ('9', '2'): 'Sitting', ('10', '1'): 'SittingDown', ('10', '2'): 'SittingDown 1', ('11', '1'): 'Smoking 1', ('11', '2'): 'Smoking', ('12', '1'): 'Photo 1', ('12', '2'): 'Photo', ('13', '1'): 'Waiting 1', ('13', '2'): 'Waiting', ('14', '1'): 'Walking 1', ('14', '2'): 'Walking', ('15', '1'): 'WalkDog 1', ('15', '2'): 'WalkDog', ('16', '1'): 'WalkTogether 1', ('16', '2'): 'WalkTogether 2'}, 'S9': {('1', '1'): '_ALL 1', ('1', '2'): '_ALL', ('2', '1'): 'Directions 1', ('2', '2'): 'Directions', ('3', '1'): 'Discussion 1', ('3', '2'): 'Discussion 2', ('4', '1'): 'Eating 1', ('4', '2'): 'Eating', ('5', '1'): 'Greeting 1', ('5', '2'): 'Greeting', ('6', '1'): 'Phoning 1', ('6', '2'): 'Phoning', ('7', '1'): 'Posing 1', ('7', '2'): 'Posing', ('8', '1'): 'Purchases 1', ('8', '2'): 'Purchases', ('9', '1'): 'Sitting 1', ('9', '2'): 'Sitting', ('10', '1'): 'SittingDown', ('10', '2'): 'SittingDown 1', ('11', '1'): 'Smoking 1', ('11', '2'): 'Smoking', ('12', '1'): 'Photo 1', ('12', '2'): 'Photo', ('13', '1'): 'Waiting 1', ('13', '2'): 'Waiting', ('14', '1'): 'Walking 1', ('14', '2'): 'Walking', ('15', '1'): 'WalkDog 1', ('15', '2'): 'WalkDog', ('16', '1'): 'WalkTogether 1', ('16', '2'): 'WalkTogether'}, 'S10': {('1', '1'): '_ALL 2', ('1', '2'): '_ALL 1', ('2', '1'): 'Directions 1', ('2', '2'): 'Directions', ('3', '1'): 'Discussion 1', ('3', '2'): 'Discussion 2', ('4', '1'): 'Eating 1', ('4', '2'): 'Eating', ('5', '1'): 'Greeting 1', ('5', '2'): 'Greeting', ('6', '1'): 'Phoning 1', ('6', '2'): 'Phoning', ('7', '1'): 'Posing 1', ('7', '2'): 'Posing', ('8', '1'): 'Purchases 1', ('8', '2'): 'Purchases', ('9', '1'): 'Sitting 1', ('9', '2'): 'Sitting', ('10', '1'): 'SittingDown', ('10', '2'): 'SittingDown 1', ('11', '1'): 'Smoking 2', ('11', '2'): 'Smoking', ('12', '1'): 'Photo 1', ('12', '2'): 'Photo', ('13', '1'): 'Waiting 1', ('13', '2'): 'Waiting', ('14', '1'): 'Walking 1', ('14', '2'): 'Walking', ('15', '1'): 'WalkDog 1', ('15', '2'): 'WalkDog', ('16', '1'): 'WalkTogether 1', ('16', '2'): 'WalkTogether'}, 'S11': {('1', '1'): '_ALL 1', ('1', '2'): '_ALL', ('2', '1'): 'Directions 1', ('2', '2'): 'Directions', ('3', '1'): 'Discussion 1', ('3', '2'): 'Discussion 2', ('4', '1'): 'Eating 1', ('4', '2'): 'Eating', ('5', '1'): 'Greeting 2', ('5', '2'): 'Greeting', ('6', '1'): 'Phoning 3', ('6', '2'): 'Phoning 2', ('7', '1'): 'Posing 1', ('7', '2'): 'Posing', ('8', '1'): 'Purchases 1', ('8', '2'): 'Purchases', ('9', '1'): 'Sitting 1', ('9', '2'): 'Sitting', ('10', '1'): 'SittingDown', ('10', '2'): 'SittingDown 1', ('11', '1'): 'Smoking 2', ('11', '2'): 'Smoking', ('12', '1'): 'Photo 1', ('12', '2'): 'Photo', ('13', '1'): 'Waiting 1', ('13', '2'): 'Waiting', ('14', '1'): 'Walking 1', ('14', '2'): 'Walking', ('15', '1'): 'WalkDog 1', ('15', '2'): 'WalkDog', ('16', '1'): 'WalkTogether 1', ('16', '2'): 'WalkTogether'}}


def read_3d_data(dataset):
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            anim = dataset[subject][action]

            positions_3d = []
            for cam in anim['cameras']:
                pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                # pos_3d[:, :] -= pos_3d[:, :1]  # Remove global offset
                positions_3d.append(pos_3d)
            anim['positions_3d'] = positions_3d

    return dataset


def equilize_aspect_ratio(min_y, min_x, max_y, max_x):
    current_aspect_ratio = (max_x - min_x) / (max_y - min_y)
    desired_aspect_ratio = 1.0
    
    # Adjust the bounding box to match the desired aspect ratio
    if current_aspect_ratio > desired_aspect_ratio:
        # Increase the height of the bounding box
        center_y = (min_y + max_y) / 2
        new_height = (max_x - min_x) / desired_aspect_ratio
        min_y = center_y - new_height / 2
        max_y = center_y + new_height / 2
    else:
        # Increase the width of the bounding box
        center_x = (min_x + max_x) / 2
        new_width = (max_y - min_y) * desired_aspect_ratio
        min_x = center_x - new_width / 2
        max_x = center_x + new_width / 2
        
    return (min_y, min_x, max_y, max_x)

def normalize_cordinates(X, min_y, min_x, max_y, max_x):
    return ((X - [min_x, min_y]) / [(max_x - min_x), (max_y - min_y)]) *2 -1


def read_3d_data_me(dataset):
    bb_pose = np.load("./data/bboxes-Human36M-GT.npy", allow_pickle=True).item()
    for subject in dataset.subjects():
        subject_mapping = {v: k for k, v in mapping[subject].items()}
        for action in dataset[subject].keys():

            # Action in bound_box_file
            if subject == 'S1' and 'Photo' in action:
                folder_action = action.split(' ')[0] + '-' + subject_mapping[action.replace('Photo', 'TakingPhoto')][1]
            elif subject == 'S1' and 'WalkDog' in action:
                folder_action = action.split(' ')[0] + '-' + subject_mapping[action.replace('WalkDog', 'WalkingDog')][1]
            else:
                folder_action = action.split(' ')[0] + '-' + subject_mapping[action][1]
            if 'WalkDog' in folder_action:
                folder_action = folder_action.replace('WalkDog', 'WalkingDog')
            elif 'Photo' in folder_action:
                folder_action = folder_action.replace('Photo', 'TakingPhoto')
            elif 'WalkTogether' in folder_action:
                folder_action = folder_action.replace('WalkTogether', 'WalkingTogether')


            anim = dataset[subject][action]

            positions_3d = []
            positions_2d = []
            camerad_para = []
            for cam in anim['cameras']:
                print(cam['id'])
                pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                # pos_3d[:, :] -= pos_3d[:, :1]  # Remove global offset
                positions_3d.append(pos_3d)
                camerad_para.append(camera_dict[cam['id']])

                pos_2d = wrap(project_to_2d, True, pos_3d, cam['intrinsic'])
                pos_2d_pixel_space = image_coordinates(pos_2d, w=cam['res_w'], h=cam['res_h'])

                # Rescale to bounding box
                for i, pose in enumerate(pos_2d_pixel_space) : # Iterate through frames
                    (top, left, bottom, right) = bb_pose[subject][folder_action][cam['id']][i]

                    pose[:, 0] -= left
                    pose[:, 1] -= top
                    
                    (min_y, min_x, max_y, max_x) = equilize_aspect_ratio(0, 0, bottom - top, right - left)

                    pose[:, :] = normalize_cordinates(pose, min_y, min_x, max_y, max_x)
                
                positions_2d.append(pos_2d_pixel_space.astype('float32'))
    
            anim['positions_3d'] = positions_3d
            anim['camerad_para'] = camerad_para
            anim['positions_2d'] = positions_2d

    return dataset

def read_3d_data_me_xyz(dataset):
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            anim = dataset[subject][action]

            positions_3d = []
            camerad_para = []
            for cam in anim['cameras']:
                pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                pos_3d[:, :] -= pos_3d[:, :1]  # Remove global offset
                positions_3d.append(pos_3d)
                camerad_para.append(camera_dict[cam['id']])
    
            anim['positions_3d'] = positions_3d
            anim['camerad_para'] = camerad_para

    return dataset

def create_2d_data(data_path, dataset):
    keypoints = np.load(data_path, allow_pickle=True)
    keypoints = keypoints['positions_2d'].item()

    bb_pose = np.load("./data/bboxes-Human36M-GT.npy", allow_pickle=True).item()

    ### GJ: adjust the length of 2d data ###
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            for cam_idx in range(len(keypoints[subject][action])):
                mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
                assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length
                if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                    keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]


    for subject in keypoints.keys():
        subject_mapping = {v: k for k, v in mapping[subject].items()}
        for action in keypoints[subject]:
            
            # Action in bound_box_file
            if subject == 'S1' and 'Photo' in action:
                folder_action = action.split(' ')[0] + '-' + subject_mapping[action.replace('Photo', 'TakingPhoto')][1]
            elif subject == 'S1' and 'WalkDog' in action:
                folder_action = action.split(' ')[0] + '-' + subject_mapping[action.replace('WalkDog', 'WalkingDog')][1]
            else:
                folder_action = action.split(' ')[0] + '-' + subject_mapping[action][1]
            if 'WalkDog' in folder_action:
                folder_action = folder_action.replace('WalkDog', 'WalkingDog')
            elif 'Photo' in folder_action:
                folder_action = folder_action.replace('Photo', 'TakingPhoto')
            elif 'WalkTogether' in folder_action:
                folder_action = folder_action.replace('WalkTogether', 'WalkingTogether')

            cameras = ['54138969', '55011271', '58860488', '60457274']
            for i, cam in enumerate(cameras):
                for f_idx, kps in enumerate(keypoints[subject][action][i]):
                    (top, left, bottom, right) = bb_pose[subject][folder_action][cam][f_idx]
                    kps[:, :, 1] -= left
                    kps[:, :, 2] -= top
                    
                    (min_y, min_x, max_y, max_x) = equilize_aspect_ratio(0, 0, bottom - top, right - left)
                    
                    kps[:, :, 1:3] = normalize_cordinates(kps[:, :, 1:3], min_y, min_x, max_y, max_x)
                    keypoints[subject][action][i][f_idx] = kps

            # for cam_idx, kps in enumerate(keypoints[subject][action]):
            #     # Normalize camera frame
            #     cam = dataset.cameras()[subject][cam_idx]
            #     kps[..., 1:3] = normalize_screen_coordinates(kps[..., 1:3], w=cam['res_w'], h=cam['res_h'])
            #     keypoints[subject][action][cam_idx] = kps

    return keypoints

def fetch(subjects, dataset, keypoints, action_filter=None, stride=1, parse_3d_poses=True):
    out_poses_3d = []
    out_poses_2d = []
    out_actions = []

    for subject in subjects:
        for action in keypoints[subject].keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    # if action.startswith(a):
                    if action.split(' ')[0] == a:
                        found = True
                        break
                if not found:
                    continue

            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)):  # Iterate across cameras
                out_poses_2d.append(poses_2d[i])
                out_actions.append([action.split(' ')[0]] * poses_2d[i].shape[0])

            if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_3d)):  # Iterate across cameras
                    out_poses_3d.append(poses_3d[i])

    if len(out_poses_3d) == 0:
        out_poses_3d = None

    if stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            out_actions[i] = out_actions[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]

    return out_poses_3d, out_poses_2d, out_actions


def fetch_me(subjects, dataset, keypoints, action_filter=None, stride=1, parse_2d_poses_gt=True):
    out_poses_2d_gt = []
    out_poses_2d = []
    out_actions = []
    out_camera_para = []
    
    for subject in subjects:
        for action in keypoints[subject].keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    # if action.startswith(a):
                    if action.split(' ')[0] == a:
                        found = True
                        break
                if not found:
                    continue

            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)):  # Iterate across cameras
                out_poses_2d.append(poses_2d[i])
                out_actions.append([action.split(' ')[0]] * poses_2d[i].shape[0])

            if parse_2d_poses_gt and 'positions_2d' in dataset[subject][action]:
                poses_2d_gt = dataset[subject][action]['positions_2d']
                camera_para = dataset[subject][action]['camerad_para']
                assert len(poses_2d_gt) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_2d_gt)):  # Iterate across cameras
                    out_poses_2d_gt.append(poses_2d_gt[i])
                    out_camera_para.append([camera_para[i]]* poses_2d_gt[i].shape[0])

    if len(out_poses_2d_gt) == 0:
        out_poses_2d_gt = None

    if stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            out_actions[i] = out_actions[i][::stride]
            if out_poses_2d_gt is not None:
                out_poses_2d_gt[i] = out_poses_2d_gt[i][::stride]
                out_camera_para[i] = out_poses_2d_gt[i][::stride]
                
    return out_poses_2d_gt, out_poses_2d, out_actions, out_camera_para