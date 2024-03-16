from __future__ import absolute_import, division
import os
import subprocess

import numpy as np

from .camera import world_to_camera, normalize_screen_coordinates

camera_dict = {
    '54138969': [2.2901, 2.2876, 0.0251, 0.0289],
    '55011271': [2.2994, 2.2952, 0.0177, 0.0161],
    '58860488': [2.2983, 2.2976, 0.0396, 0.0028],
    '60457274': [2.2910, 2.2895, 0.0299, 0.0018],
}


mapping = {'S1': {('1', '1'): '_ALL 1', ('1', '2'): '_ALL', ('2', '1'): 'Directions 1', ('2', '2'): 'Directions', ('3', '1'): 'Discussion 1', ('3', '2'): 'Discussion', ('4', '1'): 'Eating 2', ('4', '2'): 'Eating', ('5', '1'): 'Greeting 1', ('5', '2'): 'Greeting', ('6', '1'): 'Phoning 1', ('6', '2'): 'Phoning', ('7', '1'): 'Posing 1', ('7', '2'): 'Posing', ('8', '1'): 'Purchases 1', ('8', '2'): 'Purchases', ('9', '1'): 'Sitting 1', ('9', '2'): 'Sitting 2', ('10', '1'): 'SittingDown 2', ('10', '2'): 'SittingDown', ('11', '1'): 'Smoking 1', ('11', '2'): 'Smoking', ('12', '1'): 'TakingPhoto 1', ('12', '2'): 'TakingPhoto', ('13', '1'): 'Waiting 1', ('13', '2'): 'Waiting', ('14', '1'): 'Walking 1', ('14', '2'): 'Walking', ('15', '1'): 'WalkingDog 1', ('15', '2'): 'WalkingDog', ('16', '1'): 'WalkTogether 1', ('16', '2'): 'WalkTogether'}, 'S2': {('1', '1'): '_ALL 2', ('1', '2'): '_ALL 1', ('2', '1'): 'Directions 1', ('2', '2'): 'Directions', ('3', '1'): 'Discussion 1', ('3', '2'): 'Discussion', ('4', '1'): 'Eating 1', ('4', '2'): 'Eating 2', ('5', '1'): 'Greeting 1', ('5', '2'): 'Greeting', ('6', '1'): 'Phoning 1', ('6', '2'): 'Phoning', ('7', '1'): 'Posing 1', ('7', '2'): 'Posing', ('8', '1'): 'Purchases 1', ('8', '2'): 'Purchases', ('9', '1'): 'Sitting 1', ('9', '2'): 'Sitting', ('10', '1'): 'SittingDown 2', ('10', '2'): 'SittingDown 3', ('11', '1'): 'Smoking 1', ('11', '2'): 'Smoking', ('12', '1'): 'Photo 1', ('12', '2'): 'Photo', ('13', '1'): 'Waiting 1', ('13', '2'): 'Waiting', ('14', '1'): 'Walking 1', ('14', '2'): 'Walking', ('15', '1'): 'WalkDog 1', ('15', '2'): 'WalkDog', ('16', '1'): 'WalkTogether 1', ('16', '2'): 'WalkTogether'}, 'S3': {('1', '1'): '_ALL 1', ('1', '2'): '_ALL', ('2', '1'): 'Directions 1', ('2', '2'): 'Directions', ('3', '1'): 'Discussion 1', ('3', '2'): 'Discussion', ('4', '1'): 'Eating 1', ('4', '2'): 'Eating 2', ('5', '1'): 'Greeting 1', ('5', '2'): 'Greeting', ('6', '1'): 'Phoning 1', ('6', '2'): 'Phoning', ('7', '1'): 'Posing 1', ('7', '2'): 'Posing 2', ('8', '1'): 'Purchases 1', ('8', '2'): 'Purchases', ('9', '1'): 'Sitting 1', ('9', '2'): 'Sitting', ('10', '1'): 'SittingDown 1', ('10', '2'): 'SittingDown', ('11', '1'): 'Smoking 1', ('11', '2'): 'Smoking', ('12', '1'): 'Photo 1', ('12', '2'): 'Photo', ('13', '1'): 'Waiting 1', ('13', '2'): 'Waiting', ('14', '1'): 'Walking 1', ('14', '2'): 'Walking 2', ('15', '1'): 'WalkDog 1', ('15', '2'): 'WalkDog', ('16', '1'): 'WalkTogether 1', ('16', '2'): 'WalkTogether'}, 'S4': {('1', '1'): '_ALL 1', ('1', '2'): '_ALL', ('2', '1'): 'Directions 1', ('2', '2'): 'Directions', ('3', '1'): 'Discussion 1', ('3', '2'): 'Discussion', ('4', '1'): 'Eating 1', ('4', '2'): 'Eating', ('5', '1'): 'Greeting 1', ('5', '2'): 'Greeting', ('6', '1'): 'Phoning 1', ('6', '2'): 'Phoning', ('7', '1'): 'Posing 1', ('7', '2'): 'Posing', ('8', '1'): 'Purchases 1', ('8', '2'): 'Purchases', ('9', '1'): 'Sitting 1', ('9', '2'): 'Sitting', ('10', '1'): 'SittingDown 1', ('10', '2'): 'SittingDown 2', ('11', '1'): 'Smoking 1', ('11', '2'): 'Smoking', ('12', '1'): 'Photo 1', ('12', '2'): 'Photo', ('13', '1'): 'Waiting 1', ('13', '2'): 'Waiting', ('14', '1'): 'Walking 1', ('14', '2'): 'Walking', ('15', '1'): 'WalkDog 1', ('15', '2'): 'WalkDog', ('16', '1'): 'WalkTogether 2', ('16', '2'): 'WalkTogether 3'}, 'S5': {('1', '1'): '_ALL 1', ('1', '2'): '_ALL', ('2', '1'): 'Directions 1', ('2', '2'): 'Directions 2', ('3', '1'): 'Discussion 2', ('3', '2'): 'Discussion 3', ('4', '1'): 'Eating 1', ('4', '2'): 'Eating', ('5', '1'): 'Greeting 1', ('5', '2'): 'Greeting 2', ('6', '1'): 'Phoning 1', ('6', '2'): 'Phoning', ('7', '1'): 'Posing 1', ('7', '2'): 'Posing', ('8', '1'): 'Purchases 1', ('8', '2'): 'Purchases', ('9', '1'): 'Sitting 1', ('9', '2'): 'Sitting', ('10', '1'): 'SittingDown', ('10', '2'): 'SittingDown 1', ('11', '1'): 'Smoking 1', ('11', '2'): 'Smoking', ('12', '1'): 'Photo', ('12', '2'): 'Photo 2', ('13', '1'): 'Waiting 1', ('13', '2'): 'Waiting 2', ('14', '1'): 'Walking 1', ('14', '2'): 'Walking', ('15', '1'): 'WalkDog 1', ('15', '2'): 'WalkDog', ('16', '1'): 'WalkTogether 1', ('16', '2'): 'WalkTogether'}, 'S6': {('1', '1'): '_ALL 1', ('1', '2'): '_ALL', ('2', '1'): 'Directions 1', ('2', '2'): 'Directions', ('3', '1'): 'Discussion 1', ('3', '2'): 'Discussion', ('4', '1'): 'Eating 1', ('4', '2'): 'Eating 2', ('5', '1'): 'Greeting 1', ('5', '2'): 'Greeting', ('6', '1'): 'Phoning 1', ('6', '2'): 'Phoning', ('7', '1'): 'Posing 2', ('7', '2'): 'Posing', ('8', '1'): 'Purchases 1', ('8', '2'): 'Purchases', ('9', '1'): 'Sitting 1', ('9', '2'): 'Sitting 2', ('10', '1'): 'SittingDown 1', ('10', '2'): 'SittingDown', ('11', '1'): 'Smoking 1', ('11', '2'): 'Smoking', ('12', '1'): 'Photo', ('12', '2'): 'Photo 1', ('13', '1'): 'Waiting 3', ('13', '2'): 'Waiting', ('14', '1'): 'Walking 1', ('14', '2'): 'Walking', ('15', '1'): 'WalkDog 1', ('15', '2'): 'WalkDog', ('16', '1'): 'WalkTogether 1', ('16', '2'): 'WalkTogether'}, 'S7': {('1', '1'): '_ALL 1', ('1', '2'): '_ALL', ('2', '1'): 'Directions 1', ('2', '2'): 'Directions', ('3', '1'): 'Discussion 1', ('3', '2'): 'Discussion', ('4', '1'): 'Eating 1', ('4', '2'): 'Eating', ('5', '1'): 'Greeting 1', ('5', '2'): 'Greeting', ('6', '1'): 'Phoning 2', ('6', '2'): 'Phoning', ('7', '1'): 'Posing 1', ('7', '2'): 'Posing', ('8', '1'): 'Purchases 1', ('8', '2'): 'Purchases', ('9', '1'): 'Sitting 1', ('9', '2'): 'Sitting', ('10', '1'): 'SittingDown', ('10', '2'): 'SittingDown 1', ('11', '1'): 'Smoking 1', ('11', '2'): 'Smoking', ('12', '1'): 'Photo', ('12', '2'): 'Photo 1', ('13', '1'): 'Waiting 1', ('13', '2'): 'Waiting 2', ('14', '1'): 'Walking 1', ('14', '2'): 'Walking 2', ('15', '1'): 'WalkDog 1', ('15', '2'): 'WalkDog', ('16', '1'): 'WalkTogether 1', ('16', '2'): 'WalkTogether'}, 'S8': {('1', '1'): '_ALL 1', ('1', '2'): '_ALL', ('2', '1'): 'Directions 1', ('2', '2'): 'Directions', ('3', '1'): 'Discussion 1', ('3', '2'): 'Discussion', ('4', '1'): 'Eating 1', ('4', '2'): 'Eating', ('5', '1'): 'Greeting 1', ('5', '2'): 'Greeting', ('6', '1'): 'Phoning 1', ('6', '2'): 'Phoning', ('7', '1'): 'Posing 1', ('7', '2'): 'Posing', ('8', '1'): 'Purchases 1', ('8', '2'): 'Purchases', ('9', '1'): 'Sitting 1', ('9', '2'): 'Sitting', ('10', '1'): 'SittingDown', ('10', '2'): 'SittingDown 1', ('11', '1'): 'Smoking 1', ('11', '2'): 'Smoking', ('12', '1'): 'Photo 1', ('12', '2'): 'Photo', ('13', '1'): 'Waiting 1', ('13', '2'): 'Waiting', ('14', '1'): 'Walking 1', ('14', '2'): 'Walking', ('15', '1'): 'WalkDog 1', ('15', '2'): 'WalkDog', ('16', '1'): 'WalkTogether 1', ('16', '2'): 'WalkTogether 2'}, 'S9': {('1', '1'): '_ALL 1', ('1', '2'): '_ALL', ('2', '1'): 'Directions 1', ('2', '2'): 'Directions', ('3', '1'): 'Discussion 1', ('3', '2'): 'Discussion 2', ('4', '1'): 'Eating 1', ('4', '2'): 'Eating', ('5', '1'): 'Greeting 1', ('5', '2'): 'Greeting', ('6', '1'): 'Phoning 1', ('6', '2'): 'Phoning', ('7', '1'): 'Posing 1', ('7', '2'): 'Posing', ('8', '1'): 'Purchases 1', ('8', '2'): 'Purchases', ('9', '1'): 'Sitting 1', ('9', '2'): 'Sitting', ('10', '1'): 'SittingDown', ('10', '2'): 'SittingDown 1', ('11', '1'): 'Smoking 1', ('11', '2'): 'Smoking', ('12', '1'): 'Photo 1', ('12', '2'): 'Photo', ('13', '1'): 'Waiting 1', ('13', '2'): 'Waiting', ('14', '1'): 'Walking 1', ('14', '2'): 'Walking', ('15', '1'): 'WalkDog 1', ('15', '2'): 'WalkDog', ('16', '1'): 'WalkTogether 1', ('16', '2'): 'WalkTogether'}, 'S10': {('1', '1'): '_ALL 2', ('1', '2'): '_ALL 1', ('2', '1'): 'Directions 1', ('2', '2'): 'Directions', ('3', '1'): 'Discussion 1', ('3', '2'): 'Discussion 2', ('4', '1'): 'Eating 1', ('4', '2'): 'Eating', ('5', '1'): 'Greeting 1', ('5', '2'): 'Greeting', ('6', '1'): 'Phoning 1', ('6', '2'): 'Phoning', ('7', '1'): 'Posing 1', ('7', '2'): 'Posing', ('8', '1'): 'Purchases 1', ('8', '2'): 'Purchases', ('9', '1'): 'Sitting 1', ('9', '2'): 'Sitting', ('10', '1'): 'SittingDown', ('10', '2'): 'SittingDown 1', ('11', '1'): 'Smoking 2', ('11', '2'): 'Smoking', ('12', '1'): 'Photo 1', ('12', '2'): 'Photo', ('13', '1'): 'Waiting 1', ('13', '2'): 'Waiting', ('14', '1'): 'Walking 1', ('14', '2'): 'Walking', ('15', '1'): 'WalkDog 1', ('15', '2'): 'WalkDog', ('16', '1'): 'WalkTogether 1', ('16', '2'): 'WalkTogether'}, 'S11': {('1', '1'): '_ALL 1', ('1', '2'): '_ALL', ('2', '1'): 'Directions 1', ('2', '2'): 'Directions', ('3', '1'): 'Discussion 1', ('3', '2'): 'Discussion 2', ('4', '1'): 'Eating 1', ('4', '2'): 'Eating', ('5', '1'): 'Greeting 2', ('5', '2'): 'Greeting', ('6', '1'): 'Phoning 3', ('6', '2'): 'Phoning 2', ('7', '1'): 'Posing 1', ('7', '2'): 'Posing', ('8', '1'): 'Purchases 1', ('8', '2'): 'Purchases', ('9', '1'): 'Sitting 1', ('9', '2'): 'Sitting', ('10', '1'): 'SittingDown', ('10', '2'): 'SittingDown 1', ('11', '1'): 'Smoking 2', ('11', '2'): 'Smoking', ('12', '1'): 'Photo 1', ('12', '2'): 'Photo', ('13', '1'): 'Waiting 1', ('13', '2'): 'Waiting', ('14', '1'): 'Walking 1', ('14', '2'): 'Walking', ('15', '1'): 'WalkDog 1', ('15', '2'): 'WalkDog', ('16', '1'): 'WalkTogether 1', ('16', '2'): 'WalkTogether'}}


# For local sagemaker
#images_base_path = "/root/IISc/SOTA/learnable_triangulation/learnable-triangulation-pytorch/data/human36m/processed"

# For instance run
images_base_path = '/dataset/human36m/processed'

def download_data(subjects):
    if not os.path.exists("/dataset/"):
        os.makedirs("/dataset/")
    if not os.path.exists("/dataset/human36m/"):
        os.makedirs("/dataset/human36m/")
    if not os.path.exists("/dataset/human36m/processed/"):
        os.makedirs("/dataset/human36m/processed/")

    for subject in subjects:
        if not os.path.exists(f"/dataset/human36m/processed/{subject}/"):
            os.makedirs(f"/dataset/human36m/processed/{subject}/")
        
        # Expt run actions
        actions = ['Directions-1']
        for action in actions:
            if not os.path.exists(f"/dataset/human36m/processed/{subject}/action/"):
                os.makedirs(f"/dataset/human36m/processed/{subject}/{action}/")
            
            s3_path = f"s3://pi-expt-use1-dev/ml_forecasting/s.goyal/IISc/data/human36m/processed/{subject}/{action}"
            local_path = f"/dataset/human36m/processed/{subject}/{action}/"
            subprocess.check_call(["aws", "s3", "cp", s3_path, local_path, "--recursive"])


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


def read_3d_data_me(dataset):
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            anim = dataset[subject][action]

            positions_3d = []
            camerad_para = []
            for cam in anim['cameras']:
                pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                # pos_3d[:, :] -= pos_3d[:, :1]  # Remove global offset
                positions_3d.append(pos_3d)
                camerad_para.append(camera_dict[cam['id']])
    
            anim['positions_3d'] = positions_3d
            anim['camerad_para'] = camerad_para

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

    ### GJ: adjust the length of 2d data ###
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            for cam_idx in range(len(keypoints[subject][action])):
                mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
                assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length
                if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                    keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]


    for subject in keypoints.keys():
        for action in keypoints[subject]:
            for cam_idx, kps in enumerate(keypoints[subject][action]):
                # Normalize camera frame
                cam = dataset.cameras()[subject][cam_idx]
                kps[..., 1:3] = normalize_screen_coordinates(kps[..., 1:3], w=cam['res_w'], h=cam['res_h'])
                keypoints[subject][action][cam_idx] = kps

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


def fetch_me(subjects, dataset, keypoints, action_filter=None, stride=1, parse_3d_poses=True):
    out_poses_3d = []
    out_poses_2d = []
    out_actions = []
    out_camera_para = []
    out_image_paths = [] # flattened on cameras
    
    for subject in subjects:
        subject_mapping = {v: k for k, v in mapping[subject].items()}
        for action in list(keypoints[subject].keys())[:1]:
            print(action)
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
                break

            if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                camera_para = dataset[subject][action]['camerad_para']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_3d)):  # Iterate across cameras
                    out_poses_3d.append(poses_3d[i])
                    out_camera_para.append([camera_para[i]]* poses_3d[i].shape[0])
                    break


            # Get image paths
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
                file_list = os.listdir(f"{images_base_path}/{subject}/{folder_action}/imageSequence/{cam}")
                file_list = [file for file in file_list if file[0] != '.']
                indexes = np.array([int(file.split('.')[0].split('_')[1]) - 1 for file in file_list])
                indexes.sort()

                img_file_names = [f"{images_base_path}/{subject}/{folder_action}/imageSequence/{cam}/" + "img_%06d.jpg" % (idx+1) for idx in indexes]
                out_image_paths.extend(img_file_names)
                break
            

    if len(out_poses_3d) == 0:
        out_poses_3d = None

    if stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            out_actions[i] = out_actions[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]
                out_camera_para[i] = out_poses_3d[i][::stride]

                
                
    return out_poses_3d, out_poses_2d, out_actions, out_camera_para, out_image_paths