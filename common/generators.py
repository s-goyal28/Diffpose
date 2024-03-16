from __future__ import print_function, absolute_import
import os
import subprocess

import numpy as np
import torch
from torch.utils.data import Dataset
from functools import reduce

from PIL import Image


class PoseGenerator_gmm(Dataset):
    def __init__(self, poses_3d, poses_2d_gmm, actions, camerapara, image_paths, image_processor):
        assert poses_3d is not None

        self._poses_3d = np.concatenate(poses_3d)
        self._poses_2d_gmm = np.concatenate(poses_2d_gmm)
        self._actions = reduce(lambda x, y: x + y, actions)
        self._camerapara = np.concatenate(camerapara)
        self._kernel_n = self._poses_2d_gmm.shape[2]

        self.image_paths = image_paths # ALready flattened on subjects, actions , cameras
        self.image_processor = image_processor

        self._poses_3d[:,:,:] = self._poses_3d[:,:,:]-self._poses_3d[:,:1,:]

        assert self._poses_3d.shape[0] == self._poses_2d_gmm.shape[0] and self._poses_3d.shape[0] == len(self._actions)
        print('Generating {} poses...'.format(len(self._actions)))

    def __getitem__(self, index):
        out_pose_2d = self._poses_3d[index][:,:2]
        out_pose_2d_gmm = self._poses_2d_gmm[index]
        out_action = self._actions[index]
        out_camerapara = self._camerapara[index]

        # randomly select a kernel from gmm
        out_pose_2d_kernel = np.zeros([out_pose_2d_gmm.shape[0],out_pose_2d_gmm.shape[2]])
        for i in range(out_pose_2d_gmm.shape[0]):
            out_pose_2d_kernel[i] = out_pose_2d_gmm[i,np.random.choice(self._kernel_n, 1, p=out_pose_2d_gmm[i,:,0]).item()]
        
        # generate uvxyz and uvxyz noise scale
        kernel_mean = out_pose_2d_kernel[:,1:3]
        kernel_variance = out_pose_2d_kernel[:,3:]

        out_pose_uvxy = np.concatenate((kernel_mean,out_pose_2d),axis=1)
        out_pose_noise_scale = np.concatenate((kernel_variance,np.ones(out_pose_2d.shape)),axis=1)

        out_pose_uvxy = torch.from_numpy(out_pose_uvxy).float()
        out_pose_noise_scale = torch.from_numpy(out_pose_noise_scale).float()
        out_pose_2d_mean = torch.from_numpy(kernel_mean).float()
        out_pose_2d = torch.from_numpy(out_pose_2d).float()
        out_camerapara = torch.from_numpy(out_camerapara).float()

        # download and load image
        s3_path = f"s3://pi-expt-use1-dev/ml_forecasting/s.goyal/IISc/data/{self.image_paths[index]}"
        local_path = f"/dataset/{self.image_paths[index]}"

        if not os.path.exists(local_path):
            subprocess.check_call(["aws", "s3", "cp", s3_path, local_path])
        image = Image.open(local_path)
        image_feats = self.image_processor(image, return_tensors="pt")
        
        return out_pose_uvxy, out_pose_noise_scale, out_pose_2d_mean, out_pose_2d, out_action, out_camerapara, image_feats

    def __len__(self):
        return len(self._actions)