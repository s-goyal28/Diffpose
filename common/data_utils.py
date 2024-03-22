from __future__ import absolute_import, division
import os
import subprocess
import multiprocessing 

import numpy as np
from boto3.session import Session as BotoSession
from botocore.exceptions import ClientError

from .camera import world_to_camera, normalize_screen_coordinates, image_coordinates, project_to_2d
from .utils import wrap

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
##images_base_path = '/dataset/human36m/processed'

# For downloading in Dataloader
images_base_path = 'human36m/processed'


def subprocess_call(action, subjects):
    for subject in subjects:
        if not os.path.exists(f"/dataset/human36m/processed/{subject}/{action}/"):
            os.makedirs(f"/dataset/human36m/processed/{subject}/{action}/")
        s3_path = f"s3://pi-expt-use1-dev/ml_forecasting/s.goyal/IISc/data/human36m/processed/{subject}/{action}"
        local_path = f"/dataset/human36m/processed/{subject}/{action}"
        subprocess.check_call(["aws", "s3", "cp", s3_path, local_path, "--recursive"])

def download_data(train_subjects, test_subjects, all_data):
    if not os.path.exists("/dataset/"):
        os.makedirs("/dataset/")
    if not os.path.exists("/dataset/human36m/"):
        os.makedirs("/dataset/human36m/")
    if not os.path.exists("/dataset/human36m/processed/"):
        os.makedirs("/dataset/human36m/processed/")
    for subject in train_subjects + test_subjects:
            if not os.path.exists(f"/dataset/human36m/processed/{subject}/"):
                os.makedirs(f"/dataset/human36m/processed/{subject}/")


    if all_data:

        actions = ['Sitting-1', 'SittingDown-1', 'Greeting-2', 'Walking-2', 'Purchases-1', 'Phoning-2', 'Directions-1', 'Waiting-2', 'Discussion-1', 'Smoking-1', 'TakingPhoto-1', 'Eating-1', 'WalkingDog-2', 'Posing-1', 'MySegmentsMat', 'WalkingTogether-2', 'Sitting-2', 'Phoning-1', 'Waiting-1', 'SittingDown-2', 'WalkingDog-1', 'Purchases-2', 'WalkingTogether-1', 'Directions-2', 'Greeting-1', 'Discussion-2', 'Walking-1', 'Smoking-2', 'TakingPhoto-2', 'Eating-2', 'Posing-2']

        processes = []
        for action in actions:
            p = multiprocessing.Process(target=subprocess_call, args=(action, train_subjects + test_subjects))
            processes.append(p)
            p.start()

        for process in processes:
            process.join()

    else:
            
            # Expt run actions
            actions = ['Directions-1']
            for action in actions:
                if not os.path.exists(f"/dataset/human36m/processed/{subject}/{action}/"):
                    os.makedirs(f"/dataset/human36m/processed/{subject}/{action}/")
                
                s3_path = f"s3://pi-expt-use1-dev/ml_forecasting/s.goyal/IISc/data/human36m/processed/{subject}/{action}"
                local_path = f"/dataset/human36m/processed/{subject}/{action}/"
                subprocess.check_call(["aws", "s3", "cp", s3_path, local_path, "--recursive"])


boto_creds = {
    'fs.s3a.endpoint' : 's3.amazonaws.com',
    'fs.s3a.region' : 'us-east-1'
}
def create_boto3_session(s3_creds):
    """
    Creates AWS Boto session to access S3
    Args:
        :param s3_creds: <dict> access key, secret key and region name to access S3
        :return: returns a boto session object
    """
    if "fs.s3a.access.key" in s3_creds.keys():
        boto_session = BotoSession(
            aws_access_key_id=s3_creds["fs.s3a.access.key"],
            aws_secret_access_key=s3_creds["fs.s3a.secret.key"],
            region_name=s3_creds["fs.s3a.region"],
        )
    else:
        boto_session = BotoSession(region_name=s3_creds["fs.s3a.region"])

    return boto_session


class S3Boto(object):
    """
    Class for Boto3 to load, save data and list files in s3
    """

    def __init__(self, boto_session, bucket):
        self.boto = boto_session.client("s3")
        self.bucket = bucket

    def list_files(
        self,
        data_path=None,
        recursive=False,
        list_dirs=False,
        list_objs=True,
        limit=None,
        full_path=True,
        s3_prefix="s3://",
    ):
        """
        Lists all files in given S3 path
        Args:
            :param data_path: Path to folder in which we need to list all files
            :param recursive: If true, list all objects; if false, list "depth-0" directories or objects
            :param list_dirs: Has no effect when recursive=True. For non-recursive listing, if false,
                              directories will not be included
            :param list_objs: If false, objects will not be included
            :param limit: Optional. If specified, then lists at most this many items.
            :param full_path: Appends the s3:// and the bucket to every file path in the given directory
            :param s3_prefix: The prefix to prepend to path
            :return: List of file paths in given S3 folder
        """
        results = []

        kwargs = dict()
        kwargs.update(Bucket=self.bucket)
        kwargs.update(RequestPayer="requester")

        if data_path is not None:
            if not data_path.endswith("/"):
                data_path += "/"
            kwargs.update(Prefix=data_path)

        if not recursive:
            kwargs.update(Delimiter="/")

        if limit is not None:
            kwargs.update(MaxKeys=limit)
            
        kwargs.update(PaginationConfig={"PageSize": 1000})
        paginator = self.boto.get_paginator("list_objects_v2")
        response = paginator.paginate(**kwargs) #Bucket=self.bucket, PaginationConfig={"PageSize": 400})
        for _objects in response:
            
        # print(f"Boto3 parameters for listing files {kwargs}")
        
        # _objects = self.boto.Bucket(self.bucket).meta.client.list_objects_v2(
        #     **kwargs
        # )

            if list_dirs and ("CommonPrefixes" in _objects):
                for _obj in _objects.get("CommonPrefixes"):
                    _path = _obj.get("Prefix")
                    results.append(_path)

            if list_objs and ("Contents" in _objects):
                for _obj in _objects.get("Contents"):
                    _path = _obj.get("Key")
                    results.append(_path)
        

        if full_path:
            results = [
                s3_prefix + self.bucket + "/" + _path for _path in results
            ]
            
        print(f"List files Done for path : {data_path}")
        return results

    def path_exists(self, path):
        """
        check if file/folder exists
        Args:
            :param path: Path to file/folder that needs to be checked
            :return: Bool
        """

        try:
            self.boto.Object(self.bucket, path).load(RequestPayer="requester")
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                # The object does not exist.
                print(f"file {path} not found")
            return False
        return True

    def copy_file(self: "S3Boto", path: str, dest_bucket: str, dest_path: str):
        """
        copy file to dest bucket
        Args:
            :param path: Path to file that needs to be copied
            :param dest_bucket: Destination bucket
            :param dest_path: Destination path
        """

        copy_source = {"Bucket": self.bucket, "Key": path}
        extra_args = {
            "RequestPayer": "requester",
            "ACL": "bucket-owner-full-control",
        }

        try:
            self.boto.meta.client.copy(
                copy_source, dest_bucket, dest_path, extra_args
            )
        except ClientError as ex:
            raise ex
        


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


# def equilize_aspect_ratio(min_y, min_x, max_y, max_x):
#     current_aspect_ratio = (max_x - min_x) / (max_y - min_y)
#     desired_aspect_ratio = 1.0
    
#     # Adjust the bounding box to match the desired aspect ratio
#     if current_aspect_ratio > desired_aspect_ratio:
#         # Increase the height of the bounding box
#         center_y = (min_y + max_y) / 2
#         new_height = (max_x - min_x) / desired_aspect_ratio
#         min_y = center_y - new_height / 2
#         max_y = center_y + new_height / 2
#     else:
#         # Increase the width of the bounding box
#         center_x = (min_x + max_x) / 2
#         new_width = (max_y - min_y) * desired_aspect_ratio
#         min_x = center_x - new_width / 2
#         max_x = center_x + new_width / 2
        
#     return (min_y, min_x, max_y, max_x)

# def normalize_cordinates(X, min_y, min_x, max_y, max_x):
#     return ((X - [min_x, min_y]) / [(max_x - min_x), (max_y - min_y)]) *2 -1


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

                    # Ofset with bounding box
                    pose[:, 0] -= left
                    pose[:, 1] -= top
                    
                    # Scale to 224 x 224 as in ViT
                    pose[:, :] = (pose[:, :] / [right-left, bottom-top]) * [224, 224]
                    
                    #Normalize between -1 and 1
                    pose[:, :] = normalize_screen_coordinates(pose, w=224, h=224)
                
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
                    
                    kps[:, :, 1:3] = (kps[:, :, 1:3] / [right-left, bottom-top]) * [224, 224]
                    
                    kps[:, :, 1:3] = normalize_screen_coordinates(kps[:, :, 1:3], w=224, h=224)
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
    out_image_paths = [] # flattened on cameras
    
    for subject in subjects:
        subject_mapping = {v: k for k, v in mapping[subject].items()}
        for action in list(keypoints[subject].keys())[:1]:
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

            if parse_2d_poses_gt and 'positions_2d' in dataset[subject][action]:
                poses_2d_gt = dataset[subject][action]['positions_2d']
                camera_para = dataset[subject][action]['camerad_para']
                assert len(poses_2d_gt) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_2d_gt)):  # Iterate across cameras
                    out_poses_2d_gt.append(poses_2d_gt[i])
                    out_camera_para.append([camera_para[i]]* poses_2d_gt[i].shape[0])
                    break


            # Get image paths
            boto_session = create_boto3_session(boto_creds)
            boto_io = S3Boto(boto_session, 'pi-expt-use1-dev')

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
                #file_list = os.listdir(f"{images_base_path}/{subject}/{folder_action}/imageSequence/{cam}")
                file_list = boto_io.list_files(f"ml_forecasting/s.goyal/IISc/data/{images_base_path}/{subject}/{folder_action}/imageSequence/{cam}")
                file_list = [file for file in file_list if file.split('/')[-1][0] != '.']
                if len(file_list) != poses_2d[i].shape[0]:
                    print(f"ml_forecasting/s.goyal/IISc/data/{images_base_path}/{subject}/{folder_action}/imageSequence/{cam}")
                    print(len(file_list), poses_2d[i].shape[0])
                indexes = np.array([int(file.split('/')[-1].split('.')[0].split('_')[1]) - 1 for file in file_list])
                indexes.sort()

                img_file_names = [f"{images_base_path}/{subject}/{folder_action}/imageSequence/{cam}/" + "img_%06d.jpg" % (idx+1) for idx in indexes]
                out_image_paths.extend(img_file_names)
                break
            

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

                
                
    return out_poses_2d_gt, out_poses_2d, out_actions, out_camera_para, out_image_paths