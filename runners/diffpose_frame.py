import os
import logging
import subprocess
import time
import glob
import argparse

import os.path as path
import numpy as np
import tqdm
import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


from models.gcnpose import GCNpose, adj_mx_from_edges
from models.gcndiff import GCNdiff, adj_mx_from_edges
from models.ema import EMAHelper

from common.utils import *
from common.utils_diff import get_beta_schedule, generalized_steps
from common.data_utils import fetch_me, read_3d_data_me, create_2d_data, download_data
from common.generators import PoseGenerator_gmm
from common.loss import mpjpe, p_mpjpe


from transformers import AutoImageProcessor, ViTModel

class Diffpose(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        # GraFormer mask
        self.src_mask = torch.tensor([[[True, True, True, True, True, True, True, True, True, True,
                                True, True, True, True, True, True, True]]]).to(self.device)
        
        # Generate Diffusion sequence parameters
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]
        
        # Setup ViT model
        self.image_processor = AutoImageProcessor.from_pretrained("vit-base-patch16-224-in21k")
        self.vit_model = ViTModel.from_pretrained("vit-base-patch16-224-in21k")#, output_hidden_states=True)
        self.vit_model = self.vit_model.to(self.device)

    # prepare 2D and 3D skeleton for model training and testing 
    def prepare_data(self):
        args, config = self.args, self.config
        print('==> Using settings {}'.format(args))
        print('==> Using configures {}'.format(config))
        
        # load dataset
        if config.data.dataset == "human36m":
            from common.h36m_dataset import Human36mDataset, TRAIN_SUBJECTS, TEST_SUBJECTS
            dataset = Human36mDataset(config.data.dataset_path)
            self.subjects_train = TRAIN_SUBJECTS
            self.subjects_test = TEST_SUBJECTS
            self.dataset = read_3d_data_me(dataset)
            self.keypoints_train = create_2d_data(config.data.dataset_path_train_2d, dataset)
            self.keypoints_test = create_2d_data(config.data.dataset_path_test_2d, dataset)

            self.action_filter = None if args.actions == '*' else args.actions.split(',')
            if self.action_filter is not None:
                self.action_filter = map(lambda x: dataset.define_actions(x)[0], self.action_filter)
                print('==> Selected actions: {}'.format(self.action_filter))

            # Download Image data
            download_data(TRAIN_SUBJECTS, TEST_SUBJECTS, True)

        else:
            raise KeyError('Invalid dataset')

    # create diffusion model
    def create_diffusion_model(self, model_path = None):
        args, config = self.args, self.config
        edges = torch.tensor([[0, 1], [1, 2], [2, 3],
                            [0, 4], [4, 5], [5, 6],
                            [0, 7], [7, 8], [8, 9], [9,10],
                            [8, 11], [11, 12], [12, 13],
                            [8, 14], [14, 15], [15, 16]], dtype=torch.long)
        adj = adj_mx_from_edges(num_pts=17, edges=edges, sparse=False)
        self.model_diff = GCNdiff(adj.to(self.device), config).to(self.device)
        #self.model_diff = torch.nn.DataParallel(self.model_diff)
        
        # load pretrained model
        if model_path:
            states = torch.load(model_path)
            self.model_diff.load_state_dict(states[0])

        self.model_diff = DDP(self.model_diff)
            
    def create_pose_model(self, model_path = None):
        args, config = self.args, self.config
        
        # [input dimension u v, output dimension x y z]
        config.model.coords_dim = [2,3]
        edges = torch.tensor([[0, 1], [1, 2], [2, 3],
                            [0, 4], [4, 5], [5, 6],
                            [0, 7], [7, 8], [8, 9], [9,10],
                            [8, 11], [11, 12], [12, 13],
                            [8, 14], [14, 15], [15, 16]], dtype=torch.long)
        adj = adj_mx_from_edges(num_pts=17, edges=edges, sparse=False)
        self.model_pose = GCNpose(adj.to(self.device), config).to(self.device)
        self.model_pose = torch.nn.DataParallel(self.model_pose)
        
        # load pretrained model
        if model_path:
            logging.info('initialize model by:' + model_path)
            states = torch.load(model_path, map_location=torch.device(self.device))
            self.model_pose.load_state_dict(states[0])
        else:
            logging.info('initialize model randomly')

    def train(self, n_gpu, cur_rank):
        cudnn.benchmark = True

        args, config, src_mask = self.args, self.config, self.src_mask

        # initialize the recorded best performance
        best_p1, best_epoch = 1000, 0
        # skip rate when sample skeletons from video
        stride = self.args.downsample
        
        # create dataloader
        if config.data.dataset == "human36m":
            poses_train_2d_gt, poses_train_2d, actions_train, camerapara_train, out_image_paths_train\
                = fetch_me(self.subjects_train, self.dataset, self.keypoints_train, self.action_filter, stride)
            
            dataset_class = PoseGenerator_gmm(poses_train_2d_gt, poses_train_2d, actions_train, camerapara_train, out_image_paths_train, self.image_processor)
            train_sampler = DistributedSampler(
                dataset_class, n_gpu, cur_rank
            )
            data_loader = train_loader = data.DataLoader(
                dataset_class,
                batch_size=config.training.batch_size, shuffle=False,\
                    num_workers=config.training.num_workers, pin_memory=True,
                    sampler=train_sampler)
        else:
            raise KeyError('Invalid dataset')
        
        optimizer = get_optimizer(self.config, self.model_diff.parameters())
        
        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(self.model_diff)
        else:
            ema_helper = None
        
        start_epoch, step = 0, 0
        
        lr_init, decay, gamma = self.config.optim.lr, self.config.optim.decay, self.config.optim.lr_gamma
      
        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0

            # Switch to train mode
            torch.set_grad_enabled(True)
            self.model_diff.train()
            
            epoch_loss_diff = AverageMeter()

            data_loader.sampler.set_epoch(epoch)

            for i, (targets_uvxy, targets_noise_scale, _, targets_2d, _, _, image_feats) in enumerate(data_loader):
                data_time += time.time() - data_start
                step += 1

                # Get image embeeding from ViT
                input_feats = image_feats['pixel_values'].reshape((-1, 3, 224, 224))
                input_feats = input_feats.to(self.device)
                with torch.no_grad():
                    outputs = self.vit_model(pixel_values = input_feats)

                image_features = outputs.last_hidden_state
                #print("image_features shape", image_features.shape)

                # to cuda
                targets_uvxy, targets_noise_scale, targets_2d = \
                    targets_uvxy.to(self.device), targets_noise_scale.to(self.device), targets_2d.to(self.device)
                
                # generate nosiy sample based on seleted time t and beta
                n = targets_2d.size(0)
                x = targets_uvxy
                e = torch.randn_like(x)
                b = self.betas            
                t = torch.randint(low=0, high=self.num_timesteps,
                                  size=(n // 2 + 1,)).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                e = e*(targets_noise_scale)
                a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1)
                # generate x_t (refer to DDIM equation)
                x = x * a.sqrt() + e * (1.0 - a).sqrt()
                
                # predict noise
                output_noise = self.model_diff(x, src_mask, t.float(), image_features)
                loss_diff = (e - output_noise).square().sum(dim=(1, 2)).mean(dim=0) # Check this sum on dims
                
                optimizer.zero_grad()
                loss_diff.backward()
                
                torch.nn.utils.clip_grad_norm_(
                    self.model_diff.parameters(), config.optim.grad_clip)                
                optimizer.step()
            
                epoch_loss_diff.update(loss_diff.item(), n)
            
                if self.config.model.ema:
                    ema_helper.update(self.model_diff)
                
                if cur_rank == 0 and i%100 == 0 and i != 0:
                    logging.info('| Epoch{:0>4d}: {:0>4d}/{:0>4d} | Step {:0>6d} | Data: {:.6f} | Loss: {:.6f} |'\
                        .format(epoch, i+1, len(data_loader), step, data_time, epoch_loss_diff.avg))
            
            data_start = time.time()

            if epoch % decay == 0:
                lr_now = lr_decay(optimizer, epoch, lr_init, decay, gamma) 
                
            if cur_rank == 0 and epoch % 1 == 0:
                states = [
                    self.model_diff.state_dict(),
                    optimizer.state_dict(),
                    epoch,
                    step,
                ]
                if self.config.model.ema:
                    states.append(ema_helper.state_dict())

                torch.save(states,os.path.join(self.args.log_path, "ckpt_{}.pth".format(epoch)))
                torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))
            
                logging.info('test the performance of current model')

                p1, p2 = self.test_hyber(is_train=True)

                if p1 < best_p1:
                    best_p1 = p1
                    best_epoch = epoch
                logging.info('| Best Epoch: {:0>4d} MPJPE: {:.2f} | Epoch: {:0>4d} MPJEPE: {:.2f} PA-MPJPE: {:.2f} |'\
                    .format(best_epoch, best_p1, epoch, p1, p2))
                
                
                #Saving Model to S3
                # logging.info('Files in log path')
                # subprocess.check_call(
                #     ["aws", "s3", "ls", self.args.log_path]
                # )
                logging.info('Saving Checkpoint')
                s3_model_dir = "s3://pi-expt-use1-dev/ml_forecasting/s.goyal/IISc/diffPose-2D_cond_project_bound_scale/"
                subprocess.check_call(
                    ["aws", "s3", "cp", self.args.log_path, s3_model_dir, "--recursive"]
                )
    
    def test_hyber(self, is_train=False):
        cudnn.benchmark = True

        args, config, src_mask = self.args, self.config, self.src_mask
        test_times, test_timesteps, test_num_diffusion_timesteps, stride = \
            config.testing.test_times, config.testing.test_timesteps, config.testing.test_num_diffusion_timesteps, args.downsample
                
        if config.data.dataset == "human36m":
            poses_valid, poses_valid_2d, actions_valid, camerapara_valid, out_image_paths_valid= \
                fetch_me(self.subjects_test, self.dataset, self.keypoints_test, self.action_filter, stride)
            data_loader = valid_loader = data.DataLoader(
                PoseGenerator_gmm(poses_valid, poses_valid_2d, actions_valid, camerapara_valid, out_image_paths_valid, self.image_processor),
                batch_size=config.training.batch_size, shuffle=False, 
                num_workers=config.training.num_workers, pin_memory=True)
        else:
            raise KeyError('Invalid dataset') 

        data_start = time.time()
        data_time = 0

        # Switch to test mode
        torch.set_grad_enabled(False)
        self.model_diff.eval()
        self.model_pose.eval()
        
        try:
            skip = self.args.skip
        except Exception:
            skip = 1
        
        if self.args.skip_type == "uniform":
            skip = test_num_diffusion_timesteps // test_timesteps
            seq = range(0, test_num_diffusion_timesteps, skip)
        elif self.args.skip_type == "quad":
            seq = (np.linspace(0, np.sqrt(test_num_diffusion_timesteps * 0.8), test_timesteps)** 2)
            seq = [int(s) for s in list(seq)]
        else:
            raise NotImplementedError
        
        epoch_loss_3d_pos = AverageMeter()
        epoch_loss_3d_pos_procrustes = AverageMeter()
        self.test_action_list = ['Directions','Discussion','Eating','Greeting','Phoning','Photo','Posing','Purchases','Sitting',\
            'SittingDown','Smoking','Waiting','WalkDog','Walking','WalkTogether']
        action_error_sum = define_error_list(self.test_action_list)        

        for i, (_, input_noise_scale, input_2d, targets_2d, input_action, camera_para, image_feats) in enumerate(data_loader):
            data_time += time.time() - data_start

            # Get image embeeding from ViT
            input_feats = image_feats['pixel_values'].reshape((-1, 3, 224, 224))
            input_feats = input_feats.to(self.device)
            with torch.no_grad():
                outputs = self.vit_model(pixel_values = input_feats)

            image_features = outputs.last_hidden_state
            #print("valid image_features shape", image_features.shape)

            input_noise_scale, input_2d, targets_2d = \
                input_noise_scale.to(self.device), input_2d.to(self.device), targets_2d.to(self.device)

            # build uvxyz
            #inputs_xyz = self.model_pose(input_2d, src_mask)
            #inputs_xy = inputs_xyz[:, : , :2]
            #print("inputs_xy", inputs_xy.shape)
            #inputs_xy[:, :, :2] -= inputs_xy[:, :1, :2] 
            
            
            # Build noisy pose
            x = input_2d
            e = torch.randn_like(x)
            b = self.betas          
            t = torch.tensor([20]).to(self.device)
            #e = e*(target_noise_scale) Using 1 scale
            a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1)
            # generate x_t (refer to DDIM equation)
            inputs_xy = x * a.sqrt() + e * (1.0 - a).sqrt()
            inputs_xy[:, :, :] -= inputs_xy[:, :1, :] 

            
            input_uvxy = torch.cat([input_2d,inputs_xy],dim=2)
            
            
            #return input_2d, inputs_xy, targets_2d
                        
            # generate distribution
            input_uvxy = input_uvxy.repeat(test_times,1,1)
            #input_noise_scale = input_noise_scale.repeat(test_times,1,1)
            # select diffusion step
            #t = torch.ones(input_uvxy.size(0)).type(torch.LongTensor).to(self.device)*test_num_diffusion_timesteps
            
            
            #print('Logging shapes')
            #print("input_2d", input_2d.shape)
            #print("inputs_xy", inputs_xy.shape)
            #print("input_noise_scale", input_noise_scale.shape)
            
            # prepare the diffusion parameters
            x = input_uvxy.clone()
            # e = torch.randn_like(input_uvxy)
            # b = self.betas   
            # e = e*input_noise_scale        
            # a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1)
            # x = x * a.sqrt() + e * (1.0 - a).sqrt()
            
            output_uvxy = generalized_steps(x, src_mask, seq, self.model_diff, self.betas, image_features, self.device, eta=self.args.eta)
            output_uvxy = output_uvxy[0][-1]            
            output_uvxy = torch.mean(output_uvxy.reshape(test_times,-1,17,4),0)
            output_xy = output_uvxy[:,:,2:]
            output_xy[:, :, :] -= output_xy[:, :1, :]
            
            new_targets_2d = targets_2d - targets_2d[:, :1, :]
            epoch_loss_3d_pos.update(mpjpe(output_xy, new_targets_2d).item() * 1000.0, new_targets_2d.size(0))
            epoch_loss_3d_pos_procrustes.update(p_mpjpe(output_xy.cpu().numpy(), new_targets_2d.cpu().numpy()).item() * 1000.0, new_targets_2d.size(0))\
            
            data_start = time.time()
            
            action_error_sum = test_calculation(output_xy, new_targets_2d, input_action, action_error_sum, None, None)
            
            if i%100 == 0 and i != 0:
                logging.info('({batch}/{size}) Data: {data:.6f}s | MPJPE: {e1: .4f} | P-MPJPE: {e2: .4f}'\
                        .format(batch=i + 1, size=len(data_loader), data=data_time, e1=epoch_loss_3d_pos.avg,\
                            e2=epoch_loss_3d_pos_procrustes.avg))
        logging.info('sum ({batch}/{size}) Data: {data:.6f}s | MPJPE: {e1: .4f} | P-MPJPE: {e2: .4f}'\
                .format(batch=i + 1, size=len(data_loader), data=data_time, e1=epoch_loss_3d_pos.avg,\
                    e2=epoch_loss_3d_pos_procrustes.avg))
        
        p1, p2 = print_error(None, action_error_sum, is_train)

        return p1, p2