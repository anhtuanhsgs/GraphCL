import os, sys, glob, time, copy
from os import sys, path
import gym
import numpy as np
from collections import deque
from gym.spaces.box import Box

from skimage.measure import label
from skimage.morphology import disk
from skimage.morphology import binary_dilation

from sklearn.metrics import adjusted_rand_score
from cv2 import resize
from Utils.utils import *
from Utils.img_aug_func import *
import albumentations as A
import cv2
import random
from gym.spaces import Box, Discrete, Tuple
import matplotlib.pyplot as plt
from malis import rand_index 
from random import shuffle
from PIL import Image, ImageFilter
from utils import guassian_weight_map, density_map, malis_rand_index, malis_f1_score, adjusted_rand_index
from skimage.draw import line_aa
from misc.Voronoi import *
import time
from rewards import *

# python main.py --env EM_env_DEBUG_1 --gpu-id 0 1 2 3 4 5 6 7 --workers 12 --lbl-agents 2 \--num-steps 5 --max-episode-length 5 --reward normal --model DilatedUNet --merge_radius 16 --merge_speed 2 --split_radius 64 --split_speed 4  --use-lbl --size 128 128 --hidden-feat 2  --log-period 10 --features 32 64 128 256 --downsample 2 --data zebrafish
debug = True

class General_env (gym.Env):
    def init (self, config):
        self.T = config ['T']
        self.merge_radius = config ["merge_radius"]
        self.split_radius = config ["split_radius"]
        self.merge_speed = config ["merge_speed"]
        self.split_speed = config ["split_speed"]
        self.size = config ["size"]
        if config ["use_lbl"]:
            self.observation_space = Box (0, 1.0, shape=[config["observation_shape"][0]] + self.size, dtype=np.float32)
        else:
            self.observation_space = Box (-1.0, 1.0, shape=[config["observation_shape"][0]] + self.size, dtype=np.float32)
        self.rng = np.random.RandomState(time_seed ())
        self.max_lbl = 2 ** (self .T) - 1
        self.pred_lbl2rgb = color_generator (self.max_lbl + 1)
        self.gt_lbl2rgb = color_generator (111)

    def seed (self, seed):
        self.rng = np.random.RandomState(seed)

    def aug (self, image, mask):
        
        if self.config ["data"] == "zebrafish":
            randomBrightness =  A.RandomBrightness (p=0.3, limit=0.1)
            RandomContrast = A.RandomContrast (p=0.1)
        else:
            randomBrightness = A.RandomBrightness (p=0.7, limit=0.5)
            RandomContrast = A.RandomContrast (p=0.5)

        if image.shape [-1] == 3:
            if self.config ["data"] in ["Cityscape", "kitti"]:
                aug = A.Compose([
                        A.HorizontalFlip (p=0.5),
                        A.OneOf([
                            A.ElasticTransform(p=0.9, alpha=1, sigma=5, alpha_affine=5, interpolation=cv2.INTER_NEAREST),
                            A.GridDistortion(p=0.9, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT),
                            A.OpticalDistortion(p=0.9, distort_limit=(0.2, 0.2), shift_limit=(0, 0), interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT),                 
                            ], p=0.7),
                        A.ShiftScaleRotate (p=0.7, shift_limit=0.2, rotate_limit=10, interpolation=cv2.INTER_NEAREST, scale_limit=(-0.4, 0.4), border_mode=cv2.BORDER_CONSTANT),
                        A.RandomBrightness (p=0.7, limit=0.5),
                        A.RandomContrast (p=0.5),
                        A.GaussNoise (p=0.5),
                        A.Blur (p=0.5, blur_limit=4),
                        ]
                    )
            else:
                aug = A.Compose([
                            A.HorizontalFlip (p=0.5),
                            A.VerticalFlip(p=0.5),              
                            A.RandomRotate90(p=0.5),
                            A.Transpose (p=0.5),
                            A.OneOf([
                                A.ElasticTransform(p=0.9, alpha=1, sigma=5, alpha_affine=5, interpolation=cv2.INTER_NEAREST),
                                A.GridDistortion(p=0.9, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT),
                                A.OpticalDistortion(p=0.9, distort_limit=(0.2, 0.2), shift_limit=(0, 0), interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT),                 
                                ], p=0.7),
                            A.ShiftScaleRotate (p=0.7, shift_limit=0.3, rotate_limit=180, interpolation=cv2.INTER_NEAREST, scale_limit=(-0.3, 0.5), border_mode=cv2.BORDER_CONSTANT),
                            A.CLAHE(p=0.3),
                            A.RandomBrightness (p=0.7, limit=0.5),
                            A.RandomContrast (p=0.5),
                            A.GaussNoise (p=0.5),
                            A.Blur (p=0.5, blur_limit=4),
                            ]
                        )
        else:
            aug = A.Compose([
                        A.HorizontalFlip (p=0.5),
                        A.VerticalFlip(p=0.5),              
                        A.RandomRotate90(p=0.5),
                        A.Transpose (p=0.5),
                        A.OneOf([
                            A.ElasticTransform(p=0.5, alpha=1, sigma=5, alpha_affine=5, interpolation=cv2.INTER_NEAREST),
                            A.GridDistortion(p=0.5, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT),
                            A.OpticalDistortion(p=0.5, distort_limit=(0.2, 0.2), shift_limit=(0, 0), interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT),                 
                            ], p=0.6),
                        A.ShiftScaleRotate (p=0.5, shift_limit=0.3, rotate_limit=180, interpolation=cv2.INTER_NEAREST, scale_limit=(-0.2, 0.2), border_mode=cv2.BORDER_CONSTANT),
                        # A.CLAHE(p=0.3),
                        randomBrightness,
                        RandomContrast,
                        A.GaussNoise (p=0.5),
                        A.Blur (p=0.3, blur_limit=4),
                        ]
                    )
        aug = A.Compose ([])
        ret = aug (image=image, mask=mask)
        return ret ['image'], ret ['mask']

    def step_inference (self, action):
        self.action = action
        self.new_lbl = self.lbl + action * (2 ** self.step_cnt)
        self.lbl = self.new_lbl
        done = False
        info = {}
        reward = np.zeros (self.size, dtype=np.float32)
        
        self.mask [self.step_cnt:self.step_cnt+1] += (2 * action - 1) * 255
        self.step_cnt += 1
        if self.step_cnt >= self.T:
            done = True
        ret = (self.observation (), reward, done, info)
        return ret

    def step (self, action):
        self.action = action
        self.new_lbl = self.lbl + action * (2 ** self.step_cnt)
        done = False

        self.mask [self.step_cnt:self.step_cnt+1] += (2 * action - 1) * 255
        info = {}

        if (self.step_cnt == 0):
            reward = self.first_step_reward ()
            self.lbl = self.new_lbl
            self.step_cnt += 1
            self.rewards.append (reward)    
            self.sum_reward += reward
            ret = (self.observation (), reward, done, info)
            return ret

        reward = np.zeros (self.size, dtype=np.float32)

        # reward += self.foreground_reward (self.step_cnt>=self.T)
        reward += self.background_reward (False)
        
        split_reward = np.zeros (self.size, dtype=np.float32)
        merge_reward = np.zeros (self.size, dtype=np.float32)
        split_reward_inr = np.zeros (self.size, dtype=np.float32)

        if self.config ["reward"] == "seg":
            if self.config ["seg_scale"]:
                scaler = self.scaler
            else:
                scaler = None
            # print (len (self.bdrs [1]), len (self.bdrs [0]), len (np.unique (self.gt_lbl)), len (self.segs), len (self.inrs))
            # while (True):
            #     pass
            for i in range (len (self.bdrs)):
                split_reward += split_reward_s (self.lbl, self.new_lbl, self.gt_lbl, self.step_cnt==0, self.inrs [0], self.inrs [0], self.bdrs [i], self.T, scaler)
            for i in range (len (self.inrs)):
                merge_reward += merge_reward_s (self.lbl, self.new_lbl, self.gt_lbl, self.step_cnt==0, self.segs, self.inrs [i], self.bdrs [0], self.T, scaler)
            # merge_reward += merge_pen_action (action, self.gt_lbl, self.step_cnt==0, self.segs, self.inrs [0], self.bdrs [0], self.T, scaler)
            # split_reward += split_rew_action (action, self.gt_lbl, self.step_cnt==0, self.segs, self.inrs [0], self.bdrs [0], self.T, scaler)
            
            # split_reward_inr += split_reward_s_onlyInr (self.lbl, self.new_lbl, self.gt_lbl, self.step_cnt==0, self.inrs, self.inrs, self.bdrs, self.T, scaler)
            reward += self.config ["spl_w"] * split_reward + self.config ["mer_w"] * merge_reward #+ split_reward * merge_reward`

        self.lbl = self.new_lbl
        self.step_cnt += 1
        
        #Reward
        self.rewards.append (reward)    
        self.sum_reward += reward
        if self.step_cnt >= self.T:
            done = True
        ret = (self.observation (), reward, done, info)
        return ret

    def unique (self):
        return np.unique (self.lbl, return_counts=True)

    def reset_end (self):
        """
            Must call after reset
        """
        self.w_map = None
        self.density = None
        if self.config ["reward"] == "density":
            self.density = density_map (self.gt_lbl)
        elif self.config ["reward"] == "seg" and self.type == "train":

            self.gt_lbl = relabel (reorder_label (self.gt_lbl))
            self.segs = [self.gt_lbl == idx for idx in np.unique (self.gt_lbl)]
            self.bdrs = []
            self.inrs = []

            for radius in self.config ["out_radius"]:
                self.bdrs += [[seg ^ budget_binary_dilation (seg, radius) for seg in self.segs]]
            for radius in self.config ["in_radius"]:
                self.inrs += [[budget_binary_erosion (seg, radius, minsize=self.config["minsize"]) for seg in self.segs]]
            # self.inrs = [seg for seg in self.segs]

            if self.config ["seg_scale"]:
                self.scaler = np.zeros (self.gt_lbl.shape, dtype=np.float32)
                for value in np.unique (self.gt_lbl):
                    if (value == 0):
                        continue;
                    seg = self.gt_lbl == value
                    area = np.sqrt (np.count_nonzero (seg))
                    max_area = np.sqrt (self.gt_lbl.shape [0] * self.gt_lbl.shape [1])
                    _area =  max_area - area
                    self.scaler += (seg *  - np.log (area / max_area))
                self.scaler += 1 - (self.gt_lbl > 0)   
        else:
            self.density = np.ones (self.size, dtype=np.float32)
            

    def first_step_reward (self, density=None):
        reward = np.zeros (self.size, dtype=np.float32)
        st_foregr_ratio = self.config ["st_fgbg_ratio"]
        reward += ((self.new_lbl != 0) & (self.gt_lbl != 0)) * (1.0 - st_foregr_ratio)
        reward += ((self.new_lbl == 0) & (self.gt_lbl == 0)) * (st_foregr_ratio)
        reward -= ((self.new_lbl == 0) & (self.gt_lbl != 0)) * (1.0 - st_foregr_ratio)
        reward -= ((self.new_lbl != 0) & (self.gt_lbl == 0)) * (st_foregr_ratio)
        if self.config ["seg_scale"]:
            reward *= self.scaler
        return reward

    def fgbg_reward (self, scaler=None):
        reward = np.zeros (self.size, dtype=np.float32)
        foregr_ratio = self.config ["fgbg_ratio"]
        # backgr reward, penalty
        reward += ((self.new_lbl == 0) & (self.gt_lbl == 0)) * foregr_ratio
        reward -= ((self.new_lbl != 0) & (self.gt_lbl == 0)) * foregr_ratio
        # foregr reward, penalty
        reward += ((self.new_lbl != 0) & (self.gt_lbl != 0)) * (1 - foregr_ratio)
        reward -= ((self.new_lbl == 0) & (self.gt_lbl != 0)) * (1 - foregr_ratio)
        if self.config ["seg_scale"]:
            reward *= self.scaler
        return reward

    def background_reward (self, last_step):
        reward = np.zeros (self.size, dtype=np.float32)
        foregr_ratio = self.config ["fgbg_ratio"]
        if last_step:
            reward += ((self.new_lbl == 0) & (self.gt_lbl == 0)) * foregr_ratio
        reward -= ((self.new_lbl != 0) & (self.lbl == 0) & (self.gt_lbl == 0)) * foregr_ratio
        if self.config ["seg_scale"]:
            reward *= self.scaler
        return reward   
    
    def foreground_reward (self, last_step):
        reward = np.zeros (self.size, dtype=np.float32)
        foregr_ratio = self.config ["fgbg_ratio"]
        reward += ((self.new_lbl != 0) & (self.lbl == 0) & (self.gt_lbl != 0)) * (1 - foregr_ratio)
        if last_step:
            reward -= ((self.new_lbl == 0) & (self.gt_lbl != 0)) * (1 - foregr_ratio)
        if self.config ["seg_scale"]:
            reward *= self.scaler
        return reward

    def observation (self):
        lbl = self.lbl / self.max_lbl * 255.0
        done_mask = np.zeros (self.size, dtype=np.float32)
        if self.step_cnt >= self.T:
            done_mask += 255.0
        if self.config ["data_chan"] == 1:
            obs = [self.raw [None].astype (np.float32), done_mask [None]]
        elif self.config ["data_chan"] == 3:
            obs = [self.raw.astype (np.float32).transpose ([2, 0, 1]), done_mask [None]]
        if self.config ["use_lbl"]:
            obs.append (lbl [None])
        if self.config ["use_masks"]:
            obs.append (self.mask)

        obs = np.concatenate (obs, 0)
        
        # tmps = []
        # for tmp in range (len (obs)):
        #     tmps += [obs[tmp]]
        # tmp = np.concatenate (tmps, 1)
        # self.debug_img = tmp
        # plt.imshow (self.debug_img)
        # plt.show ()

        return obs / 255.0

    def render (self):
        if self.config ["data_chan"] == 1:
            raw = np.repeat (np.expand_dims (self.raw, -1), 3, -1).astype (np.uint8)
        elif self.config ["data_chan"] == 3:
            raw = self.raw
        lbl = self.lbl.astype (np.int32)

        lbl = self.pred_lbl2rgb (lbl)
        gt_lbl = self.gt_lbl % 111
        gt_lbl += ((gt_lbl == 0) & (self.gt_lbl != 0))
        gt_lbl = self.gt_lbl2rgb (gt_lbl)
        
        masks = []
        for i in range (self.T):
            mask_i = self.mask [i]
            mask_i = np.repeat (np.expand_dims (mask_i, -1), 3, -1).astype (np.uint8)
            masks.append (mask_i)

        max_reward = 7
        rewards = []
        for reward_i in [self.sum_reward] + self.rewards:
            reward_i = ((reward_i + max_reward) / (2 * max_reward) * 255).astype (np.uint8) 

            reward_i = np.repeat (np.expand_dims (reward_i, -1), 3, -1)
            rewards.append (reward_i)
        while (len (rewards) < self.T + 1):
            rewards.append (np.zeros_like (rewards [0]))

        line1 = [raw, lbl,gt_lbl, ] + masks
        while (len (rewards) < len (line1)):
            rewards = [np.zeros_like (rewards [-1])] + rewards
        line1 = np.concatenate (line1, 1)
        line2 = np.concatenate (rewards, 1)
        ret = np.concatenate ([line1, line2], 0)

        return ret

class EM_env (General_env):
    def __init__ (self, raw_list, config, type, gt_lbl_list=None, obs_format="CHW", seed=0):
        self.type = type
        self.raw_list = raw_list
        self.gt_lbl_list = gt_lbl_list
        self.rng = np.random.RandomState(seed)
        self.config = config
        self.obs_format = obs_format
        self.init (config)

    def random_crop (self, size, imgs):
        y0 = self.rng.randint (imgs[0].shape[0] - size[0] + 1)
        x0 = self.rng.randint (imgs[0].shape[1] - size[1] + 1)
        ret = []
        for img in imgs:
            ret += [img[y0:y0+size[0], x0:x0+size[1]]]
        return ret


    def reset (self, model=None, gpu_id=0):
        self.step_cnt = 0
        z0 = self.rng.randint (0, len (self.raw_list))
        self.raw = np.array (self.raw_list [z0], copy=True)
        if (self.gt_lbl_list is not None):
            self.gt_lbl = copy.deepcopy (self.gt_lbl_list [z0])
        else:
            self.gt_lbl = np.zeros_like (self.raw)
        columns = 2
        rows = 2
        # print (self.raw.dtype)
        # fig=plt.figure(figsize=(8, 8))
        # fig.add_subplot(rows, columns, 1)
        # plt.title ("raw")
        # plt.imshow (self.raw)
        # fig.add_subplot(rows, columns, 2)
        # plt.imshow (self.gt_lbl, cmap='tab20')
        self.raw, self.gt_lbl = self.aug (self.raw, self.gt_lbl)
        # fig.add_subplot(rows, columns, 3)
        # plt.imshow (self.raw)
        # fig.add_subplot(rows, columns, 4)
        # plt.imshow (self.gt_lbl, cmap='tab20')
        # plt.show ()

        self.raw, self.gt_lbl = self.random_crop (self.size, [self.raw, self.gt_lbl])
        # self.gt_lbl = label (self.gt_lbl > 0, connectivity=1).astype (np.int32)
        # self.gt_lbl = self.shuffle_lbl (self.gt_lbl)
        # self.gt_lbl_cp = np.pad (self.gt_lbl, self.r, 'constant', constant_values=0)
        self.mask = np.zeros ([self.T] + self.size, dtype=np.float32)
        self.lbl = np.zeros (self.size, dtype=np.int32)
        self.sum_reward = np.zeros (self.size, dtype=np.float32)
        self.rewards = []

        # plt.imshow (self.lbl, cmap="tab20")
        # plt.show ()

        self.reset_end ()
        return self.observation ()

    def set_sample (self, idx, resize=False):
        self.step_cnt = 0
        z0 = idx
        while (self.raw_list [z0].shape [0] < self.size [0] \
            or self.raw_list [z0].shape [1] < self.size [1]):
            z0 = self.rng.randint (len (self.raw_list))
        self.raw = np.array (self.raw_list [z0], copy=True)
        if self.gt_lbl_list is not None:
            self.gt_lbl = np.array (self.gt_lbl_list [z0], copy=True)
        else:
            self.gt_lbl = np.zeros (self.size, dtype=np.int32)


        # print (self.raw.shape, self.gt_lbl.shape)

        # plt.imshow (self.gt_lbl)
        # plt.show ()
        if (not resize):
            if self.gt_lbl_list is not None:
                self.raw, self.gt_lbl = self.random_crop (self.size, [self.raw, self.gt_lbl])
            else:
                self.raw = self.random_crop (self.size, [self.raw]) [0]
        else:
            self.raw = cv2.resize (self.raw, (self.size [1], self.size[0]), interpolation=cv2.INTER_NEAREST)
            self.gt_lbl = cv2.resize (self.gt_lbl, (self.size [1], self.size [0]), interpolation=cv2.INTER_NEAREST)

        # print (self.raw.shape)

        # print (self.raw.shape, self.gt_lbl.shape)

        # fig=plt.figure(figsize=(8, 8))
        # fig.add_subplot (1, 2, 1)
        # plt.imshow (self.raw)
        # fig.add_subplot (1, 2, 2)
        # plt.imshow (self.gt_lbl)
        # plt.show ()

        # plt.imshow (self.gt_lbl)
        # plt.show ()

        self.mask = np.zeros ([self.T] + self.size, dtype=np.float32)
        self.lbl = np.zeros (self.size, dtype=np.int32)
        self.sum_reward = np.zeros (self.size, dtype=np.float32)
        self.rewards = []
        self.reset_end ()
        return self.observation ()