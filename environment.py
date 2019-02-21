import os, sys, glob, time, copy
from os import sys, path
import gym
import numpy as np
from collections import deque
from gym.spaces.box import Box
from skimage.measure import label
from sklearn.metrics import adjusted_rand_score
from cv2 import resize
from Utils.utils import *
from Utils.img_aug_func import *
import albumentations as A
import random
from gym.spaces import Box, Discrete, Tuple
import matplotlib.pyplot as plt
from malis import rand_index 
from random import shuffle
from PIL import Image, ImageFilter
from utils import reward_scaler, build_blend_weight
from skimage.draw import line_aa
from misc.Voronoi import *

debug = False

class Voronoi_env (gym.Env):
    def __init__ (self, config, obs_format="CHW"):
        self.config = config
        self.obs_format = obs_format
        self.type = "train"
        self.init (config)

    def init (self, config):
        self.T = config ['T']
        self.num_segs = config ["num_segs"]
        self.r = config ["radius"]
        self.size = config ["size"]
        self.speed = config ["speed"]
        self.max_lbl = self.num_segs + 1
        self.observation_space = Box (-1.0, 1.0, shape=[self.T + 1] + self.size, dtype=np.float32)
        self.rng = np.random.RandomState(time_seed ())

    def reset (self):
        self.step_cnt = 0
        if not debug:
            self.raw = generate_voronoi_diagram (self.config ["size"][0], self.config ["size"][1], self.num_segs, self.rng)
        else:
            size = self.size
            self.raw = np.zeros (self.size)
            half = self.size [0] // 2
            self.raw [2:half, 2:half] = 1
            self.raw [2:half, half:size[1]] = 2
            self.raw [half:size[0], 2:half] = 3
            self.raw [half:size[0], half:size[1]] = 4

        self.prob = get_boudary (self.raw [None], 0, 0) [0].astype (np.float32)
        # self.gt_lbl = label (self.prob > 128).astype (np.int32)
        self.gt_lbl = self.raw.astype (np.int32)
        # plt.imshow (self.gt_lbl)
        # plt.show ()
        # print (self.gt_lbl.dtype)
        self.gt_lbl_cp = np.pad (self.gt_lbl, self.r, 'constant', constant_values=0)
        self.mask = np.zeros ([self.T] + self.size, dtype=np.float32)
        self.lbl = np.zeros (self.size, dtype=np.int32)
        self.sum_reward = np.zeros (self.size, dtype=np.float32)
        return self.observation ()

    def split_penalty_step (self):
        lbl_cp = np.pad (self.lbl, self.r, 'constant', constant_values=0)
        new_lbl_cp = np.pad (self.new_lbl, self.r, 'constant', constant_values=0)
        reward = np.zeros (self.size, dtype=np.float32)
        first_step = np.max (lbl_cp) < 1
        r = self.r
        # Wrongly different label with neighbor pixels
        for yr in range (-r, r + 1, self.speed):
            for xr in range (-r, r + 1, self.speed):
                if (yr == 0 and xr == 0):
                    continue
                y_base = r + yr; x_base = r + xr
                I = self.new_lbl == new_lbl_cp [y_base:y_base+self.size[0], x_base:x_base+self.size[1]]
                I_hat = self.gt_lbl == self.gt_lbl_cp [y_base:y_base+self.size[0], x_base:x_base+self.size[1]]
                I_old = self.lbl == lbl_cp [y_base:y_base+self.size[0], x_base:x_base+self.size[1]]
                reward += (I != I_hat) & (I_hat == True) & ((I_old == I_hat) | first_step)
        reward *= -1
        reward = reward / (((self.r * 2 + 1) / self.speed) ** 2)
        return reward

    def split_reward_step (self):
        lbl_cp = np.pad (self.lbl, self.r, 'constant', constant_values=0)
        new_lbl_cp = np.pad (self.new_lbl, self.r, 'constant', constant_values=0)
        reward = np.zeros (self.size, dtype=np.float32)
        r = self.r
        first_step = np.max (lbl_cp) < 1
        # Correctly different label with neighbor pixels
        for yr in range (-r, r + 1, self.speed):
            for xr in range (-r, r + 1, self.speed):
                if (yr == 0 and xr == 0):
                    continue
                y_base = r + yr; x_base = r + xr
                I = self.new_lbl == new_lbl_cp [y_base:y_base+self.size[0], x_base:x_base+self.size[1]]
                I_hat = self.gt_lbl == self.gt_lbl_cp [y_base:y_base+self.size[0], x_base:x_base+self.size[1]]
                I_old = self.lbl == lbl_cp [y_base:y_base+self.size[0], x_base:x_base+self.size[1]]
                reward += (I == I_hat) & (I_hat == False) & ((I_old != I_hat) | first_step)
        reward = reward / (((self.r * 2 + 1) / self.speed) ** 2)
        return reward

    def split_reward_done (self):
        new_lbl_cp = np.pad (self.new_lbl, self.r, 'constant', constant_values=0)
        reward = np.zeros (self.size, dtype=np.float32)
        r = self.r

        # Wrongly same label with background neighbor pixels
        for yr in range (-r, r + 1, self.speed):
            for xr in range (-r, r + 1, self.speed):
                if yr == 0 and xr == 0:
                    continue
                y_base = r + yr; x_base = r + xr
                I = self.new_lbl == new_lbl_cp [y_base:y_base+self.size[0], x_base:x_base+self.size[1]]
                I_hat = self.gt_lbl == self.gt_lbl_cp [y_base:y_base+self.size[0], x_base:x_base+self.size[1]]
                reward -= (I != I_hat) & (I_hat == False)
        
        # Wrongly not classifed as segment
        reward += (self.gt_lbl != 0) & (self.new_lbl == 0)
        reward *= -1
        # plt.imshow (reward)
        # plt.show ()
        reward = reward / (((self.r * 2 + 1) / self.speed) ** 2)
        return reward

    def merge_reward_done (self):
        new_lbl_cp = np.pad (self.new_lbl, self.r, 'constant', constant_values=0)
        reward = np.zeros (self.size, dtype=np.float32)
        r = self.r

        # Correctly same labels with neighbor pixels
        for yr in range (-r, r + 1, self.speed):
            for xr in range (-r, r + 1, self.speed):
                if yr == 0 and xr == 0:
                    continue
                y_base = r + yr; x_base = r + xr
                I = self.new_lbl == new_lbl_cp [y_base:y_base+self.size[0], x_base:x_base+self.size[1]]
                I_hat = self.gt_lbl == self.gt_lbl_cp [y_base:y_base+self.size[0], x_base:x_base+self.size[1]]
                reward += (I == I_hat) & I_hat & \
                            (((self.gt_lbl != 0) & (self.new_lbl != 0)) | \
                             ((self.gt_lbl != 0) & (self.new_lbl != 0)))

        # Wrongly classified as background
        reward -= (self.gt_lbl != 0) & (self.new_lbl == 0)

        # plt.imshow (reward)
        # plt.show ()
        reward = reward / (((self.r * 2 + 1) / self.speed) ** 2)
        return reward

    def observation (self):
        obs = np.concatenate ([
                self.prob [None] * 2 - 255.0,
                self.mask,
            ], 0)

        if self.obs_format == "CHW":
            ret = obs.astype (np.float32) / 255.0
            return ret 
        else:
            ret = np.transpose (obs, [1, 2, 0]) / 255.0
            return ret

    def step (self, action):
        self.new_lbl = self.lbl * 2 + action
        done = False
        reward = self.split_penalty_step ()
        reward += self.split_reward_step ()

        self.lbl = self.new_lbl
        self.mask [self.step_cnt:self.step_cnt+1] += (2 * action - 1) * 255

        self.step_cnt += 1
        if self.step_cnt >= self.T:
            done = True
            # reward += self.split_reward_done ()
            # reward += self.merge_reward_done ()
        info = {}
        self.sum_reward += reward
        return self.observation (), reward, done, info

    def render (self):
        raw = np.repeat (np.expand_dims (self.raw, -1), 3, -1).astype (np.uint8)
        prob = np.repeat (np.expand_dims (self.prob, -1), 3, -1).astype (np.uint8)
        lbl = self.lbl.astype (np.int32)
        lbl = lbl2rgb (lbl)
        gt_lbl = lbl2rgb (self.gt_lbl)
        masks = []
        for i in range (self.T):
            mask_i = self.mask [i]
            mask_i = np.repeat (np.expand_dims (mask_i, -1), 3, -1).astype (np.uint8)
            masks.append (mask_i)

        ret = np.concatenate ([prob,
                lbl,
                gt_lbl,
            ] + masks, 1)

        return ret

