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
from utils import guassian_weight_map, density_map, malis_rand_index, malis_f1_score, adjusted_rand_index
from skimage.draw import line_aa
from misc.Voronoi import *

debug = True

class General_env (gym.Env):
    def init (self, config):
        self.T = config ['T']
        self.r = config ["radius"]
        self.size = config ["size"]
        self.speed = config ["speed"]
        if config ["use_lbl"]:
            self.observation_space = Box (0, 1.0, shape=[config["observation_shape"][0]] + self.size, dtype=np.float32)
        else:
            self.observation_space = Box (-1.0, 1.0, shape=[config["observation_shape"][0]] + self.size, dtype=np.float32)
        self.rng = np.random.RandomState(time_seed ())
        self.max_lbl = 2 ** self.T - 1

    def step (self, action):
        self.action = action
        self.new_lbl = self.lbl + action * (2 ** (self.T - self.step_cnt - 1))
        done = False

        reward = np.zeros (self.size, dtype=np.float32)
        reward = self.middle_step_reward (density=self.density)  
                
        self.lbl = self.new_lbl
        self.mask [self.step_cnt:self.step_cnt+1] += (2 * action - 1) * 255

        reward += self.foregr_backgr_reward () / self.T
        if self.step_cnt == 0:
            reward += self.first_step_reward () * 0.2

        if self.T == 1:
            self.mask [self.step_cnt:self.step_cnt+1] += (2 * action - (self.config ["max_lbl"] - 1)) * 255

        self.step_cnt += 1
        info = {}
        if self.config ["quality"]:
            current_score = self.merge_quality () * self.split_quality () * self.cell_quality ()
            reward = current_score - self.old_score
            self.old_score = current_score

        if self.step_cnt >= self.T:
            done = True
            # reward += self.merge_quality () * self.split_quality ()
            # reward += self.foregr_backgr_reward () * 0.2
            reward += self.final_step_reward (density=self.density, last_step=(self.step_cnt>=self.T)) 

        if self.config ["cell_norm"]:
            reward = reward * self.norm_map
        self.rewards.append (reward)    
        self.sum_reward += reward

        return self.observation (), reward, done, info

    def shuffle_lbl (self, lbl):
        per = list (range (1, np.max (lbl) + 1))
        shuffle(per)
        per = [0] + per
        vf = np.vectorize (lambda x: per[x])
        return vf (lbl)

    def reset_end (self):
        """
            Must call after reset
        """
        self.w_map = None
        self.density = None
        if self.config ["reward"] == "density":
            self.density = density_map (self.gt_lbl)
        else:
            self.density = np.ones (self.size, dtype=np.float32)
        if self.config ["cell_norm"]:
            self.norm_map = np.zeros (self.size, dtype=np.float32)
            for lbl in np.unique (self.gt_lbl):
                cell_size = np.count_nonzero (self.gt_lbl == lbl)
                window_area =  self.size[0] * self.size[1]
                log_window_area = np.log (window_area)
                log_cel_size = np.log (cell_size)
                self.norm_map += (self.gt_lbl == lbl) * (1.0 - min (cell_size * 2.5, window_area * 0.8) / window_area)
                # self.norm_map += (self.gt_lbl == lbl) * (1.0 - log_cel_size / log_window_area)
        if self.config ["quality"]:
            self.new_lbl = np.zeros (self.size, dtype=np.int32)
            self.old_score = self.merge_quality () * self.split_quality () * self.cell_quality ()
            # print ("old_score")
            # plt.imshow (self.old_score)
            # plt.show ()

    def get_I (self, lbl_cp, new_lbl_cp, yr, xr, r):
        y_base = r + yr; x_base = r + xr
        I = self.new_lbl == new_lbl_cp [y_base:y_base+self.size[0], x_base:x_base+self.size[1]]
        I_hat = self.gt_lbl == self.gt_lbl_cp [y_base:y_base+self.size[0], x_base:x_base+self.size[1]]
        I_old = self.lbl == lbl_cp [y_base:y_base+self.size[0], x_base:x_base+self.size[1]]
        return I, I_hat, I_old

    def tofloat (self, arrs):
        ret = []
        for arr in arrs:
            ret.append (arr.astype (np.float32))
        return ret

    def merge_quality (self):
        # Based on IOU
        lbl_cp = np.pad (self.lbl, self.r, 'constant', constant_values=0)
        new_lbl_cp = np.pad (self.new_lbl, self.r, 'constant', constant_values=0)
        self.gt_lbl_cp = np.pad (self.gt_lbl, self.r, 'constant', constant_values=0)
        ret = np.zeros (self.size, dtype=np.float32)
        r = self.r
        intersection = np.zeros (self.size, dtype=np.float32)
        union = np.zeros (self.size, dtype=np.float32)
        for yr in range (-r, r + 1, self.speed):
            for xr in range (-r, r + 1, self.speed):
                if (yr == 0 and xr == 0):
                    continue
                y_base = r + yr; x_base = r + xr
                I, I_hat, I_old = self.get_I (lbl_cp, new_lbl_cp, yr, xr, r)
                intersection += (I_hat == True) & (I == True)
                union += ((I_hat == True) | (I == True))
        union += 1
        ret = intersection / union
        return ret

    def split_quality (self):
        # Based on IOU
        lbl_cp = np.pad (self.lbl, self.r, 'constant', constant_values=0)
        new_lbl_cp = np.pad (self.new_lbl, self.r, 'constant', constant_values=0)
        self.gt_lbl_cp = np.pad (self.gt_lbl, self.r, 'constant', constant_values=0)
        ret = np.zeros (self.size, dtype=np.float32)
        r = self.r
        intersection = np.zeros (self.size, dtype=np.float32)
        union = np.zeros (self.size, dtype=np.float32)
        for yr in range (-r, r + 1, self.speed):
            for xr in range (-r, r + 1, self.speed):
                if (yr == 0 and xr == 0):
                    continue
                y_base = r + yr; x_base = r + xr
                I, I_hat, I_old = self.get_I (lbl_cp, new_lbl_cp, yr, xr, r)
                intersection += (I_hat == False) & (I == False)
                union += ((I_hat == False) | (I == False))
        union += 1
        ret = intersection / union
        return ret

    def cell_quality (self):
        new_lbl = self.new_lbl
        gt_lbl = self.gt_lbl
        ret = np.zeros (self.size, dtype=np.float32)
        gt_lbl_values = np.unique (gt_lbl)
        lbl_values = np.unique (new_lbl)
        #Union
        for gt_value in gt_lbl_values:
            single_cell_map = gt_lbl == gt_value
            for value in lbl_values:
                value_map = value == new_lbl
                intersection = single_cell_map * (value_map * np.sum (value_map * single_cell_map))
                union = np.sum (single_cell_map | value_map) + 1
                ret += intersection / union
        return ret

    def first_step_reward (self, density=None):
        reward = np.zeros (self.size, dtype=np.float32)
        foregr_ratio = 0.67
        reward += ((self.new_lbl != 0) & (self.gt_lbl != 0)) * (1.0 - foregr_ratio)
        reward += ((self.new_lbl == 0) & (self.gt_lbl == 0)) * (foregr_ratio)
        reward -= ((self.new_lbl == 0) & (self.gt_lbl != 0)) * (1.0 - foregr_ratio)
        reward -= ((self.new_lbl != 0) & (self.gt_lbl == 0)) * (foregr_ratio)
        return reward

    def middle_step_reward (self, density=None):
        lbl_cp = np.pad (self.lbl, self.r, 'constant', constant_values=0)
        new_lbl_cp = np.pad (self.new_lbl, self.r, 'constant', constant_values=0)
        self.gt_lbl_cp = np.pad (self.gt_lbl, self.r, 'constant', constant_values=0)
        density_cp = np.pad (self.density, self.r, 'constant', constant_values=0)
        reward = np.zeros (self.size, dtype=np.float32)
        r = self.r
        I_hat_true_cnt = np.zeros (self.size, dtype=np.float32)
        I_hat_false_cnt = np.zeros (self.size, dtype=np.float32)
        true_split_reward = np.zeros (self.size, dtype=np.float32)
        false_split_penalty = np.zeros (self.size, dtype=np.float32)
        first_step = self.step_cnt == 0
        for yr in range (-r, r + 1, self.speed):
            for xr in range (-r, r + 1, self.speed):
                if (yr == 0 and xr == 0):
                    continue
                y_base = r + yr; x_base = r + xr
                I, I_hat, I_old = self.get_I (lbl_cp, new_lbl_cp, yr, xr, r)
                density_v = density_cp [y_base:y_base+self.size[0], x_base:x_base+self.size[1]]
                density_u = density
                I_hat_true_cnt += density_v * (I_hat == True)
                I_hat_false_cnt += density_v * (I_hat == False)
                true_split_reward += density_u * ((I_hat == False) & (I == False) & ((I_old == True) | first_step)) * density_v
                false_split_penalty += density_u * ((I_hat == True) & (I == False) & ((I_old == True) | first_step)) * density_v

        I_hat_false_cnt += (I_hat_false_cnt == 0)
        I_hat_true_cnt += (I_hat_true_cnt == 0)
        reward += true_split_reward / I_hat_false_cnt
        reward -= false_split_penalty / I_hat_true_cnt
        return reward

    def final_step_reward (self, density=None, last_step=False):
        lbl_cp = np.pad (self.lbl, self.r, 'constant', constant_values=0)
        new_lbl_cp = np.pad (self.new_lbl, self.r, 'constant', constant_values=0)
        self.gt_lbl_cp = np.pad (self.gt_lbl, self.r, 'constant', constant_values=0)
        density_cp = np.pad (self.density, self.r, 'constant', constant_values=0)
        reward = np.zeros (self.size, dtype=np.float32)
        r = self.r
        I_hat_true_cnt = np.zeros (self.size, dtype=np.float32)
        I_hat_false_cnt = np.zeros (self.size, dtype=np.float32)
        true_merge_reward = np.zeros (self.size, dtype=np.float32)
        false_merge_penalty = np.zeros (self.size, dtype=np.float32)

        for yr in range (-r, r + 1, self.speed):
            for xr in range (-r, r + 1, self.speed):
                if (yr == 0 and xr == 0):
                    continue
                y_base = r + yr; x_base = r + xr
                I, I_hat, I_old = self.get_I (lbl_cp, new_lbl_cp, yr, xr, r)
                density_v = density_cp [y_base:y_base+self.size[0], x_base:x_base+self.size[1]]
                density_u = density
                I_hat_true_cnt += density_v * (I_hat == True)
                I_hat_false_cnt += density_v * (I_hat == False)
                true_merge_reward += density_u * ((I_hat == True) & (I == True)) * density_v
                false_merge_penalty += density_u * ((I_hat == False) & (I == True)) * density_v
        
        I_hat_false_cnt += (I_hat_false_cnt == 0)
        I_hat_true_cnt += (I_hat_true_cnt == 0)
        reward -= (false_merge_penalty / I_hat_false_cnt)
        reward += true_merge_reward / I_hat_true_cnt
        return reward

    def calculate_reward_step_kernel (self):
        size = self.config["ker_size"]
        step = self.config["ker_step"]
        full_size = self.size
        y0 = 0; endY = False
        x0 = 0; endX = False
        reward = np.zeros (self.size, dtype=np.float32)
        cnt = np.zeros (self.size, dtype=np.float32)
        while y0 < full_size [0]:
            if y0 + size >= full_size [0]:
                y0 = full_size [0] - size
                endY = True
            while x0 < full_size [1]:
                if x0 + size >= full_size [1]:
                    x0 = full_size [1] - size
                    endX = True
                patch_lbl = self.lbl [y0:y0+size, x0:x0+size].astype (np.int32)
                patch_new_lbl = self.new_lbl [y0:y0+size, x0:x0+size].astype (np.int32)
                patch_gt_lbl = self.gt_lbl [y0:y0+size, x0:x0+size].astype (np.int32)
                old_score = adjusted_rand_index (patch_gt_lbl, patch_lbl)
                score = adjusted_rand_index (patch_gt_lbl, patch_new_lbl)
                cnt [y0:y0+size, x0:x0+size] += 1
                reward [y0:y0+size, x0:x0+size] += score - old_score
                x0 += step
                if endX:
                    x0 = 0; endX = False
                    break
            y0 += step
            if endY:
                break

        reward = reward / cnt
        return reward

    def foregr_backgr_reward (self):
        reward = np.zeros (self.size, dtype=np.float32)
        # foregr_ratio = np.count_nonzero (self.gt_lbl) / np.prod (self.gt_lbl.shape)
        foregr_ratio = 0.67
        # backgr reward, penalty
        reward += ((self.lbl == 0) & (self.gt_lbl == 0)).astype (np.float32) * foregr_ratio
        reward -= ((self.lbl != 0) & (self.gt_lbl == 0)).astype (np.float32) * foregr_ratio
        # foregr reward, penalty
        reward += ((self.lbl != 0) & (self.gt_lbl != 0)).astype (np.float32) * (1 - foregr_ratio)
        reward -= ((self.lbl == 0) & (self.gt_lbl != 0)).astype (np.float32) * (1 - foregr_ratio)
        return reward

    def observation (self):
        if not self.config ["use_lbl"]:
            
            obs = np.concatenate ([
                    self.raw [None] * 2 - 255.0,
                    self.mask,
                ], 0)

        else:
            lbl = self.lbl / self.max_lbl * 255.0
            done_mask = np.zeros_like (self.raw)
            if self.step_cnt >= self.T:
                done_mask += 255.0


            if self.config ["use_masks"]:
                obs = np.concatenate ([
                        self.raw [None].astype (np.float32),
                        lbl [None], 
                        done_mask [None],
                        (self.mask + 255.0) / 2
                    ])
            else:
                obs = np.concatenate ([
                        self.raw [None].astype (np.float32),
                        lbl [None], 
                        done_mask [None],
                        # (self.mask + 255.0) / 2
                    ])
        if self.obs_format == "CHW":
            ret = obs.astype (np.float32) / 255.0
        else:
            ret = np.transpose (obs, [1, 2, 0]) / 255.0
        return ret

    def render (self):
        raw = np.repeat (np.expand_dims (self.raw, -1), 3, -1).astype (np.uint8)
        lbl = self.lbl.astype (np.int32)

        lbl = lbl2rgb (lbl)
        gt_lbl = lbl2rgb (self.gt_lbl)
        
        masks = []
        for i in range (self.T):
            mask_i = self.mask [i]
            mask_i = np.repeat (np.expand_dims (mask_i, -1), 3, -1).astype (np.uint8)
            masks.append (mask_i)

        max_reward = 3
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

class Voronoi_env (General_env):
    def __init__ (self, config, obs_format="CHW"):
        self.config = config
        self.obs_format = obs_format
        self.type = "train"
        self.init (config)

    def init (self, config):
        super ().init (config)
        self.num_segs = config ["num_segs"]

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

        prob = get_boudary (self.raw [None], 0, 0) [0].astype (np.float32)
        self.gt_lbl = label (prob > 128, connectivity=1).astype (np.int32)
        # self.gt_lbl = self.raw.astype (np.int32)
        self.gt_lbl_cp = np.pad (self.gt_lbl, self.r, 'constant', constant_values=0)
        self.mask = np.zeros ([self.T] + self.size, dtype=np.float32)
        self.lbl = np.zeros (self.size, dtype=np.int32)
        self.sum_reward = np.zeros (self.size, dtype=np.float32)
        self.raw = prob
        self.rewards = []
        self.reset_end ()
        return self.observation ()


class EM_env (General_env):
    def __init__ (self, raw_list, config, type, gt_lbl_list=None, obs_format="CHW"):
        self.type = type
        self.raw_list = raw_list.astype (np.float32)
        self.gt_lbl_list = gt_lbl_list
        self.rng = np.random.RandomState(time_seed ())
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

    def reset (self):
        self.step_cnt = 0
        z0 = self.rng.randint (0, len (self.raw_list))
        self.raw = copy.deepcopy (self.raw_list [z0])
        if (self.gt_lbl_list is not None):
            self.gt_lbl = copy.deepcopy (self.gt_lbl_list [z0])
        else:
            self.gt_lbl = np.zeros_like (self.raw)

        self.raw, self.gt_lbl = self.random_crop (self.size, [self.raw, self.gt_lbl])
        self.gt_lbl = label (self.gt_lbl > 0, connectivity=1).astype (np.int32)
        self.gt_lbl = self.shuffle_lbl (self.gt_lbl)
        self.gt_lbl_cp = np.pad (self.gt_lbl, self.r, 'constant', constant_values=0)
        self.mask = np.zeros ([self.T] + self.size, dtype=np.float32)
        self.lbl = np.zeros (self.size, dtype=np.int32)
        self.sum_reward = np.zeros (self.size, dtype=np.float32)
        self.rewards = []
        self.reset_end ()
        return self.observation ()



