from __future__ import division
import numpy as np
import torch
import json
import logging
import math as m
from torch.autograd import Variable
from scipy import ndimage as ndi
from natsort import natsorted
import os, sys, glob, time
from Utils.img_aug_func import *
from skimage.measure import label
from skimage.filters import sobel
from malis import rand_index 
from sklearn.metrics import adjusted_rand_score

def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)

def malis_rand_index (gt_lbl, pred_lbl):
    ret = rand_index (gt_lbl, pred_lbl) [0]
    ret = float (ret)
    return ret

def malis_f1_score (gt_lbl, pred_lbl):
    if (np.max (gt_lbl) == 0):
        gt_lbl += 1
    ret = rand_index (gt_lbl, pred_lbl) [1]
    ret = float (ret)
    return ret

def adjusted_rand_index (gt_lbl, pred_lbl):
    gt_lbl = gt_lbl.flatten ()
    pred_lbl = pred_lbl.flatten ()
    return adjusted_rand_score (gt_lbl, pred_lbl)

def build_blend_weight (shape):
    # print ("patch shape = ", shape)
    yy, xx = np.meshgrid (
            np.linspace(-1,1,shape[0], dtype=np.float32),
            np.linspace(-1,1,shape[1], dtype=np.float32)
        )
    d = np.sqrt(xx*xx+yy*yy)
    sigma, mu = 0.5, 0.0
    v_weight = 1e-6+np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
    v_weight = v_weight/v_weight.max()
    return v_weight

def guassian_weight_map (shape):
    # print ("patch shape = ", shape)
    yy, xx = np.meshgrid (
            np.linspace(-1,1,shape[0], dtype=np.float32),
            np.linspace(-1,1,shape[1], dtype=np.float32)
        )
    d = np.sqrt(xx*xx+yy*yy)
    sigma, mu = 0.5, 0.0
    v_weight = 1e-6+np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
    v_weight = v_weight/v_weight.max()
    return v_weight

def read_config(file_path):
    """Read JSON config."""
    json_object = json.load(open(file_path, 'r'))
    return json_object


def norm_col_init(weights, std=1.0):
    x = torch.randn(weights.size())
    x *= std / torch.sqrt((x**2).sum(1, keepdim=True))
    return x


def ensure_shared_grads(model, shared_model, gpu=False):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None and not gpu:
            return
        elif not gpu:
            shared_param._grad = param.grad
        else:
            shared_param._grad = param.grad.cpu()

def density_map (lbl):
    lbl = np.pad (lbl, 1, 'constant', constant_values=0)
    if (np.min (lbl) > 0) or (np.max (lbl) == 0):
        return np.ones (lbl.shape, dtype=np.float32)
    distance = ndi.distance_transform_edt(lbl)
    idx_list = np.unique (lbl)
    
    max_dist = np.max (distance)
    local_peak_dist_list = []
    ret = np.zeros (lbl.shape, dtype=np.float32)
    for i in idx_list:
        if i == 0:
            continue
        local_dist_map = distance * (lbl == i)
        local_peak_dist = np.max (local_dist_map)
        local_peak_dist_list.append (local_peak_dist)
        ret += local_dist_map * (max_dist / local_peak_dist)

    ret = ret / np.max (ret)
    ret = np.clip (ret, 0.33, 1.0) * (ret == 0)
    return ret [1:, 1:][:-1,:-1]

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)

def reward_scaler (r, alpha, beta):
    r = m.pow (alpha, (r * beta)) / m.pow (alpha, 1 * beta)
    return r

def normal(x, mu, sigma, gpu_id, gpu=False):
    pi = np.array([m.pi])
    pi = torch.from_numpy(pi).float()
    if gpu:
        with torch.cuda.device(gpu_id):
            pi = Variable(pi).cuda()
    else:
        pi = Variable(pi)
    a = (-1 * (x - mu).pow(2) / (2 * sigma)).exp()
    b = 1 / (2 * sigma * pi.expand_as(sigma)).sqrt()
    return a * b

def get_cell_prob (lbl, dilation, erosion):
    ESP = 1e-5
    elevation_map = []
    for img in lbl:
        elevation_map += [sobel (img)]
    elevation_map = np.array (elevation_map)
    elevation_map = elevation_map > ESP
    cell_prob = ((lbl > 0) ^ elevation_map) & (lbl > 0)
    for i in range (len (cell_prob)):
        for j in range (erosion):
            cell_prob [i] = binary_erosion (cell_prob [i])
    for i in range (len (cell_prob)):
        for j in range (dilation):
            cell_prob [i] = binary_dilation (cell_prob [i])
    return np.array (cell_prob, dtype=np.uint8) * 255

def get_data (path, args):
    train_path = natsorted (glob.glob(path + 'A/*.tif'))
    train_label_path = natsorted (glob.glob(path + 'B/*.tif'))
    X_train = read_im (train_path)
    y_train = read_im (train_label_path)

    if (len (X_train) > 0):
        X_train = X_train [0]
    if (len (y_train) > 0):
        y_train = y_train [0]
        gt_prob = get_cell_prob (y_train, 0, 0)
        y_train = []
        for img in gt_prob:
            y_train += [label (img).astype (np.int32)]
        y_train = np.array (y_train)
    else:
        y_train = np.zeros_like (X_train)
    return X_train, y_train

if __name__ == "__main__":
    r = float (input ())
    print (reward_scaler (r))