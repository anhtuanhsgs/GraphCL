from environment import *
from Utils.img_aug_func import *
from natsort import natsorted
from skimage import io
from skimage.measure import label
import matplotlib.pyplot as plt
from torch.autograd import Variable
from models.models import *

import torch
import torch.nn as nn
import torch.optim as optim

from utils import *


IN_CH = 2
OUT_CH = 1
FEATURES = [16, 32, 64, 128]
continuous = True

gpu_id = 0
def setup_env_conf ():

    env_conf = {
        "T": 3,
        "size": [96, 96],
        "env_gpu": 0,
        "num_segs": 12,
        "radius": 22,
        "speed": 2,
        # "reward": "gaussian",
        # "reward": "normal",
        "reward": "density",
        "use_lbl": True
    }
    return env_conf

env_config = setup_env_conf ()
env = Voronoi_env (env_config)

def obs2tensor (obs):
    ret = obs [None]
    if gpu_id >= 0:
        ret = torch.tensor (ret, dtype=torch.float32).cuda ()
    return ret

def setup_rl_model (env, env_conf):
    pass

done = False
cnt = 0

obs = env.reset ()
plt.imshow (env.render ())
plt.show ()

# model = setup_rl_model (env, env_config)

sum_score = 0
partition = 2

while not done:

    # obs_t = obs2tensor (obs)
    size_patch = env_config ["size"][0] // partition
    action = np.zeros (env_config ["size"])
    action_tmp = np.zeros ((partition, partition))
    print ("action = ")
    for i in range (partition):
        line = input ()
        line = line.split ()
        for j in range (partition):
            action_tmp [i, j] = int (line [j])
            i_base = i*size_patch
            j_base = j*size_patch
            action [i_base: i_base + size_patch,
                    j_base: j_base + size_patch] = action_tmp [i, j]
    # print (action_tmp)
    # plt.imshow (action)
    # plt.show ()
    
    obs, reward, done, info = env.step (action)
    tmp = []
    for c in range (len (obs)):
        tmp += [obs [c]]
    tmp = np.concatenate (tmp, 1)
    # print ("reward:", reward.shape)
    # print ("done: ", done)
    plt.imshow (tmp, cmap='gray')
    plt.show ()
    plt.imshow (reward, cmap='gray')
    plt.show ()
    # plt.imshow (env.render ())
    # plt.show ()
    sum_score += reward
    # plt.imshow (env.render ())
    # plt.show ()
    # print ('action: ', action) 
    # print ('reward: ', reward)
plt.imshow (env.sum_reward)
plt.show ()
plt.imshow (env.render ())
plt.show ()
