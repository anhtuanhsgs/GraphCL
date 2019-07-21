from __future__ import print_function, division
import os, sys, glob, time
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import torch
import torch.multiprocessing as mp
from environment import *
from utils import get_cell_prob, get_data
from models.models import *
from train import train
from test import test
from natsort import natsorted

from Utils.img_aug_func import *
from Utils.utils import *
from skimage.measure import label
from shared_optim import SharedRMSprop, SharedAdam
from models.models import *

# python main.py --env EM_env_cpu_reward_7 --gpu-id 0 1 2 3 4 5 6 7 --workers 12 --lbl-agents 0 \--num-steps 6 --max-episode-length 6 --reward normal --model AttUNet --merge_radius 16 --merge_speed 2 --split_radius 64 --split_speed 4  --use-lbl --size 128 128 --hidden-feat 2  --log-period 10 --features 16 32 64 128 256 --downsample 3 --data cvppp


parser = argparse.ArgumentParser(description='A3C')
parser.add_argument(
    '--env',
    default='Voronoi_env',
    metavar='ENV',
    help='environment to train on (default: Voronoi_env)')

parser.add_argument(
    '--lr',
    type=float,
    default=0.0001,
    metavar='LR',
    help='learning rate (default: 0.0001)')

parser.add_argument(
    '--gamma',
    type=float,
    default=1,
    metavar='G',
    help='discount factor for rewards (default: 1)')

parser.add_argument(
    '--tau',
    type=float,
    default=1.00,
    metavar='T',
    help='parameter for GAE (default: 1.00)')

parser.add_argument(
    '--seed',
    type=int,
    default=1,
    metavar='S',
    help='random seed (default: 1)')

parser.add_argument(
    '--workers',
    type=int,
    default=4,
    metavar='W',
    help='how many training processes to use (default: 32)')

parser.add_argument(
    '--num-steps',
    type=int,
    default=1,
    metavar='NS',
    help='number of forward steps in A3C (default: 20)')

parser.add_argument(
    '--max-episode-length',
    type=int,
    default=3,
    metavar='M',
    help='maximum length of an episode (default: 10000)')

parser.add_argument(
    '--save-max',
    default=True,
    metavar='SM',
    help='Save model on every test run high score matched or bested')

parser.add_argument(
    '--load', default=False, metavar='L', help='load a trained model')

parser.add_argument(
    '--optimizer',
    default='Adam',
    metavar='OPT',
    help='shares optimizer choice of Adam or RMSprop')

parser.add_argument(
    '--load-model-dir',
    default='trained_models/',
    metavar='LMD',
    help='folder to load trained models from')

parser.add_argument(
    '--save-model-dir',
    default='trained_models/',
    metavar='SMD',
    help='folder to save trained models')

parser.add_argument(
    '--log-dir', default='logs/', metavar='LG', help='folder to save logs')

parser.add_argument(
    '--gpu-ids',
    type=int,
    default=-1,
    nargs='+',
    help='GPUs to use [-1 CPU only] (default: -1)')

parser.add_argument(
    '--amsgrad',
    default=True,
    metavar='AM',
    help='Adam optimizer amsgrad parameter')

parser.add_argument(
    '--save-period',
    type=int,
    default=100,
    metavar='SP',
    help='Save period')

parser.add_argument(
    '--log-period',
    type=int,
    default=10,
    metavar='LP',
    help='Log period')

parser.add_argument (
    '--train-log-period',
    type=int,
    default=16,
    metavar='TLP',
)

parser.add_argument(
    '--shared-optimizer',
    action='store_true'
)

parser.add_argument (
    '--hidden-feat',
    type=int,
    default=64,
    metavar='HF'
)

parser.add_argument (
    '--erosion',
    type=float,
    default=1.1
)

parser.add_argument (
    '--radius',
    type=int,
    default=[16, 48, 96],
    nargs='+'
)

parser.add_argument (
    '--merge_radius',
    type=int,
    default=[16, 48, 96],
    nargs='+'
)

parser.add_argument (
    '--split_radius',
    type=int,
    default=[16, 48, 96],
    nargs='+'
)

parser.add_argument (
    '--speed',
    type=int,
    default=[1, 2, 4],
    nargs='+'
)

parser.add_argument (
    '--merge_speed',
    type=int,
    default=[1, 2, 4],
    nargs='+'
)

parser.add_argument (
    '--split_speed',
    type=int,
    default=[1, 2, 4],
    nargs='+'
)

parser.add_argument (
    '--features',
    type=int,
    default= [32, 64, 128, 256],
    nargs='+'
)

parser.add_argument (
    '--size',
    type=int,
    default= [96, 96],
    nargs='+'
)

parser.add_argument (
    '--model',
    default='UNet',
    choices=['UNet', 'FusionNetLstm', "FusionNet", "UNetLstm", "FCN_GRU", "UNetGRU", 
                "DilatedUNet", "UNetEX", "UNetFuse", "AttUNet", "ASPPAttUNet"]
)

parser.add_argument (
    "--reward",
    default="normal",
    choices=["normal", "gaussian", "density", "seg"]
)

parser.add_argument (
    "--use-lbl",
    action="store_true"
)

parser.add_argument (
    "--use-masks",
    action="store_true"
)

parser.add_argument (
    "--one-step",
    type=int,
    default=None
)

parser.add_argument (
    '--downsample',
    type=int,
    default=1
)

parser.add_argument (
    '--cell-norm',
    action="store_true"
)

parser.add_argument (
    '--DEBUG',
    action="store_true"
)

parser.add_argument (
    '--data',
    default='snemi',
    choices=['syn', 'snemi', 'voronoi', 'zebrafish', 'cvppp', 'sb2018', 'kitti']
)

parser.add_argument (
    '--SEMI_DEBUG',
    action="store_true"
)

parser.add_argument (
    '--reward-gpu',
    action="store_true"
)

parser.add_argument (
    '--eps-lbl',
    default=[0.1, 0.1, 0.1, 0.1],
    type=float,
    nargs='+',
)

parser.add_argument (
    '--eps-lbl-step',
    default=[750, 1500, 2500, 3500],
    type=int,
    nargs='+',
)

parser.add_argument (
    '--rand-step-prob',
    default=0.0,
    type=float
)

# parser.add_argument (
#     '--lbl-action-ratio',
#     type=float,
#     default=0.125
# )

parser.add_argument (
    '--lbl-agents',
    type=int,
    default=0
)

def setup_env_conf (args):
    if args.one_step:
        args.max_episode_length = 1

    env_conf = {
        "T": args.max_episode_length,
        "size": args.size,
        "num_segs": 12,
        "radius": args.radius,
        "split_radius": args.split_radius,
        "merge_radius": args.merge_radius,
        "speed": args.speed,
        "merge_speed": args.merge_speed,
        "split_speed": args.split_speed,
        "reward": args.reward,
        "use_lbl": args.use_lbl,
        "use_masks": args.use_masks,
        "cell_norm": args.cell_norm,
        "DEBUG": args.DEBUG,
        "rand_step_p": args.rand_step_prob,
        "erosion": args.erosion,
    }
    env_conf ["observation_shape"] = [env_conf ["T"] + 1] + env_conf ["size"]
    if args.one_step:
        env_conf ["max_lbl"] = args.one_step
    args.env += "_" + args.model
    env_conf ["data_chan"] = args.data_channel 
    if args.use_lbl:
        args.env += "_lbl"
        env_conf ["observation_shape"][0] = args.data_channel + 2 #Raw, lbl, stop
    if args.use_masks:
        args.env += "_masks"
        env_conf ["observation_shape"][0] += env_conf ["T"]

    args.env += "_" + args.reward
    args.log_dir += args.env + "/"
    args.save_model_dir += args.env + "/"
    create_dir (args.save_model_dir)
    return env_conf
 
def setup_data (args):
    path_test = None
    if args.data == 'syn':
        path_train = 'Data/syn/'
        path_valid = 'Data/syn/'
        args.data_channel = 1
    if args.data == 'snemi':
        path_train = 'Data/snemi/train/'
        path_valid = 'Data/snemi/test/'
        path_test = 'Data/snemi/test/'
        args.data_channel = 1
    if args.data == "zebrafish":
        path_train = "Data/Zebrafish/train/"
        path_valid = "Data/Zebrafish/valid/"
        args.data_channel = 1
    if args.data == "cvppp":
        path_train = "Data/CVPPP_Challenge/train/"
        path_valid = "Data/CVPPP_Challenge/train/"
        path_test = "Data/CVPPP_Challenge/test/"
        args.data_channel = 3
    if args.data == 'sb2018':
        path_train = "Data/ScienceBowl2018/train/"
        path_valid = "Data/ScienceBowl2018/train/"
        args.data_channel = 3
    if args.data == 'kitti':
        path_train = "Data/kitti/train/"
        path_valid = "Data/kitti/train/"
        path_test = "Data/kitti/test/"
        args.data_channel = 3

    relabel = args.data not in ['cvppp', 'sb2018', 'kitti']
    
    raw, gt_lbl = get_data (path=path_train, relabel=relabel)
    raw_valid, gt_lbl_valid = get_data (path=path_valid, relabel=relabel)

    raw_test = None
    gt_lbl_test = None
    if path_test is not None:
        raw_test, _ = get_data (path=path_test, relabel=relabel)

    if (args.DEBUG):
        size = args.size [0] * args.downsample
        raw = raw[:1,:size,:size]
        gt_lbl = gt_lbl [:1,:size,:size]
        raw_valid = np.copy (raw)
        gt_lbl_valid = np.copy (gt_lbl)

    if (args.SEMI_DEBUG):
        size = args.size [0] * args.downsample
        raw = raw[:3]
        gt_lbl = gt_lbl [:3]
    print (raw.shape, gt_lbl.shape)

    return raw, gt_lbl, raw_valid, gt_lbl_valid, raw_test, gt_lbl_test

if __name__ == '__main__':
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    if args.gpu_ids == -1:
        args.gpu_ids = [-1]
    else:
        torch.cuda.manual_seed(args.seed)
        mp.set_start_method('spawn')

    if "EM_env" in args.env:
        raw, gt_lbl, raw_valid, gt_lbl_valid, raw_test, gt_lbl_test = setup_data (args)
        ds = args.downsample
        if args.downsample:
            raw = raw [:, ::ds, ::ds]
            gt_lbl = gt_lbl [:, ::ds, ::ds]
            raw_valid = raw_valid [:, ::ds, ::ds]
            gt_lbl_valid = gt_lbl_valid [:, ::ds, ::ds]
            if raw_test is not None:
                raw_test = raw_test [:, ::ds, ::ds]

    env_conf = setup_env_conf (args)

    num_actions = 2
    if args.one_step:
        num_actions = args.one_step

    if (args.model == 'UNet'):
        shared_model = UNet (env_conf ["observation_shape"][0], args.features, num_actions)
    elif (args.model == "FusionNetLstm"):
        shared_model = FusionNetLstm (env_conf ["observation_shape"], args.features, num_actions, args.hidden_feat)
    elif (args.model == "FusionNet"):
        shared_model = FusionNet (env_conf ["observation_shape"][0], args.features, num_actions)
    elif (args.model == "UNetLstm"):
        shared_model = UNetLstm (env_conf ["observation_shape"], args.features, num_actions, args.hidden_feat)
    elif (args.model == "FCN_GRU"):
        shared_model = DilatedFCN_GRU (env_conf ["observation_shape"], args.features, num_actions, args.hidden_feat)
    elif (args.model == "UNetGRU"):
        shared_model = UNetGRU (env_conf ["observation_shape"], args.features, num_actions, args.hidden_feat)
    elif (args.model == "DilatedUNet"): 
        shared_model = DilatedUNet (env_conf ["observation_shape"][0], args.features, num_actions)
    elif (args.model == "UNetEX"):
        shared_model = UNetEX (env_conf ["observation_shape"][0], args.features, num_actions)
    elif (args.model == "UNetFuse"):
        shared_model = UNetFuse (env_conf ["observation_shape"][0], args.features, num_actions)
    elif (args.model == "AttUNet"):
        shared_model = AttU_Net (env_conf ["observation_shape"][0], args.features, num_actions, split=args.data_channel)
    elif (args.model == "ASPPAttUNet"):
        shared_model = ASPPAttU_Net (env_conf ["observation_shape"][0], args.features, num_actions, split=args.data_channel)

    if args.load:
        saved_state = torch.load(
            args.load,
            map_location=lambda storage, loc: storage)
        shared_model.load_state_dict(saved_state)
    shared_model.share_memory()
    
    if args.shared_optimizer:
        if args.optimizer == 'RMSprop':
            optimizer = SharedRMSprop(shared_model.parameters(), lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = SharedAdam(
                shared_model.parameters(), lr=args.lr, amsgrad=args.amsgrad)
        optimizer.share_memory()
    else:
        optimizer = None

    processes = []
    if "EM_env" in args.env:
        if raw_test is not None:
            p = mp.Process(target=test, args=(args, shared_model, env_conf, [raw_valid, gt_lbl_valid], True, raw_test))
        else:
            p = mp.Process(target=test, args=(args, shared_model, env_conf, [raw_valid, gt_lbl_valid], True))
    else:
        p = mp.Process(target=test, args=(args, shared_model, env_conf))
    p.start()
    processes.append(p)
    time.sleep(0.1)

    for rank in range(0, args.workers):
        if "EM_env" in args.env:
            p = mp.Process(
                target=train, args=(rank, args, shared_model, optimizer, env_conf, [raw, gt_lbl]))
        else:
             p = mp.Process(
                target=train, args=(rank, args, shared_model, optimizer, env_conf))
        p.start()
        processes.append(p)
        time.sleep(0.1)

    for p in processes:
        time.sleep(0.1)
        p.join()

