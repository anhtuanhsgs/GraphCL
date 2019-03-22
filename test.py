from __future__ import division
from setproctitle import setproctitle as ptitle
import torch
from environment import *
from utils import setup_logger
from models.models import *
from player_util import Agent
from torch.autograd import Variable
import time
import logging
from Utils.Logger import Logger
from Utils.utils import *
import numpy as np

def test (args, shared_model, env_conf, datasets=None, hasLbl=True):
    if hasLbl:
        ptitle ('Valid agent')
    else:
        ptitle ("Test agent")

    gpu_id = args.gpu_ids [-1]
    env_conf ["env_gpu"] = gpu_id
    log = {}
    logger = Logger (args.log_dir)

    setup_logger ('{}_log'.format (args.env), r'{0}{1}_log'.format (args.log_dir, args.env))
    log['{}_log'.format(args.env)] = logging.getLogger('{}_log'.format(
        args.env))
    d_args = vars (args)

    if hasLbl:
        for k in d_args.keys ():
            log ['{}_log'.format (args.env)].info ('{0}: {1}'.format (k, d_args[k]))

    torch.manual_seed (args.seed)

    if gpu_id >= 0:
        torch.cuda.manual_seed (args.seed)

    if "EM_env" in args.env:
        raw_list, gt_lbl_list = datasets
        env = EM_env (raw_list, env_conf, type="train", gt_lbl_list=gt_lbl_list)
    else:  
        env = Voronoi_env (env_conf)

    reward_sum = 0
    start_time = time.time ()
    num_tests = 0
    reward_total_sum = 0

    player = Agent (None, env, args, None)

    player.gpu_id = gpu_id
    
    if args.model == "UNet":
        player.model = UNet (env.observation_space.shape [0], args.features, 2)
    elif args.model == "FusionNetLstm":
        player.model = FusionNetLstm (env.observation_space.shape, args.features, 2, args.hidden_feat)
    elif args.model == "FusionNet":
        player.model = FusionNet (env.observation_space.shape [0], args.features, 2)
    elif (args.model == "UNetLstm"):
        player.model = UNetLstm (env.observation_space.shape, args.features, 2, args.hidden_feat)

    player.state = player.env.reset ()
    player.state = torch.from_numpy (player.state).float ()
    if gpu_id >= 0:
        with torch.cuda.device (gpu_id):
            player.model = player.model.cuda ()
            player.state = player.state.cuda ()
    player.model.eval ()

    flag = True

    create_dir (args.save_model_dir)

    recent_episode_scores = []
    renderlist = []
    renderlist.append (player.env.render ())
    max_score = 0
    while True:
        if flag:
            if gpu_id >= 0:
                with torch.cuda.device (gpu_id):
                    player.model.load_state_dict (shared_model.state_dict ())
            else:
                player.model.load_state_dict (shared_model.state_dict ())
            player.model.eval ()
            flag = False

        player.action_test ()
        reward_sum += player.reward.mean ()
        renderlist.append (player.env.render ()) 

        if player.done:
            flag = True

            num_tests += 1
            reward_total_sum += reward_sum
            reward_mean = reward_total_sum / num_tests
            if hasLbl:
                log ['{}_log'.format (args.env)].info (
                    "VALID: Time {0}, episode reward {1}, num tests {4}, episode length {2}, reward mean {3:.4f}".
                    format (
                        time.strftime ("%Hh %Mm %Ss", time.gmtime (time.time () - start_time)),
                        reward_sum, player.eps_len, reward_mean, num_tests))

            recent_episode_scores += [reward_sum]
            if len (recent_episode_scores) > 200:
                recent_episode_scores.pop (0)

            if args.save_max and np.mean (recent_episode_scores) >= max_score:
                max_score = np.mean (recent_episode_scores)
                if gpu_id >= 0:
                    with torch.cuda.device (gpu_id):
                        state_to_save = player.model.state_dict ()
                        torch.save (state_to_save, '{0}{1}.dat'.format (args.save_model_dir, 'best_model_' + args.env))

            if num_tests % args.save_period == 0:
                if gpu_id >= 0:
                    with torch.cuda.device (gpu_id):
                        state_to_save = player.model.state_dict ()
                        torch.save (state_to_save, '{0}{1}.dat'.format (args.save_model_dir, args.env + '_' + str (num_tests)))

            if num_tests % args.log_period == 0:
                if hasLbl:
                    print ("----------------------VALID SET--------------------------")
                    print ("Log test #:", num_tests)
                    print ("rewards: ", player.reward.mean ())
                    print ("sum rewards: ", reward_sum)
                    print ("------------------------------------------------")

                log_img = np.concatenate (renderlist, 0)
                if hasLbl:
                    log_info = {"valid_sample": log_img}
                else:
                    log_info = {"test_sample": log_img}

                for tag, img in log_info.items ():
                    img = img [None]
                    logger.image_summary (tag, img, num_tests)

                if hasLbl:
                    log_info = {'mean_valid_reward': reward_mean}
                    for tag, value in log_info.items ():
                        logger.scalar_summary (tag, value, num_tests)

            renderlist = []
            reward_sum = 0
            player.eps_len = 0
            
            player.clear_actions ()
            state = player.env.reset ()
            renderlist.append (player.env.render ())
            time.sleep (15)
            player.state = torch.from_numpy (state).float ()
            if gpu_id >= 0:
                with torch.cuda.device (gpu_id):
                    player.state = player.state.cuda ()









        

