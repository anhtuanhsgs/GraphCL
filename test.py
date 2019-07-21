from __future__ import division
from setproctitle import setproctitle as ptitle
import torch
from environment import *
from models.models import *
from player_util import Agent
from torch.autograd import Variable
import time
import logging
from Utils.Logger import Logger
from Utils.utils import create_dir
from Utils.metrics import GetDices, DiffFGLabels
from utils import setup_logger, relabel
import numpy as np
import cv2

def inference (args, logger, model, tests, test_env, gpu_id, rng, iter):
    log_img = []
    # idxs = rng.choice (len (tests), 10)
    idxs = []
    idxs.append (rng.randint (len (tests)))
    for i in range (min (len (tests), 33)):
        idxs.append ((idxs [-1] + 1) % len (tests))

    if args.data in ['cvppp']:
        resize = True
    else:
        resize = False

    for i in idxs:
        obs = test_env.set_sample (i, resize)
        done = False
        while (not done):
            with torch.no_grad ():
                with torch.cuda.device (gpu_id):
                    t_obs = torch.tensor (obs[None], dtype=torch.float32, device="cuda")
                    value, logit = model (t_obs)
                    prob = F.softmax (logit, dim=1)
                    action = prob.max (1)[1].data.cpu ().numpy ()

            obs, _, done, _ = test_env.step_inference (action [0])
        img = test_env.render ()
        log_img.append (img [:len(img)//2])

    log_img = np.concatenate (log_img, 0)
    log_info = {"test_samples": log_img}
    for tag, img in log_info.items ():
        img = img [None]
        logger.image_summary (tag, img, iter)

def evaluate (args, logger, env, iter):

    pred_lbl = relabel (env.lbl)
    gt_lbl = env.gt_lbl

    bestDice, FgBgDice = GetDices (pred_lbl, gt_lbl)
    diffFG = DiffFGLabels (pred_lbl, gt_lbl)

    log_info = {
        "bestDice": bestDice,
        "FgBgDice": FgBgDice,
        "diffFG": diffFG
    }

    for tag, value in log_info.items ():
        logger.scalar_summary (tag, value, iter)

    return bestDice, FgBgDice, diffFG 


def test (args, shared_model, env_conf, datasets=None, hasLbl=True, tests=None):
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

    if tests is not None:
        test_env = EM_env (tests, env_conf, type="test")

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
    num_actions = 2
    if args.one_step:
        num_actions = args.one_step
    
    if args.model == "UNet":
        player.model = UNet (env.observation_space.shape [0], args.features, num_actions)
    elif args.model == "FusionNetLstm":
        player.model = FusionNetLstm (env.observation_space.shape, args.features, num_actions, args.hidden_feat)
    elif args.model == "FusionNet":
        player.model = FusionNet (env.observation_space.shape [0], args.features, num_actions)
    elif (args.model == "UNetLstm"):
        player.model = UNetLstm (env.observation_space.shape, args.features, num_actions, args.hidden_feat)
    elif (args.model == "FCN_GRU"):
        player.model = DilatedFCN_GRU (env.observation_space.shape, args.features, num_actions, args.hidden_feat)
    elif (args.model == "UNetGRU"):
        player.model = UNetGRU (env.observation_space.shape, args.features, num_actions, args.hidden_feat)
    elif (args.model == "DilatedUNet"):                 
        player.model = DilatedUNet (env.observation_space.shape [0], args.features, num_actions)
    elif (args.model == "UNetEX"):
        player.model = UNetEX (env.observation_space.shape [0], args.features, num_actions)
    elif (args.model == "UNetFuse"):
        player.model = UNetFuse (env.observation_space.shape [0], args.features, num_actions)
    elif (args.model == "AttUNet"):
        player.model = AttU_Net (env.observation_space.shape [0], args.features, num_actions, split=args.data_channel)
    elif (args.model == "ASPPAttUNet"):
        player.model = ASPPAttU_Net (env_conf ["observation_shape"][0], args.features, num_actions, split=args.data_channel)

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
            if len (recent_episode_scores) > 100:
                recent_episode_scores.pop (0)

            if args.save_max and np.mean (recent_episode_scores) >= max_score:
                max_score = np.mean (recent_episode_scores)
                if gpu_id >= 0:
                    with torch.cuda.device (gpu_id):
                        state_to_save = {}
                        state_to_save = player.model.state_dict ()
                        torch.save (state_to_save, '{0}{1}.dat'.format (args.save_model_dir, 'best_model_' + args.env))

            if num_tests % args.save_period == 0:
                if gpu_id >= 0:
                    with torch.cuda.device (gpu_id):
                        state_to_save = player.model.state_dict ()
                        torch.save (state_to_save, '{0}{1}.dat'.format (args.save_model_dir, args.env + '_' + str (num_tests)))

            if num_tests % args.log_period == 0:
                inference (args, logger, player.model, tests, test_env, gpu_id, player.env.rng, num_tests)
                if (np.max (env.lbl) != 0 and np.max (env.gt_lbl) != 0):
                    bestDice, FgBgDice, diffFG = evaluate (args, logger, player.env, num_tests)
                else:
                    bestDice, FgBgDice, diffFG = 0, 0, 0

                
                if hasLbl:
                    print ("----------------------VALID SET--------------------------")
                    print ("bestDice:", bestDice, "FgBgDice:", FgBgDice, "diffFG:", diffFG)
                    print ("Log test #:", num_tests)
                    print ("rewards: ", player.reward.mean ())
                    print ("sum rewards: ", reward_sum)
                    print ("#gt_values:", len (np.unique (player.env.gt_lbl)))
                    print ("values:")
                    values = player.env.unique ()
                    print (np.concatenate ([values[0][None], values[1][None]], 0))
                    print ("------------------------------------------------")

                log_img = np.concatenate (renderlist, 0)

                if hasLbl:
                    log_info = {"valid_sample": log_img}
                else:
                    log_info = {"test_sample": log_img}

                if "EX" in args.model:
                    cell_probs = []
                    for cell_prob in player.cell_probs:
                        cell_prob = cell_prob.data.cpu ().numpy () [0][0]
                        cell_prob = np.repeat (np.expand_dims (cell_prob, -1), 3, -1) * 255
                        cell_prob = cell_prob.astype (np.uint8) 
                        cell_probs.append (cell_prob)
                    cell_probs = np.concatenate (cell_probs, 1)
                    log_info ["cell_probs"] = cell_probs

                for tag, img in log_info.items ():
                    img = img [None]
                    logger.image_summary (tag, img, num_tests)

                if hasLbl:
                    log_info = {'mean_valid_reward': reward_mean,
                                '100_mean_reward': np.mean (recent_episode_scores)}
                    for tag, value in log_info.items ():
                        logger.scalar_summary (tag, value, num_tests)


            renderlist = []
            reward_sum = 0
            player.eps_len = 0
            
            player.clear_actions ()
            state = player.env.reset (player.model, gpu_id)
            renderlist.append (player.env.render ())

            time.sleep (15)
            player.state = torch.from_numpy (state).float ()
            if gpu_id >= 0:
                with torch.cuda.device (gpu_id):
                    player.state = player.state.cuda ()









        

