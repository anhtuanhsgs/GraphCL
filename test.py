from __future__ import division
from setproctitle import setproctitle as ptitle
import torch
from environment import *
from models.models import *
from player_util import Agent
from torch.autograd import Variable
import time, os
import logging
from Utils.Logger import Logger
from Utils.utils import create_dir
from Utils.metrics import GetDices, DiffFGLabels
from utils import setup_logger, clean_reindex, ScalaTracker
import numpy as np
import cv2
import skimage.io as io

def inference (args, logger, model, tests, test_env, gpu_id, rng, iter):
    log_img = []
    idxs = []
    if (len (tests) <= 40):
        idxs.append (0)
    else:
        idxs.append (rng.randint (len (tests)))

    for i in range (min (len (tests), 33)):
        idxs.append ((idxs [-1] + 1) % len (tests))

    if args.data in ['cvppp']:
        resize = True
    else:
        resize = False

    if args.deploy:
        ret = []

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
        if not args.deploy:
            log_img.append (img [:len(img)//2])
        else:
            ret.append (test_env.lbl)

    if not args.deploy:
        log_img = np.concatenate (log_img, 0)
        log_info = {"test_samples": log_img}
        for tag, img in log_info.items ():
            img = img [None]
            logger.image_summary (tag, img, iter)
    else:
        ret = np.array (ret, dtype=np.int32)
        io.imsave ("deploy/" + "deploy_" + args.data + ".tif", ret)
        print ("Done!")

def evaluate (args, env):

    pred_lbl = clean_reindex (env.lbl)
    gt_lbl = env.gt_lbl

    bestDice, FgBgDice = GetDices (pred_lbl, gt_lbl)
    diffFG = DiffFGLabels (pred_lbl, gt_lbl)

    return bestDice, FgBgDice, diffFG 


def test (args, shared_model, env_conf, datasets=None, tests=None):

    ptitle ('Valid agent')

    gpu_id = args.gpu_ids [-1]
    env_conf ["env_gpu"] = gpu_id

    if not args.deploy:
        log = {}
        logger = Logger (args.log_dir)

        create_dir (args.log_dir + "models/")

        os.system ("cp *.py " + args.log_dir)
        os.system ("cp models/models.py " + args.log_dir + "models/")
        os.system ("cp models/basic_modules.py " + args.log_dir + "models/")

        setup_logger ('{}_log'.format (args.env), r'{0}{1}_log'.format (args.log_dir, args.env))
        log['{}_log'.format(args.env)] = logging.getLogger('{}_log'.format(
            args.env))
        d_args = vars (args)
        env_conf_log = env_conf

    if tests is not None:
        if args.testlbl:
            test_env = EM_env (tests [0], env_conf, type="test", gt_lbl_list=tests[1])
        else:
            test_env = EM_env (tests [0], env_conf, type="test")

    if not args.deploy:
        for k in d_args.keys ():
            log ['{}_log'.format (args.env)].info ('{0}: {1}'.format (k, d_args[k]))
        for k in env_conf_log.keys ():
            log ['{}_log'.format (args.env)].info ('{0}: {1}'.format (k, env_conf_log[k]))

    torch.manual_seed (args.seed)

    if gpu_id >= 0:
        torch.cuda.manual_seed (args.seed)

    raw_list, gt_lbl_list = datasets
    env = EM_env (raw_list, env_conf, type="train", gt_lbl_list=gt_lbl_list)

    reward_sum = 0
    start_time = time.time ()
    num_tests = 0
    reward_total_sum = 0

    player = Agent (None, env, args, None)

    player.gpu_id = gpu_id
    
    player.model = get_model (args.model, env_conf ["observation_shape"], args.features, num_actions=2, split=args.data_channel)

    player.state = player.env.reset ()
    player.state = torch.from_numpy (player.state).float ()
    if gpu_id >= 0:
        with torch.cuda.device (gpu_id):
            player.model = player.model.cuda ()
            player.state = player.state.cuda ()
    player.model.eval ()

    flag = True
    if not args.deploy:
        create_dir (args.save_model_dir)

    recent_episode_scores = ScalaTracker (100)
    recent_FgBgDice = ScalaTracker (100)
    recent_bestDice = ScalaTracker (100)
    recent_diffFG = ScalaTracker (100)
    renderlist = []
    renderlist.append (player.env.render ())
    max_score = 0

    if args.deploy:
        with torch.cuda.device (gpu_id):
            player.model.load_state_dict (shared_model.state_dict ())
        inference (args, None, player.model, tests [0], test_env, gpu_id, player.env.rng, len (tests [0]))
        return

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

            log ['{}_log'.format (args.env)].info (
                "VALID: Time {0}, episode reward {1}, num tests {4}, episode length {2}, reward mean {3:.4f}".
                format (
                    time.strftime ("%Hh %Mm %Ss", time.gmtime (time.time () - start_time)),
                    reward_sum, player.eps_len, reward_mean, num_tests))

            recent_episode_scores .push (reward_sum)

            if args.save_max and recent_episode_scores.mean () >= max_score:
                max_score = recent_episode_scores.mean ()
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
                if tests is not None:
                    inference (args, logger, player.model, tests [0], test_env, gpu_id, player.env.rng, num_tests)
                if (np.max (env.lbl) != 0 and np.max (env.gt_lbl) != 0):
                    bestDice, FgBgDice, diffFG = evaluate (args, player.env)
                    recent_FgBgDice.push (FgBgDice)
                    recent_diffFG.push (abs (diffFG))
                    recent_bestDice.push (bestDice)

                    log_info = {
                        "bestDice": recent_FgBgDice.mean (),
                        "FgBgDice": recent_bestDice.mean (),
                        "diffFG": recent_diffFG.mean ()
                    }

                    for tag, value in log_info.items ():
                        logger.scalar_summary (tag, value, num_tests)
                else:
                    bestDice, FgBgDice, diffFG = 0, 0, 0

                print ("----------------------VALID SET--------------------------")
                print ("bestDice:", bestDice, "FgBgDice:", FgBgDice, "diffFG:", diffFG)
                print ("mean bestDice")
                print ("Log test #:", num_tests)
                print ("rewards: ", player.reward.mean ())
                print ("sum rewards: ", reward_sum)
                print ("#gt_values:", len (np.unique (player.env.gt_lbl)))
                print ("values:")
                values = player.env.unique ()
                print (np.concatenate ([values[0][None], values[1][None]], 0))
                print ("------------------------------------------------")

                # log_img = np.concatenate (renderlist, 0)
                                
                for i in range (3):
                    # if i == 0 and args.seg_scale:
                    #     player.probs.insert (0, player.env.scaler / 1.5)
                    # else:
                    player.probs.insert (0, np.zeros_like (player.probs [0]))
                probslist = [np.repeat (np.expand_dims (prob, -1),3, -1) for prob in player.probs]
                probslist = np.concatenate (probslist, 1)
                probslist = (probslist * 256).astype (np.uint8, copy=False)
                log_img = renderlist [-1]
                log_img = np.concatenate ([log_img, probslist], 0)

                log_info = {"valid_sample": log_img}

                if args.seg_scale:
                    log_info ["scaler"] = player.env.scaler


                for tag, img in log_info.items ():
                    img = img [None]
                    logger.image_summary (tag, img, num_tests)

                log_info = {
                        'mean_valid_reward': reward_mean,
                        '100_mean_reward': recent_episode_scores.mean ()}

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
