from __future__ import division
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import math
from utils import normal  # , pi
import copy

class Agent (object):
    def __init__ (self, model, env, args, state, rank=0):
        self.args = args

        self.model = model
        self.env = env
        self.state = state
        self.hx = None
        self.cx = None
        
        self.eps_len = 0
        self.done = True
        self.info = None
        self.reward = 0

        self.gpu_id = -1
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.actions = []
        self.actions_explained = []

    def action_lbl_rand (self, lbl, action, t):
        val_list = np.unique (lbl)
        ret = np.zeros_like (action)

        for val in val_list:
            sigle_cell_map = lbl == val
            sigle_cell_area = np.count_nonzero (sigle_cell_map)
            action_tmp = action * sigle_cell_map
            action_1_count = np.count_nonzero (action_tmp)
            ratio = action_1_count / sigle_cell_area
            ratio = np.clip (ratio, 0.1, 0.9)
            sample = self.env.rng.rand ()

            if (sample < ratio):
                ret += sigle_cell_map
        self.action = ret
        ret = torch.from_numpy (ret [::]).long ().unsqueeze(0).unsqueeze(0)
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                ret = ret.cuda()
        return ret

    def action_train (self, use_max=False, use_lbl=False, even_step_reward=True):
        if "Lstm" in self.args.model:
            value, logit, (self.hx, self.cx) = self.model((Variable(
            self.state.unsqueeze(0)), (self.hx, self.cx)))
        else:
            value, logit = self.model (Variable(self.state.unsqueeze(0)))
        prob = F.softmax(logit, dim=1)
        log_prob = F.log_softmax(logit, dim=1)
        entropy = -(log_prob * prob).sum(1)
        self.entropies.append(entropy)
        
        prob_tp = prob.permute (0, 2, 3, 1)
        log_prob_tp = log_prob.permute (0, 2, 3, 1)
        distribution = torch.distributions.Categorical (prob_tp)
        shape = prob_tp.shape
        if not use_max:
            action_tp = distribution.sample ().reshape (1, shape[1], shape[2], 1)
            action = action_tp.permute (0, 3, 1, 2)
            self.action = action.cpu().numpy() [0][0]

            if use_lbl:
                action = self.action_lbl_rand (self.env.gt_lbl, self.action, self.env.step_cnt)

            log_prob = log_prob.gather(1, Variable(action))
            state, self.reward, self.done, self.info = self.env.step(
                self.action)

        if not use_max:
            self.state = torch.from_numpy(state).float()

        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = self.state.cuda()
        # self.reward = max(min(self.reward, 1), -1)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.rewards.append(self.reward [None][None])
        if even_step_reward:
            self.rewards = []
            for i in range (self.env.T):
                self.rewards += [self.env.sum_reward [None][None]/ self.env.T]
        self.eps_len += 1
        return self

    def action_test (self):
        with torch.no_grad():
            if "Lstm" in self.args.model:
                if self.done:
                    if self.gpu_id >= 0:
                        with torch.cuda.device (self.gpu_id):
                            self.cx, self.hx = self.model.lstm.init_hidden (batch_size=1, use_cuda=True)
                    else:
                        self.cx, self.hx = self.model.lstm.init_hidden (batch_size=1, use_cuda=False)
                else:
                    self.cx = Variable (self.cx)
                    self.hx = Variable (self.hx)
                value, logit, (self.hx, self.cx) = self.model((Variable (self.state.unsqueeze(0)), (self.hx, self.cx)))
            else:
                value, logit = self.model(Variable (self.state.unsqueeze(0)))
            
        prob = F.softmax (logit, dim=1)
        action = prob.max (1)[1].data.cpu ().numpy ()
        state, self.reward, self.done, self.info = self.env.step (action [0])
        self.state = torch.from_numpy(state).float()
        if self.gpu_id >= 0:
            with torch.cuda.device (self.gpu_id):
                self.state = self.state.cuda ()
        self.rewards.append (self.reward)
        # print ("action test", self.rewards)
        self.actions.append (action [0])
        self.eps_len += 1
        return self

    def clear_actions (self):
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.actions = []
        self.actions_explained = []
        return self

