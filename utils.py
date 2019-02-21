from __future__ import division
import numpy as np
import torch
import json
import logging
import math as m
from torch.autograd import Variable



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

if __name__ == "__main__":
    r = float (input ())
    print (reward_scaler (r))