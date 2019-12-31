import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .conv_lstm import ConvLSTMCell
from .conv_gru import ConvGRUCell
import numpy as np
import time
from .basic_modules import *
from .modeling.deeplab import DeepLab
from .att_unet import AttU_Net2, AttU_Net, AttU_Net3
from .aspp_att_unet import ASPPAttU_Net2, ASPPAttU_Net

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, bias=True):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, bias=bias, padding=kernel_size//2)

    def forward(self, x):
        x = self.conv(x)
        return x

class ActorCritic (nn.Module):
    def __init__(self, args, last_feat_ch, backbone, out_ch, gpu_id=0):
        super(ActorCritic,self).__init__()
        self.name = backbone.name
        self.backbone=backbone
        if args.lstm_feats:
            self.lstm = ConvLSTMCell (args.size, last_feat_ch, args.lstm_feats, kernel_size=(3, 3), bias=True)
            last_feat_ch = args.lstm_feats
            self.use_lstm = True
        else:
            self.use_lstm = False
        if args.noisy:
            self.actor = nn.Sequential (
                NoisyConv2d (last_feat_ch, out_ch, kernel_size=(3,3), factorised=False, gpu_id=gpu_id),
                # NoisyConv2d (16, out_ch, kernel_size=(1,1), padding=(0,0),factorised=False, gpu_id=gpu_id),
            )
        else:
            self.actor = outconv(last_feat_ch, out_ch, kernel_size=1)

        self.critic = outconv (last_feat_ch, 1, kernel_size=1)

    def forward (self, x):
        if (self.use_lstm):
            x, (hx, cx) = x
        x = self.backbone (x)
        if self.use_lstm:
            hx, cx = self.lstm (x, (hx, cx))
            x = hx
        actor = self.actor (x)
        critic = self.critic (x)
        if self.use_lstm:
            ret = (critic, actor, (hx, cx))
        else:
            ret = (critic, actor)
        return ret

def to_numpy (tensor):
    return tensor.cpu ().numpy ().squeeze ()

def debug (tensor):
    tensor_np = to_numpy (tensor)
    shape = tensor_np.shape
    for i in range (shape [0]):
        print (tensor_np [i])

def get_model (args, name, input_shape, features, num_actions, split, atrous_rates=[6, 12, 18], gpu_id=0):
    if name == "AttUNet":
        model = ActorCritic (args, features [0], AttU_Net (input_shape [0], features, num_actions, split=split), 
                                    out_ch=num_actions, gpu_id=gpu_id)
    if name == "AttUNet2":
        model = ActorCritic (args, features [0], AttU_Net2 (input_shape [0], features, num_actions, split=split), 
                                    out_ch=num_actions, gpu_id=gpu_id)
    if name == "AttUNet3":
        model = ActorCritic (args, features [0], AttU_Net3 (input_shape [0], features, num_actions, split=split), 
                                    out_ch=num_actions, gpu_id=gpu_id)
    if name == "ASPPAttUNet2":
        model = ActorCritic (args, features [0] * (len (atrous_rates) + 1), 
                                ASPPAttU_Net2 (input_shape [0], features, num_actions, split=split, atrous_rates=atrous_rates), 
                                    out_ch=num_actions, gpu_id=gpu_id)
    if name == "ASPPAttUNet":
        model = ActorCritic (args, features [0] * (len (atrous_rates) + 1), 
                                ASPPAttU_Net (input_shape [0], features, num_actions, split=split, atrous_rates=atrous_rates), 
                                    out_ch=num_actions, gpu_id=gpu_id)
    if name == "DeepLab":
        model = ActorCritic (args, features [0], DeepLab(backbone='xception', input_stride=input_shape [0], output_stride=16, split=split, num_classes=num_actions), 
                                    out_ch=num_actions, gpu_id=gpu_id)
    return model


def test_models ():
    FEATURES = [16, 32, 64, 64, 128, 128, 256]
    model = get_model ("ASPPAttUNet2", (5,256,256), features=FEATURES, num_actions=2, split=3)
    x = torch.randn ((1,5,256,256), dtype=torch.float32)
    print (x.shape)    
    value, logit = model (x)
    print (logit.shape, value.shape)
    
#     # curtime = time.time ()
#     # model = AttU_NetAttU_Net (in_ch=5, features=FEATURES, out_ch=2, split=3)
#     # optimizer = optim.Adam (model.parameters (), lr=1e-4, amsgrad=True)
#     # criterion = nn.MSELoss ()
#     # for i in range (20):
#     #     x = torch.zeros ((1,5,256,256), dtype=torch.float32)
#     #     value, logit = model (x)
#     #     gt_value = torch.ones ([1, 1, 256, 256], dtype=torch.float32)
#     #     gt_logit = torch.ones ([1, 2, 256, 256], dtype=torch.float32)

#     #     loss = criterion (value, gt_value) + criterion (logit, gt_logit)
#     #     model.zero_grad ()
#     #     loss.backward ()
#     #     curtime = time.time ()
#     #     optimizer.step ()
#     #     print (value.shape, logit.shape)
#     #     print ("timelast: ", time.time () - curtime)

#     # atrous_rates = [6, 12, 18]
#     # aspp = ASPP (4, 16, atrous_rates)
#     # x = torch.zeros ((1,4,128,128), dtype=torch.float32)
#     # x = aspp (x)
#     # print (x.shape)

#     model = DeepLab(backbone='xception', input_stride=5, output_stride=16, split=3)
#     # model = AttU_Net (in_ch=5, features=FEATURES, out_ch=2, split=3)
#     model.eval ()
#     x = torch.zeros ((1,5,256,256), dtype=torch.float32)    
#     o = model (x)
#     print (x.shape)

    # print (torch.rand ((1, 4, 4)))
    # # prob = prob.transpose (1, -1)
    # print (prob.shape)
    # prob = prob.reshape (-1, 2)
    # print (prob.shape)
    # distribution = torch.distributions.Categorical
    # m = distribution (prob)
    # print (m.sample ().shape)

    # logit = torch.rand ((1, 2, 3, 4))
    # prob = F.softmax (logit, 1)
    # debug (prob)
    # prob_tp = prob.permute (0, 2, 3, 1)
    # distribution = torch.distributions.Categorical (prob_tp)
    # sample = distribution.sample ()
    # print (prob_tp.shape)
    # shape = prob_tp.shape
    # action = sample.reshape (1, shape[1], shape[2], 1).permute (0, 3, 1, 2)
    # debug (sample)
    # action_prob = prob.gather (1, action)
    # debug (action_prob)
    # print (action_prob.shape)
    # print (action_prob[0][0].shape)

    # action_max = prob.max (1)[1]
    # print (action_max.shape)

    # nhidden = 128
    # model = FusionNetLstm ((5, 256, 256), FEATURES, 12, hidden_channels=nhidden)
    # x = torch.zeros ((1,5,256,256), dtype=torch.float32)
    # hx, cx = model.lstm.init_hidden (batch_size=1, use_cuda=False)
    # value, logit, (hx, cx) = model ((x, (hx, cx)))
    # print (value.shape)
    # print (logit.shape)
    # print (hx.shape)
    # print (cx.shape)
     

    # nhidden = 256
    # FEATURES = [64, 64, 128, 128]
    # model = DilatedFCN_GRU ((5, 256, 256), FEATURES, 2, hidden_channels=nhidden)
    # x = torch.zeros ((1,5,256,256), dtype=torch.float32)
    # hx = torch.zeros ((1, nhidden, 256, 256), dtype=torch.float32)
    # value, logit, hx = model ((x, hx))
    # print (value.shape)
    # print (logit.shape)
    # print (hx.shape)

    # nhidden = 256
    # FEATURES = [8, 16, 32, 64]
    # model = UNetGRU ((5, 256, 256), FEATURES, 2, hidden_channels=nhidden)
    # x = torch.zeros ((1,5,256,256), dtype=torch.float32)
    # hx = torch.zeros ((1, nhidden, 256, 256), dtype=torch.float32)
    # value, logit, hx = model ((x, hx))
    # print (value.shape)
    # print (logit.shape)
    # print (hx.shape)