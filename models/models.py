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

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, bias=True):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, bias=bias, padding=kernel_size//2)

    def forward(self, x):
        x = self.conv(x)
        return x

class FuseIn (nn.Module):
    def __init__ (self, in_ch, out_ch, split=1):
        super (FuseIn, self).__init__ ()
        self.split = split
        self.local0 = ConvInELU (split, out_ch, kernel_size=3)
        self.local1 = ConvInELU (out_ch, out_ch, kernel_size=3)
        self.global0 = ConvELU (in_ch-split, out_ch, kernel_size=7)
        self.global1 = ConvELU (out_ch, out_ch, kernel_size=1)
        self.global2 = ConvInELU (out_ch, out_ch, kernel_size=3)

    def forward (self, x):
        x_raw = x [:, :self.split, :, :]
        x_lbl = x [:, self.split:, :, :]
        
        x_raw = self.local0 (x_raw)
        x_raw = self.local1 (x_raw)
        
        x_lbl = self.global0 (x_lbl)
        x_lbl = self.global1 (x_lbl)
        x_lbl = self.global2 (x_lbl)
        return torch.cat ([x_raw, x_lbl], dim=1)

class AttU_Net(nn.Module):
    def __init__(self,in_ch, features, out_ch, split=1):
        super(AttU_Net,self).__init__()
        self.name = "AttU_Net"
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.fuse_in = FuseIn (in_ch, features[0] // 2, split=split)
        self.Conv1 = Residual_Conv (in_ch=features[0],out_ch=features[0])
        self.Conv2 = Residual_Conv (in_ch=features[0],out_ch=features[1])
        self.Conv3 = Residual_Conv (in_ch=features[1],out_ch=features[2])
        self.Conv4 = Residual_Conv (in_ch=features[2],out_ch=features[3])
        self.Conv5 = Residual_Conv (in_ch=features[3],out_ch=features[4])

        self.Up5 = UpConv(in_ch=features[4],out_ch=features[3])
        self.Att5 = Attention_block(F_g=features[3],F_l=features[3],F_int=features[3]//2)
        self.UpConv5 = Residual_Conv (in_ch=features[4], out_ch=features[3])

        self.Up4 = UpConv(in_ch=features[3],out_ch=features[2])
        self.Att4 = Attention_block(F_g=features[2],F_l=features[2],F_int=features[2]//2)
        self.UpConv4 = Residual_Conv (in_ch=features[3], out_ch=features[2])
        
        self.Up3 = UpConv(in_ch=features[2],out_ch=features[1])
        self.Att3 = Attention_block(F_g=features[1],F_l=features[1],F_int=features[1]//2)
        self.UpConv3 = Residual_Conv (in_ch=features[2], out_ch=features[1])
        
        self.Up2 = UpConv(in_ch=features[1],out_ch=features[0])
        self.Att2 = Attention_block(F_g=features[0],F_l=features[0],F_int=features[0]//2)
        self.UpConv2 = Residual_Conv (in_ch=features[1], out_ch=features[0])

        self.actor = outconv(features [0], out_ch, kernel_size=1)
        self.critic = outconv (features [0], 1, kernel_size=1)


    def forward(self,x):
        # encoding path
        x = self.fuse_in (x)
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.UpConv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.UpConv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.UpConv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.UpConv2(d2)

        actor = self.actor (d2)
        critic = self.critic (d2)
        return critic, actor

class ASPPAttU_Net (nn.Module):
    def __init__(self,in_ch, features, out_ch, atrous_rates = [6, 12, 18], split=1):
        super(ASPPAttU_Net,self).__init__()
        self.name = "ASPPAttU_Net"
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.fuse_in = FuseIn (in_ch, features[0] // 2, split=split)
        self.Conv1 = Residual_Conv (in_ch=features[0],out_ch=features[0])
        self.Conv2 = Residual_Conv (in_ch=features[0],out_ch=features[1])
        self.Conv3 = Residual_Conv (in_ch=features[1],out_ch=features[2])
        self.Conv4 = Residual_Conv (in_ch=features[2],out_ch=features[3])
        self.Conv5 = Residual_Conv (in_ch=features[3],out_ch=features[4])

        self.Up5 = UpConv(in_ch=features[4],out_ch=features[3])
        self.Att5 = Attention_block(F_g=features[3],F_l=features[3],F_int=features[3]//2)
        self.UpConv5 = Residual_Conv (in_ch=features[4], out_ch=features[3])

        self.Up4 = UpConv(in_ch=features[3],out_ch=features[2])
        self.Att4 = Attention_block(F_g=features[2],F_l=features[2],F_int=features[2]//2)
        self.UpConv4 = Residual_Conv (in_ch=features[3], out_ch=features[2])
        
        self.Up3 = UpConv(in_ch=features[2],out_ch=features[1])
        self.Att3 = Attention_block(F_g=features[1],F_l=features[1],F_int=features[1]//2)
        self.UpConv3 = Residual_Conv (in_ch=features[2], out_ch=features[1])
        
        self.Up2 = UpConv(in_ch=features[1],out_ch=features[0])
        self.Att2 = Attention_block(F_g=features[0],F_l=features[0],F_int=features[0]//2)
        self.UpConv2 = Residual_Conv (in_ch=features[1], out_ch=features[0])

        self.aspp = ASPP (features[0], features[0], atrous_rates)

        self.actor = outconv(features [0] * (len (atrous_rates) + 1), out_ch, kernel_size=1)
        self.critic = outconv (features [0] * (len (atrous_rates) + 1), 1, kernel_size=1)


    def forward(self,x):
        # encoding path
        x = self.fuse_in (x)
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.UpConv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.UpConv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.UpConv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.UpConv2(d2)

        aspp = self.aspp (d2)

        actor = self.actor (aspp)
        critic = self.critic (aspp)

        return critic, actor

def to_numpy (tensor):
    return tensor.cpu ().numpy ().squeeze ()

def debug (tensor):
    tensor_np = to_numpy (tensor)
    shape = tensor_np.shape
    for i in range (shape [0]):
        print (tensor_np [i])

def get_model (name, input_shape, features, num_actions, split):
    if name == "AttUNet":
        model = AttU_Net (input_shape [0], features, num_actions, split=split)
    if name == "ASPPAttUNet":
        model = ASPPAttU_Net (input_shape [0], features, num_actions, split=split)
    if name == "DeepLab":
        model = DeepLab(backbone='xception', input_stride=input_shape [0], output_stride=16, split=split, num_classes=num_actions)
    return model

# if __name__ == "__main__":
#     FEATURES = [64, 128, 256, 512, 1024]
#     # model = UNet (5, features=FEATURES, out_ch=2)
#     # x = torch.zeros ((1,5,256,256), dtype=torch.float32)
#     # value, logit = model (x)
    
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