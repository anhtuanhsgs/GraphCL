import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv_lstm import ConvLSTMCell
from .conv_gru import ConvGRUCell
import numpy as np

class INELU (nn.Module):
    def __init__ (self, out_ch):
        super (INELU, self).__init__ ()
        self.module = nn.Sequential (
                nn.InstanceNorm2d (out_ch),
                nn.ELU ()
            )

    def forward (self, x):
        x = self.module (x)
        return x

class Residual_Conv (nn.Module):
    def __init__ (self, in_ch, out_ch, bias=False):
        super (Residual_Conv, self).__init__ ()
        self.conv1 = nn.Sequential (
            nn.Conv2d (in_ch, out_ch, kernel_size=3, 
                padding=1, bias=bias),
            INELU (out_ch))
        self.conv2 = nn.Sequential (
            nn.Conv2d (out_ch, out_ch//2, kernel_size=3, 
                padding=1, bias=bias),
            INELU (out_ch))
        self.conv3 = nn.Conv2d (out_ch//2, out_ch, kernel_size=3,
            padding=1, bias=bias)

    def forward (self, x):
        _in = x
        x = self.conv1 (_in)
        x = self.conv2 (x)
        _out = self.conv3 (x)
        return _in + _out

class FusionDown (nn.Module):
    def __init__ (self, in_ch, out_ch, bias=False, kernel_size=3):
        super (FusionDown, self).__init__ ()
        self.conv_in = nn.Sequential (
            nn.Conv2d (in_ch, out_ch, kernel_size=3, stride=2, 
                padding=1, bias=bias),
            INELU (out_ch))
        self.residual = Residual_Conv (out_ch, out_ch, bias=bias)
        self.conv_out = nn.Sequential (
            nn.Conv2d (out_ch, out_ch, kernel_size=kernel_size,
                padding=1, bias=bias),
            INELU (out_ch))

    def forward (self, x):
        x = self.conv_in (x)
        x = self.residual (x)
        x = self.conv_out (x)
        return x

class FusionUp (nn.Module):
    def __init__ (self, in_ch, out_ch, bias=False, kernel_size=3):
        super (FusionUp, self).__init__ ()
        self.conv_in = nn.Sequential (
            nn.Conv2d (in_ch, out_ch, kernel_size=kernel_size, padding=1, bias=bias), 
            INELU (out_ch))
        self.residual = Residual_Conv (out_ch, out_ch, bias=bias)
        self.deconv_out = nn.Sequential (
            nn.ConvTranspose2d (out_ch, out_ch, kernel_size=3, stride=2,
                bias=bias),
            INELU (out_ch))

    def forward (self, x):
        x = self.conv_in (x)
        x = self.residual (x)
        x = self.deconv_out (x)
        N, C, H, W = x.shape
        x = x [:,:,0:H-1, 0:W-1]
        return x

class FusionNet (nn.Module):

    def __init__ (self, in_ch, features, out_ch):
        super (FusionNet, self).__init__ ()
        self.first_layer = nn.Sequential (INELU (in_ch), 
            nn.Conv2d (in_ch, features[0], 3, bias=True, padding=1))
        self.down1 = FusionDown (features[0], features[0])
        self.down2 = FusionDown (features[0], features[1])
        self.down3 = FusionDown (features[1], features[2])
        self.down4 = FusionDown (features[2], features[3])
        self.middle = nn.Dropout (p=0.5)
        self.up4 = FusionUp (features[3], features[2])
        self.up3 = FusionUp (features[2], features[1])
        self.up2 = FusionUp (features[1], features[0])
        self.up1 = FusionUp (features[0], features[0])
        self.actor = nn.Conv2d (features[0], out_ch, 3, padding=1, bias=True)
        self.critic = nn.Conv2d (features[0], 1, 3, padding=1, bias=True)


    def forward (self, x):
        x = self.first_layer (x)
        down1 = self.down1 (x)
        down2 = self.down2 (down1)
        down3 = self.down3 (down2)
        down4 = self.down4 (down3)
        middle = self.middle (down4)
        up4 = self.up4 (middle)
        up3 = self.up3 (up4 + down3)
        up2 = self.up2 (up3 + down2)
        up1 = self.up1 (up2 + down1)
        x = up1 + x
        actor = self.actor (x)
        critic = self.critic (x)
        return critic, actor

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet_down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UNet_down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class UNet_up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(UNet_up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up (x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, bias=True):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, bias=bias, padding=kernel_size//2)

    def forward(self, x):
        x = self.conv(x)
        return x

class UNet (nn.Module):
    def __init__(self, in_ch, features, out_ch):
        super(UNet, self).__init__()
        self.inc = inconv(in_ch, features [0])
        self.down1 = UNet_down(features[0], features[1])
        self.down2 = UNet_down(features[1], features[2])
        self.down3 = UNet_down(features[2], features[3])
        self.down4 = UNet_down(features[3], features[3])
        self.up1 = UNet_up(features[3] * 2, features[2])
        self.up2 = UNet_up(features[2] * 2, features[1])
        self.up3 = UNet_up(features[1] * 2, features[0])
        self.up4 = UNet_up(features[0] * 2, features[0])
        self.actor = outconv(features [0], out_ch)
        self.critic = outconv (features [0], 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)

        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        actor = self.actor (x)
        critic = self.critic (x)
        return critic, actor


def to_numpy (tensor):
    return tensor.cpu ().numpy ().squeeze ()

def debug (tensor):
    tensor_np = to_numpy (tensor)
    shape = tensor_np.shape
    for i in range (shape [0]):
        print (tensor_np [i])

class FusionNetLstm (nn.Module):
    def __init__ (self, in_size, features, out_ch, hidden_channels=256):
        super (FusionNetLstm, self).__init__ ()
        self.first_layer = nn.Sequential (INELU (in_size [0]), 
            nn.Conv2d (in_size [0], features[0], 3, bias=True, padding=1))
        self.down1 = FusionDown (features[0], features[0])
        self.down2 = FusionDown (features[0], features[1])
        self.down3 = FusionDown (features[1], features[2])
        self.down4 = FusionDown (features[2], features[3])
        self.middle = nn.Dropout (p=0.5)
        self.up4 = FusionUp (features[3], features[2])
        self.up3 = FusionUp (features[2], features[1])
        self.up2 = FusionUp (features[1], features[0])
        self.up1 = FusionUp (features[0], features[0])
        self.lstm = ConvLSTMCell (in_size[1:], features [0], hidden_channels, kernel_size=(3, 3), bias=True)
        self.actor = nn.Conv2d (hidden_channels, out_ch, 3, padding=1, bias=True)
        self.critic = nn.Conv2d (hidden_channels, 1, 3, padding=1, bias=True)
            
    def forward (self, inputs):
        x, (hx, cx) = inputs
        x = self.first_layer (x)
        down1 = self.down1 (x)
        down2 = self.down2 (down1)
        down3 = self.down3 (down2)
        down4 = self.down4 (down3)
        middle = self.middle (down4)
        up4 = self.up4 (middle)
        up3 = self.up3 (up4 + down3)
        up2 = self.up2 (up3 + down2)
        up1 = self.up1 (up2 + down1)
        x = up1 + x
        hx, cx = self.lstm (x, (hx, cx))
        x = hx
        return self.critic (x), self.actor (x), (hx, cx)

class UNetLstm (nn.Module):
    def __init__(self, in_size, features, out_ch, hidden_channels):
        super(UNetLstm, self).__init__()
        self.inc = inconv(in_size [0], features [0])
        self.down1 = UNet_down(features[0], features[1])
        self.down2 = UNet_down(features[1], features[2])
        self.down3 = UNet_down(features[2], features[3])
        self.down4 = UNet_down(features[3], features[3])
        self.up1 = UNet_up(features[3] * 2, features[2])
        self.up2 = UNet_up(features[2] * 2, features[1])
        self.up3 = UNet_up(features[1] * 2, features[0])
        self.up4 = UNet_up(features[0] * 2, features[0])
        self.lstm = ConvLSTMCell (in_size[1:], features [0], hidden_channels, kernel_size=(3, 3), bias=True)
        self.actor = outconv(hidden_channels, out_ch)
        self.critic = outconv (hidden_channels, 1)

    def forward(self, inputs):
        x, (hx, cx) = inputs
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)

        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        hx, cx = self.lstm (x, (hx, cx))
        x = hx
        actor = self.actor (x)
        critic = self.critic (x)
        return critic, actor, (hx, cx)


class DilatedModule (nn.Module):
    def __init__ (self, in_ch, out_ch, factors=[1,2,4,8], bias=False):
        super(DilatedModule, self).__init__()
        self.dilconv0 = nn.Sequential (nn.Conv2d (in_ch, out_ch, kernel_size=3, dilation=factors[0], padding=1, bias=bias), nn.InstanceNorm2d (out_ch), nn.ELU ())
        self.dilconv1 = nn.Sequential (nn.Conv2d (out_ch, out_ch, kernel_size=3, dilation=factors[1], padding=2, bias=bias), nn.InstanceNorm2d (out_ch), nn.ELU ())
        self.dilconv2 = nn.Sequential (nn.Conv2d (out_ch, out_ch, kernel_size=3, dilation=factors[2], padding=4, bias=bias), nn.InstanceNorm2d (out_ch), nn.ELU ())
        self.dilconv3 = nn.Sequential (nn.Conv2d (out_ch, out_ch, kernel_size=3, dilation=factors[3], padding=8, bias=bias), nn.InstanceNorm2d (out_ch), nn.ELU ())

    def forward (self, x):
        x = self.dilconv0 (x)
        x = self.dilconv1 (x)
        x = self.dilconv2 (x)
        x = self.dilconv3 (x)
        return x

class DilatedFCN_GRU (nn.Module):
    def __init__ (self, in_size, features, out_ch, hidden_channels):
        super (DilatedFCN_GRU, self).__init__ ()
        self.module0 = DilatedModule (in_size[0], features[0])
        self.module1 = DilatedModule (features[0], features[1])
        self.module2 = DilatedModule (features[1], features[2])
        self.module3 = DilatedModule (features[2], features[3])
        self.dropout3 = nn.Dropout (0.5)
        hidden_in_shape = [features[3]] + [in_size[1], in_size[2]]
        self.gru = ConvGRUCell (hidden_in_shape, hidden_channels, 3)
        self.actor_conv0 = nn.Sequential (nn.Conv2d (hidden_channels, features[3], 
                                kernel_size=3, padding=1, bias=True), nn.InstanceNorm2d (out_ch), nn.ELU ())
        self.critic_conv0 =nn.Sequential (nn.Conv2d (hidden_channels, features[3], 
                                kernel_size=3, padding=1, bias=True), nn.InstanceNorm2d (out_ch), nn.ELU ())
        self.actor = outconv(features[3], out_ch, kernel_size=3)
        self.critic = outconv(features[3], 1, kernel_size=3)

    def forward (self, inputs):
        x, hx = inputs
        x0 = self.module0 (x)
        x1 = self.module1 (x0)
        x2 = self.module2 (x1 + x0)
        x3 = self.module3 (x2)
        x3 = self.dropout3 (x3)
        hx = self.gru (x3 + x2, hx)
        x = hx
        actor = self.actor_conv0 (x)
        actor = self.actor (actor)
        critic = self.critic_conv0 (x)
        critic = self.critic (critic)
        return critic, actor, hx


if __name__ == "__main__":
    # FEATURES = [16, 32, 64, 128]
    # model = UNet (5, features=FEATURES, out_ch=2)
    # x = torch.zeros ((1,5,256,256), dtype=torch.float32)
    # value, logit = model (x)

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
     

    nhidden = 256
    FEATURES = [64, 64, 128, 128]
    model = DilatedFCN_GRU ((5, 256, 256), FEATURES, 2, hidden_channels=nhidden)
    x = torch.zeros ((1,5,256,256), dtype=torch.float32)
    hx = torch.zeros ((1, nhidden, 256, 256), dtype=torch.float32)
    value, logit, hx = model ((x, hx))
    print (value.shape)
    print (logit.shape)
    print (hx.shape)