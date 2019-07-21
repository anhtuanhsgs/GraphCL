import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .conv_lstm import ConvLSTMCell
from .conv_gru import ConvGRUCell
import numpy as np
import time

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


class conv_inr (nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, norm=True):
        super(conv_inr, self).__init__()
        if norm:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size, padding=(kernel_size // 2)),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size, padding=(kernel_size // 2)),
                nn.ReLU(),
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

class UNet_up_res (nn.Module):
    def __init__(self, in_ch, out_ch, bias=False):
        super(UNet_up_res, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2, bias=bias)
        self.conv = double_conv (out_ch, out_ch)

    def forward (self, x1, x2):
        x1 = self.up (x1)
        N, C, H, W = x1.shape
        x1 = x1 [:,:,0:H-1, 0:W-1]
        x = x1 + x2
        x = self.conv (x)
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

class FuseIn (nn.Module):
    def __init__ (self, in_ch, out_ch, split=1):
        super (FuseIn, self).__init__ ()
        self.split = split
        self.local0 = conv_inr (split, out_ch, kernel_size=3)
        self.local1 = conv_inr (out_ch, out_ch, kernel_size=3)
        self.global0 = conv_inr (in_ch-split, out_ch, kernel_size=5, norm=False)
        self.global1 = conv_inr (out_ch, out_ch, kernel_size=1, norm=False)
        self.global2 = conv_inr (out_ch, out_ch, kernel_size=3, norm=True)

    def forward (self, x):
        x_raw = x [:, :self.split, :, :]
        x_lbl = x [:, self.split:, :, :]
        
        x_raw = self.local0 (x_raw)
        x_raw = self.local1 (x_raw)
        
        x_lbl = self.global0 (x_lbl)
        x_lbl = self.global1 (x_lbl)
        x_lbl = self.global2 (x_lbl)
        return torch.cat ([x_raw, x_lbl], dim=1)

class UNetEX (nn.Module):
    def __init__(self, in_ch, features, out_ch):
        super(UNetEX, self).__init__()
        self.fuse_in = FuseIn(in_ch, features [0] // 2)
        self.down1 = UNet_down(features[0], features[1])
        self.down2 = UNet_down(features[1], features[2])
        self.down3 = UNet_down(features[2], features[3])
        self.down4 = UNet_down(features[3], features[3])
        self.up1 = UNet_up(features[3] * 2, features[2])
        self.up2 = UNet_up(features[2] * 2, features[1])
        self.up3 = UNet_up(features[1] * 2, features[0])
        self.up4 = UNet_up(features[0] * 2, features[0])
        self.actor = outconv(features [0], out_ch, kernel_size=3)
        self.critic = outconv (features [0], 1, kernel_size=3)
        self.cell_prob = outconv (features [0], 1, kernel_size=3)

    def forward(self, x):
        x1 = self.fuse_in(x)
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
        cell_prob = F.sigmoid (self.cell_prob (x))
        return critic, actor, cell_prob

class UNetFuse (nn.Module):
    def __init__(self, in_ch, features, out_ch):
        super(UNetFuse, self).__init__()
        self.fuse_in = FuseIn(in_ch, features [0] // 2)
        self.down1 = UNet_down(features[0], features[1])
        self.down2 = UNet_down(features[1], features[2])
        self.down3 = UNet_down(features[2], features[3])
        self.down4 = UNet_down(features[3], features[3])
        self.up1 = UNet_up(features[3] * 2, features[2])
        self.up2 = UNet_up(features[2] * 2, features[1])
        self.up3 = UNet_up(features[1] * 2, features[0])
        self.up4 = UNet_up(features[0] * 2, features[0])
        self.actor = outconv(features [0], out_ch, kernel_size=3)
        self.critic = outconv (features [0], 1, kernel_size=3)

    def forward(self, x):
        x1 = self.fuse_in(x)
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


class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.InstanceNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.InstanceNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.InstanceNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
            nn.InstanceNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.InstanceNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.InstanceNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class AttU_Net(nn.Module):
    def __init__(self,in_ch, features, out_ch, split=1):
        super(AttU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.fuse_in = FuseIn (in_ch, features[0] // 2, split=split)
        self.Conv1 = conv_block(ch_in=features[0],ch_out=features[0])
        self.Conv2 = conv_block(ch_in=features[0],ch_out=features[1])
        self.Conv3 = conv_block(ch_in=features[1],ch_out=features[2])
        self.Conv4 = conv_block(ch_in=features[2],ch_out=features[3])
        self.Conv5 = conv_block(ch_in=features[3],ch_out=features[4])

        self.Up5 = up_conv(ch_in=features[4],ch_out=features[3])
        self.Att5 = Attention_block(F_g=features[3],F_l=features[3],F_int=features[3]//2)
        self.Up_conv5 = conv_block(ch_in=features[4], ch_out=features[3])

        self.Up4 = up_conv(ch_in=features[3],ch_out=features[2])
        self.Att4 = Attention_block(F_g=features[2],F_l=features[2],F_int=features[2]//2)
        self.Up_conv4 = conv_block(ch_in=features[3], ch_out=features[2])
        
        self.Up3 = up_conv(ch_in=features[2],ch_out=features[1])
        self.Att3 = Attention_block(F_g=features[1],F_l=features[1],F_int=features[1]//2)
        self.Up_conv3 = conv_block(ch_in=features[2], ch_out=features[1])
        
        self.Up2 = up_conv(ch_in=features[1],ch_out=features[0])
        self.Att2 = Attention_block(F_g=features[0],F_l=features[0],F_int=features[0]//2)
        self.Up_conv2 = conv_block(ch_in=features[1], ch_out=features[0])

        self.actor = outconv(features [0], out_ch, kernel_size=1)
        self.critic = outconv (features [0], 1, kernel_size=3)


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
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        actor = self.actor (d2)
        critic = self.critic (d2)
        return critic, actor

class ConvInReLU (nn.Module):
    def __init__ (self, in_ch, out_ch, kernel_size, stride, padding, dilation, relu=True, conv11=False):
        super (ConvInReLU, self).__init__ ()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation, bias=False),
            nn.InstanceNorm2d (out_ch), 
            nn.ReLU(),
        )

    def forward (self, x):
        return self.conv (x)

class ASPP (nn.Module):
    def __init__(self, in_ch, out_ch, rates):
        super(ASPP, self).__init__()
        self.stages = nn.Module()
        self.stages.add_module("c0", ConvInReLU(in_ch, out_ch, 1, 1, 0, 1))
        for i, rate in enumerate(rates):
            self.stages.add_module(
                "c{}".format(i + 1),
                ConvInReLU(in_ch, out_ch, 3, 1, padding=rate, dilation=rate),
            )

    def forward(self, x):
        return torch.cat([stage(x) for stage in self.stages.children()], dim=1)

class ASPPAttU_Net (nn.Module):
    def __init__(self,in_ch, features, out_ch, atrous_rates = [6, 12, 18], split=1):
        super(ASPPAttU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.fuse_in = FuseIn (in_ch, features[0] // 2, split=split)
        self.Conv1 = conv_block(ch_in=features[0],ch_out=features[0])
        self.Conv2 = conv_block(ch_in=features[0],ch_out=features[1])
        self.Conv3 = conv_block(ch_in=features[1],ch_out=features[2])
        self.Conv4 = conv_block(ch_in=features[2],ch_out=features[3])
        self.Conv5 = conv_block(ch_in=features[3],ch_out=features[4])

        self.Up5 = up_conv(ch_in=features[4],ch_out=features[3])
        self.Att5 = Attention_block(F_g=features[3],F_l=features[3],F_int=features[3]//2)
        self.Up_conv5 = conv_block(ch_in=features[4], ch_out=features[3])

        self.Up4 = up_conv(ch_in=features[3],ch_out=features[2])
        self.Att4 = Attention_block(F_g=features[2],F_l=features[2],F_int=features[2]//2)
        self.Up_conv4 = conv_block(ch_in=features[3], ch_out=features[2])
        
        self.Up3 = up_conv(ch_in=features[2],ch_out=features[1])
        self.Att3 = Attention_block(F_g=features[1],F_l=features[1],F_int=features[1]//2)
        self.Up_conv3 = conv_block(ch_in=features[2], ch_out=features[1])
        
        self.Up2 = up_conv(ch_in=features[1],ch_out=features[0])
        self.Att2 = Attention_block(F_g=features[0],F_l=features[0],F_int=features[0]//2)
        self.Up_conv2 = conv_block(ch_in=features[1], ch_out=features[0])

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
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

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
    def __init__ (self, in_ch, out_ch, depth, bias=False):
        super(DilatedModule, self).__init__()
        self.layers = nn.ModuleList ()
        self.depth = depth
        for i in range (depth):
            self.layers.append (
                nn.Sequential (
                    nn.Conv2d (out_ch, out_ch, kernel_size=3, dilation=2**i, padding=2**i, bias=bias), 
                    nn.InstanceNorm2d (out_ch), 
                    nn.ELU ()))
        self.last_layer = nn.Sequential (nn.InstanceNorm2d (out_ch), nn.ELU ())

    def forward (self, x):
        layer_rets = []
        for layer in self.layers:
            if len (layer_rets) > 0:
                prevs_sum = torch.sum (torch.cat (layer_rets, 0), 0)
                layer_ret = layer (prevs_sum)
            else:
                layer_ret = layer (x)
            layer_rets.append (layer_ret [None])
        return layer_rets [-1][0]

class DilatedUNet (nn.Module):
    def __init__(self, in_ch, features, out_ch):
        super(DilatedUNet, self).__init__()
        self.inc = inconv(in_ch, features [0])
        self.down1 = UNet_down(features[0], features[1])
        self.down2 = UNet_down(features[1], features[2])
        self.down3 = UNet_down(features[2], features[3])
        self.down4 = UNet_down(features[3], features[3])
        self.dilated_module = DilatedModule (features[3], features[3], depth=5, bias=False)
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
        x = self.dilated_module (x5)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        actor = self.actor (x)
        critic = self.critic (x)
        return critic, actor

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

class UNetGRU (nn.Module):
    def __init__(self, in_size, features, out_ch, hidden_channels):
        super(UNetGRU, self).__init__()
        self.inc = inconv(in_size [0], features [0])
        self.down1 = UNet_down(features[0], features[1])
        self.down2 = UNet_down(features[1], features[2])
        self.down3 = UNet_down(features[2], features[3])
        self.down4 = UNet_down(features[3], features[3])
        self.up1 = UNet_up_res(features[3], features[3])
        self.up2 = UNet_up_res(features[3], features[2])
        self.up3 = UNet_up_res(features[2], features[1])
        self.up4 = UNet_up_res(features[1], features[0])
        hidden_in_shape = [features[0]] + [in_size[1], in_size[2]]
        self.gru = ConvGRUCell (hidden_in_shape, hidden_channels, 3)
        self.actor_conv0 = nn.Sequential (nn.Conv2d (hidden_channels, hidden_channels // 2, 
                                kernel_size=1, padding=1, bias=True), nn.InstanceNorm2d (out_ch), nn.ELU ())
        self.critic_conv0 =nn.Sequential (nn.Conv2d (hidden_channels, hidden_channels // 2, 
                                kernel_size=1, padding=1, bias=True), nn.InstanceNorm2d (out_ch), nn.ELU ())
        self.actor = outconv(hidden_channels // 2, out_ch)
        self.critic = outconv (hidden_channels // 2, 1)

    def forward(self, inputs):
        x, hx = inputs
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        hx = self.gru (x, hx)
        x = hx
        actor = self.actor_conv0 (x)
        actor = self.actor (actor)
        critic = self.critic_conv0 (x)
        critic = self.critic (critic)
        return critic, actor, hx

if __name__ == "__main__":
    FEATURES = [64, 128, 256, 512, 1024]
    # model = UNet (5, features=FEATURES, out_ch=2)
    # x = torch.zeros ((1,5,256,256), dtype=torch.float32)
    # value, logit = model (x)
    
    curtime = time.time ()
    model = ASPPAttU_Net (in_ch=5, features=FEATURES, out_ch=2, split=3)
    optimizer = optim.Adam (model.parameters (), lr=1e-4, amsgrad=True)
    criterion = nn.MSELoss ()
    for i in range (20):
        x = torch.zeros ((1,5,256,256), dtype=torch.float32)
        value, logit = model (x)
        gt_value = torch.ones ([1, 1, 256, 256], dtype=torch.float32)
        gt_logit = torch.ones ([1, 2, 256, 256], dtype=torch.float32)

        loss = criterion (value, gt_value) + criterion (logit, gt_logit)
        model.zero_grad ()
        loss.backward ()
        curtime = time.time ()
        optimizer.step ()
        print (value.shape, logit.shape)
        print ("timelast: ", time.time () - curtime)

    # atrous_rates = [6, 12, 18]
    # aspp = ASPP (4, 16, atrous_rates)
    # x = torch.zeros ((1,4,128,128), dtype=torch.float32)
    # x = aspp (x)
    # print (x.shape)

    model = ASPPAttU_Net (in_ch=5, features=FEATURES, out_ch=2, split=3)
    x = torch.zeros ((1,5,256,256), dtype=torch.float32)    
    o = model (x)
    print (x.shape)

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