import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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

class ConvELU (nn.Module):
    def __init__ (self, in_ch, out_ch, kernel_size=3, bias=True):
        super (ConvELU, self).__init__ ()
        self.conv = nn.Sequential(
            nn.Conv2d (in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size//2, bias=bias),
            nn.ELU ()
        )

    def forward (self, x):
        return self.conv (x)

class ConvInELU (nn.Module):
    def __init__ (self, in_ch, out_ch, kernel_size=3, bias=True):
        super (ConvInELU, self).__init__ ()
        self.module = nn.Sequential (
            nn.Conv2d (in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size//2, bias=bias),
            INELU (out_ch))

    def forward (self, x):
        x = self.module (x)
        return x

class ConvInReLU (nn.Module):
    def __init__ (self, in_ch, out_ch, kernel_size, stride, padding, dilation):
        super (ConvInReLU, self).__init__ ()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation, bias=False),
            nn.InstanceNorm2d (out_ch), 
            nn.ReLU(),
        )

    def forward (self, x):
        return self.conv (x)

class DoubleConv (nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            ConvInELU (in_ch, out_ch),
            ConvInELU (out_ch, out_ch)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class Residual_Conv (nn.Module):
    def __init__ (self, in_ch, out_ch, bias=False):
        super (Residual_Conv, self).__init__ ()
        self.conv1 = ConvInELU (in_ch, out_ch)
        self.conv2 = ConvInELU (out_ch, out_ch//2)
        self.conv3 = ConvInELU (out_ch//2, out_ch)

    def forward (self, x):
        _in = x
        x1 = self.conv1 (_in)
        x2 = self.conv2 (x1)
        _out = self.conv3 (x2)
        return x1 + _out

class UpConv (nn.Module):
    def __init__ (self, in_ch, out_ch):
        super (UpConv, self).__init__ ()
        self.up = nn.Sequential (
            nn.ConvTranspose2d (in_ch, in_ch, 2, stride=2),
            ConvInELU (in_ch, out_ch),
        )

    def forward (self, x):
        x = self.up (x)
        return x

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