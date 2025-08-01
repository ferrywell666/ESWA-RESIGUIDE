import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.layers import subpel_conv3x3
from modules.layers.conv import conv1x1, conv3x3, conv, deconv
from modules.layers.res_blk import *


class LatentResidualPrediction(nn.Module):
    def __init__(self, in_dim, out_dim, act=nn.GELU):
        super().__init__()
        self.lrp_transform = nn.Sequential(
            conv3x3(in_dim, 224),
            act(),
            conv3x3(224, 128),
            act(),
            conv3x3(128, out_dim),
        )

    def forward(self, x):
        x = self.lrp_transform(x)
        x = 0.5 * torch.tanh(x)
        return x
