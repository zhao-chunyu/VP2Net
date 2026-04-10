import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class CAM_3DModule(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_3DModule, self).__init__()
        self.chanel_in = in_dim

        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        b, c, t, h, w = x.size()
        query = x.reshape(b, c, t, -1).permute(0, 2, 1, 3)
        key = x.reshape(b, c, t, -1).permute(0, 2, 3, 1)

        energy = torch.matmul(query, key)
        # print(energy.shape)
        energy_max = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)
        energy_new = energy_max - energy

        attention = self.softmax(energy_new)
        value = x.reshape(b, c, t, -1).permute(0, 2, 1, 3)
        # print(attention.shape)
        # print(value.shape)
        output = torch.matmul(attention, value)
        output = output.view(b, t, c, h, w).permute(0, 2, 1, 3, 4)

        output = self.alpha * output + x
        return output

    
class AttnGFM(nn.Module):
    def __init__(self, rgb_channels):
        super(AttnGFM, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv3d(rgb_channels + rgb_channels, rgb_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                      padding=(1, 1, 1)),
            nn.BatchNorm3d(rgb_channels),
            nn.ReLU(inplace=True)
        )

        self.conv1 = nn.Sequential(
            nn.Conv3d(rgb_channels, rgb_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(rgb_channels),
            nn.Sigmoid()
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(rgb_channels, rgb_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(2, 2, 2),
                      dilation=2),
            nn.BatchNorm3d(rgb_channels),
            nn.Sigmoid()
        )

        self.CAM = CAM_3DModule(rgb_channels)

    def forward(self, f_rgb, f_sal):
        f_rgb_cam = self.CAM(f_rgb)

        f_fusion = torch.cat((f_rgb_cam, f_sal), dim=1)

        x1 = self.conv0(f_fusion)
        x2 = self.conv1(x1)
        x3 = self.conv2(x1)

        x4 = x2 * x3
        x5 = f_rgb + x4

        return x5