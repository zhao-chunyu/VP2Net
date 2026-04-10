import torch
import torch.nn as nn
import torchvision

from .model_feature import model_feature
from .model_saliency import model_saliency
from .model_weight import model_weight

from .model_fusion import AttnGFM as model_fusion


import torch.nn.functional as F



class VP2Net(nn.Module):
    def __init__(self, event_cls):
        super(VP2Net, self).__init__()
        self.cls = event_cls

        pretrained_cnn = torchvision.models.video.r3d_18(pretrained=True, progress=True)
        cnn_layers_2 = list(pretrained_cnn.children())[3:-2]

        self.model_feature = model_feature()
        self.model_saliency = model_saliency()
        self.model_weight = model_weight()

        self.model_fusion = model_fusion(128)

        self.sal_conv0 = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        self.sal_conv1_0 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        self.sal_conv1_1 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        self.sal_conv2_0 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True)
        )
        self.sal_conv2_1 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True)
        )

        self.resnet_2 = nn.Sequential(*cnn_layers_2)
        self.conv = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))
        self.bn = nn.BatchNorm3d(512)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.liner_1 = nn.Linear(512, 51)
        self.liner_2 = nn.Linear(51, self.cls)

        self.dropout = nn.Dropout(p=0.3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, images):

        z1 = self.model_feature(images)

        saliency = self.model_saliency(images)

        cc, sim = self.model_weight(saliency)
        alpha = 0.9

        weight_score = alpha * cc + (1 - alpha) * sim
        trans_weight_score = torch.exp((1 - weight_score) / (1 - torch.min(weight_score)))
        trans_weight_score = trans_weight_score.permute((1, 0))
        trans_weight_scores1 = trans_weight_score.unsqueeze(1)
        trans_weight_scores2 = trans_weight_scores1.unsqueeze(3)
        trans_weight_scores3 = trans_weight_scores2.unsqueeze(4)

        saliency2 = saliency * trans_weight_scores3

        x1 = self.sal_conv0(saliency2)
        x2 = self.sal_conv1_0(x1)
        x3 = self.sal_conv1_1(x2)
        x4 = self.sal_conv2_0(x3)
        x5 = self.sal_conv2_1(x4)

        y2 = self.model_fusion(z1, x5)

        y3 = self.resnet_2(y2)
        y4 = self.conv(y3)
        y5 = self.relu(self.bn(y4))
        y6 = self.pool(y5)
        y7 = y6.view((-1, 512))
        y8 = self.relu(self.liner_1(y7))

        y9 = self.dropout(y8)
        res = self.liner_2(y9)
        return res, saliency




