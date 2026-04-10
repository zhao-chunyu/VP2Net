import torch.nn as nn
import torch
import os
# device = torch.device('cuda:2')


def conv3x3(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 3, 1, 1),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_planes, out_planes, 3, 1, 1),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True))


class model_saliency(nn.Module):
    def __init__(self, pretrained=True):
        super(model_saliency, self).__init__()
        self.sali = saliency()

        # if pretrained:
        #     weight_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../initmodels', 'dernet/salnet.pth'))
        #     state_dict = torch.load(weight_path, map_location='cpu')
        #     missing_keys, unexpected_keys = self.sali.load_state_dict(state_dict, strict=False)
        #
        #     if len(missing_keys) == 0 and len(unexpected_keys) == 0:
        #         print("saliency model is successfully loaded!")
        #     else:
        #         print("[warning] saliency model is error!")

    def forward(self, images):
        features = []
        for i in range(16):
            image = images[:, :, i, :, :]
            feature = self.sali(image)
            features.append(feature)
        trans_features = torch.stack(features, dim=2)
        return trans_features


class saliency(nn.Module):
    def __init__(self):
        n, m = 8, 3

        super(saliency, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.maxpool = nn.MaxPool2d(2, 2)

        self.convd1 = conv3x3(1 * m, 1 * n)
        self.convd2 = conv3x3(1 * n, 2 * n)
        self.convd3 = conv3x3(2 * n, 4 * n)
        self.convd4 = conv3x3(4 * n, 4 * n)

        self.convu3 = conv3x3(8 * n, 4 * n)
        self.convu2 = conv3x3(6 * n, 2 * n)
        self.convu1 = conv3x3(3 * n, 1 * n)

        self.convu0 = nn.Conv2d(n, 1, 3, 1, 1)

    def forward(self, x):
        x1 = x
        x1 = self.convd1(x1)

        x2 = self.maxpool(x1)
        x2 = self.convd2(x2)

        x3 = self.maxpool(x2)
        x3 = self.convd3(x3)

        x4 = self.maxpool(x3)
        x4 = self.convd4(x4)

        y3 = self.upsample(x4)
        y3 = torch.cat([x3, y3], 1)
        y3 = self.convu3(y3)

        y2 = self.upsample(y3)
        y2 = torch.cat([x2, y2], 1)
        y2 = self.convu2(y2)

        y1 = self.upsample(y2)
        y1 = torch.cat([x1, y1], 1)
        y1 = self.convu1(y1)

        y1 = self.convu0(y1)
        y1 = self.sigmoid(y1)

        return y1


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = torch.randn((2, 3, 16, 320, 192)).cuda()
    net = model_saliency().to(device)
    out = net(inputs)
    print(out.shape)
