import torch.nn as nn
import torchvision


class model_feature(nn.Module):
    def __init__(self):
        super(model_feature, self).__init__()

        # pretrained_cnn = torchvision.models.video.r3d_18(pretrained=True, progress=True)

        weight = torchvision.models.video.R3D_18_Weights.DEFAULT
        pretrained_cnn = torchvision.models.video.r3d_18(weights=weight, progress=True)

        cnn_layers = list(pretrained_cnn.children())[:3]

        self.rgb_resnet_1 = nn.Sequential(*cnn_layers)

    def forward(self, images):
        images_feature = self.rgb_resnet_1(images)
        return images_feature
