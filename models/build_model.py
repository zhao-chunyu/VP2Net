import torch
import torch.nn as nn
import torchvision
import os

# choose one model
def chooseOneModel(model_name, head_cls):
    '''
    chooseOneModel:
        function: choose a model for train or test
    '''
    
    if model_name == 'videoMAE':
        from transformers import VideoMAEConfig, VideoMAEForVideoClassification
        path = 'videoMAE-small'
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../initmodels', path))
        
        config = VideoMAEConfig.from_json_file(path + '/config.json')
        state_dict = torch.load(path + '/pytorch_model.bin')

        model = VideoMAEForVideoClassification(config)
        model.load_state_dict(state_dict)

        model.classifier = nn.Linear(384, head_cls).cuda()

    elif model_name == 'vivit':
        from transformers import VivitConfig, VivitForVideoClassification
       
        path = 'vivit'
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../initmodels', path))

        config = VivitConfig.from_json_file(os.path.join(path, 'config-my.json'))
        state_dict = torch.load(os.path.join(path, 'pytorch_model.bin'), map_location='cpu')

        model = VivitForVideoClassification(config)
        model.load_state_dict(state_dict)
        model = model.cuda() 

        model.classifier = nn.Linear(768, head_cls).cuda()

        num_patches = (
                (config.video_size[0] // config.tubelet_size[0]) *
                (config.video_size[1] // config.tubelet_size[1]) *
                (config.video_size[2] // config.tubelet_size[2])
        )
        cls_token_num = 1
        new_pos_embed_tensor = torch.zeros(1, num_patches + cls_token_num, config.hidden_size)
        torch.nn.init.trunc_normal_(new_pos_embed_tensor, std=0.02)

        new_pos_embed = nn.Parameter(new_pos_embed_tensor.cuda())

        del model.vivit.embeddings.position_embeddings
        model.vivit.embeddings.register_parameter("position_embeddings", new_pos_embed)

    elif model_name == 'UniformerV1':
        from .uniformer.uniformer import uniformer_small
        path = 'uniformer/uniformer_small_k400_16x8.pth'
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../initmodels', path))

        model = uniformer_small()
        state_dict = torch.load(path)
        model.load_state_dict(state_dict)
        model = model.cuda()
        model.head = nn.Linear(512, head_cls).cuda()

    elif model_name == 'videoFocalNet':
        from .videofocalnet.videoFocalNet import videofocalnet_small
        path = 'videoFocalNet/video-focalnet_small_kinetics400.pth'
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../initmodels', path))

        model = videofocalnet_small()
        state_dict = torch.load(path)['model']
        model.load_state_dict(state_dict)
        model = model.cuda()
        model.head = nn.Linear(768, head_cls).cuda()

    elif model_name == 'timesformer':
        from transformers import TimesformerConfig, TimesformerForVideoClassification
        path = 'timesformer'
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../initmodels', path))

        config = TimesformerConfig.from_json_file(path + '/config.json')
        state_dict = torch.load(path + '/pytorch_model.bin')

        model = TimesformerForVideoClassification(config)
        model.load_state_dict(state_dict)
        model.classifier = nn.Linear(768, head_cls).cuda()

    elif model_name == 'I3D':
        from .I3D import I3D
        model = I3D(head_cls).cuda()

    elif model_name == 'ResNet_2plus1D':
        from .ResNet_2plus1D import ResNet_2plus1D
        model = ResNet_2plus1D(head_cls).cuda()
    
    elif model_name == 'ResNet_MC18':
        from .ResNet_MC18 import ResNet_MC18
        model = ResNet_MC18(head_cls).cuda()
        
    elif model_name == 'SlowOnly':
        from .SlowOnly import SlowOnly
        model = SlowOnly(head_cls).cuda()
    
    elif model_name == 'SlowFast':
        from .SlowFast import SlowFast
        model = SlowFast(head_cls).cuda()
    
    elif model_name == 'TPN':
        from .TPN import TPN
        model = TPN(head_cls).cuda()
    
    elif model_name == 'CSN':
        from .CSN import CSN
        model = CSN(head_cls).cuda()

    elif model_name == 'DERNet':
        from .DERNet.DERNet import DERNet
        model = DERNet(head_cls).cuda()

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


class initOneModel(nn.Module):
    def __init__(self, model_name='videoMAE', head_cls=6):
        super(initOneModel, self).__init__()
        self.model = chooseOneModel(model_name, head_cls)
        self.model_name = model_name
        self.head_cls = head_cls

        self.AdaptiveAvgPool3d = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(p=0.0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, images):
        if self.model_name in ['videoMAE', 'timesformer']:
            images0 = images.permute(0, 2, 1, 3, 4)
            cls_res = self.model(images0).logits

        elif self.model_name in ['vivit']:
            images0 = images.permute(0, 2, 1, 3, 4)
            cls_res = self.model(images0).logits

        elif self.model_name in ['UniformerV1']:
            images0 = images
            cls_res = self.model(images0)

        elif self.model_name in ['videoFocalNet']:
            images0 = images.permute(0, 2, 1, 3, 4)
            cls_res = self.model(images0)
        
        elif self.model_name in ['ResNet_2plus1D', 'ResNet_MC18', 'SlowOnly', 'SlowFast', 'TPN', 'I3D', 'CSN', 'videoSwin', 'DERNet']:
            cls_res = self.model(images)
        
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

        return cls_res
