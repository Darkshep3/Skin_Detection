"""Efficientnet
Adapted from https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/efficientnet.py
"""
from functools import partial

import math
import torch
import torch.nn as nn
from torchvision import models

from layers.efficientnet_layers import BatchNormAct2d, create_conv2d
from layers.efficientnet_blocks import SqueezeExcite
from layers.efficientnet_builder import EfficientNetBuilder, decode_arch_def, round_channels

class EfficientNet(nn.Module):
    """ EfficientNet

    A flexible and performant PyTorch implementation of efficient network architectures, including:
      * EfficientNet-V2 Small, Medium, Large, XL & B0-B3
      * EfficientNet B0-B8, L2
      * EfficientNet-EdgeTPU
      * EfficientNet-CondConv
      * MixNet S, M, L, XL
      * MnasNet A1, B1, and small
      * MobileNet-V2
      * FBNet C
      * Single-Path NAS Pixel1
      * TinyNet
    """

    def __init__(
            self,
            block_args,
            num_classes=1000,
            num_features=1280,
            in_chans=3,
            stem_size=32,
            fix_stem=False,
            output_stride=32,
            pad_type='',
            round_chs_fn=round_channels,
            act_layer=None,
            norm_layer=None,
            se_layer=None,
            drop_rate=0.,
            drop_path_rate=0.,
            global_pool='avg',
            pretrained = "",
            freeze_ratio = 0.75
    ):
        super(EfficientNet, self).__init__()
        self.pretrained = pretrained
        self.drop_rate = drop_rate
        self.num_classes = num_classes
        self.num_features = num_features
        self.freeze_ratio = freeze_ratio

        if not self.pretrained:
            act_layer = nn.SiLU
            norm_layer = nn.BatchNorm2d
            norm_act_layer = BatchNormAct2d

            se_layer = SqueezeExcite

            # Stem
            if not fix_stem:
                stem_size = round_chs_fn(stem_size)
            self.conv_stem = create_conv2d(in_chans, stem_size, 3, stride=2, padding=pad_type)
            self.bn1 = norm_act_layer(stem_size, inplace=True, act_layer= act_layer)

            # Middle stages (IR/ER/DS Blocks)
            builder = EfficientNetBuilder(
                output_stride=output_stride,
                pad_type=pad_type,
                round_chs_fn=round_chs_fn,
                act_layer=act_layer,
                norm_layer=norm_layer,
                se_layer=se_layer,
                drop_path_rate=drop_path_rate,
            )
            self.blocks = nn.Sequential(*builder(stem_size, block_args))
            self.feature_info = builder.features
            head_chs = builder.in_chs

            # Head + Pooling
            self.conv_head = create_conv2d(head_chs, self.num_features, 1, padding=pad_type)
            self.bn2 = norm_act_layer(self.num_features, inplace=True, act_layer= act_layer)
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            self.head = self.get_classifier()
            self.init_weights()
        else:
            self.model = torch.hub.load("pytorch/vision", self.pretrained["model"], weights=self.pretrained["weights"])
            self.total_layers = self.get_layers()
            self.model.classifier = self.get_classifier()
            self.freeze_weights()

    def get_layers(self):
        return sum([1 for _, _ in self.model.named_parameters()])

    def freeze_weights(self):
        for i, param in enumerate(self.model.named_parameters()):
            name, param = param
            if self.freeze_ratio > 0.0:
                if i < self.total_layers * self.freeze_ratio:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            elif self.freeze_ratio >= 1.0:
                if "classifier" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = True

    def init_weights(self):
        fix_group_fanout=True
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                if fix_group_fanout:
                    fan_out //= m.groups
                nn.init.normal_(m.weight, 0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                fan_out = m.weight.size(0)  # fan-out
                fan_in = 0
                if 'routing_fn' in n:
                    fan_in = m.weight.size(1)
                init_range = 1.0 / math.sqrt(fan_in + fan_out)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def get_classifier(self):
        return nn.Sequential(
            nn.Flatten(1),

            nn.Linear(self.num_features, 4096),
            nn.ReLU(True),
            nn.Dropout(self.drop_rate),              
            nn.Linear(4096, self.num_classes)
        )

    def forward_features(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.blocks(x)
        x = self.conv_head(x)
        x = self.bn2(x)
        return x

    def forward(self, x):
        if not self.pretrained:
            x = self.forward_features(x)
            x = self.global_pool(x)
            x = self.head(x)
        else:
            x = self.model(x)
        return x

def create_effnet(variant, num_classes = 10, pretrained = "") -> EfficientNet:
    arch_def = [
        ['ds_r1_k3_s1_e1_c16_se0.25'],
        ['ir_r2_k3_s2_e6_c24_se0.25'],
        ['ir_r2_k5_s2_e6_c40_se0.25'],
        ['ir_r3_k3_s2_e6_c80_se0.25'],
        ['ir_r3_k5_s1_e6_c112_se0.25'],
        ['ir_r4_k5_s2_e6_c192_se0.25'],
        ['ir_r1_k3_s1_e6_c320_se0.25'],
    ]
    channel_multiplier, depth_multiplier, resolution, drop_rate = cfgs[variant]
    round_chs_fn = partial(round_channels, multiplier=channel_multiplier, divisor=8)

    if pretrained:
        model = EfficientNet(block_args=decode_arch_def(arch_def, depth_multiplier, group_size=None),
                            num_features=round_chs_fn(1280),
                            stem_size=32,
                            round_chs_fn=round_chs_fn,
                            num_classes = num_classes, 
                            drop_rate=drop_rate,
                            pretrained=pretrained_cfgs[pretrained])
    else:
        model = EfficientNet(block_args=decode_arch_def(arch_def, depth_multiplier, group_size=None),
                            num_features=round_chs_fn(1280),
                            stem_size=32,
                            round_chs_fn=round_chs_fn,
                            num_classes = num_classes, 
                            drop_rate=drop_rate)
    return model

cfgs = {
    # name: (channel_multiplier, depth_multiplier, resolution, dropout_rate)
    'efficientnet-b0': (1.0, 1.0, 224, 0.2),
    'efficientnet-b1': (1.0, 1.1, 240, 0.2),
    'efficientnet-b2': (1.1, 1.2, 260, 0.3),
    'efficientnet-b3': (1.2, 1.4, 300, 0.3),
    'efficientnet-b4': (1.4, 1.8, 380, 0.4),
    'efficientnet-b5': (1.6, 2.2, 456, 0.4),
    'efficientnet-b6': (1.8, 2.6, 528, 0.5),
    'efficientnet-b7': (2.0, 3.1, 600, 0.5)
}

pretrained_cfgs = {
    'efficientnet-b0': dict(model="efficientnet_b0", weights="EfficientNet_B0_Weights.DEFAULT"),
    'efficientnet-b1': dict(model="efficientnet_b1", weights="EfficientNet_B1_Weights.DEFAULT"),
    'efficientnet-b2': dict(model="efficientnet_b2", weights="EfficientNet_B2_Weights.DEFAULT"),
    'efficientnet-b3': dict(model="efficientnet_b3", weights="EfficientNet_B3_Weights.DEFAULT"),
    'efficientnet-b4': dict(model="efficientnet_b4", weights="EfficientNet_B4_Weights.DEFAULT"),
    'efficientnet-b5': dict(model="efficientnet_b5", weights="EfficientNet_B5_Weights.DEFAULT"),
    'efficientnet-b6': dict(model="efficientnet_b6", weights="EfficientNet_B6_Weights.DEFAULT"),
    'efficientnet-b7': dict(model="efficientnet_b7", weights="EfficientNet_B7_Weights.DEFAULT"),
}

if __name__ == "__main__":
    # print("creating efficientnet-b0")
    # model = create_effnet('efficientnet-b0')
    # print("creating efficientnet-b1")
    # model = create_effnet('efficientnet-b1')
    # print("creating efficientnet-b2")
    # model = create_effnet('efficientnet-b2')
    # print("creating efficientnet-b3")
    # model = create_effnet('efficientnet-b3')
    # print("creating efficientnet-b4")
    # model = create_effnet('efficientnet-b4')
    # print("creating efficientnet-b5")
    # model = create_effnet('efficientnet-b5')
    # print("creating efficientnet-b6")
    # model = create_effnet('efficientnet-b6')
    # print("creating efficientnet-b7")
    # model = create_effnet('efficientnet-b7')

    torch.hub.set_dir("D:\Allen_2023\model_weights")

    print("creating pretrained efficientnet-b3")
    model = create_effnet('efficientnet-b3', pretrained='efficientnet-b3')
   

    # sanity
    test = torch.zeros((5, 3, 600, 600))
    out = model(test)
    print(out.shape)

    # total parameters
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)
