"""VGG
Adapted from https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vgg.py
"""

from typing import List, Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):

    def __init__(
            self,
            cfg: List[Any],
            num_classes: int = 1000,
            in_chans: int = 3,
            output_stride: int = 32,
            act_layer: nn.Module = nn.ReLU,
            conv_layer: nn.Module = nn.Conv2d,
            norm_layer: nn.Module = None,
            drop_rate: float = 0.,
    ) -> None:
        super(VGG, self).__init__()
        assert output_stride == 32
        self.num_classes = num_classes
        self.num_features = 4096
        self.drop_rate = drop_rate
        self.grad_checkpointing = False
        self.use_norm = norm_layer is not None
        self.feature_info = []
        prev_chs = in_chans
        net_stride = 1
        pool_layer = nn.MaxPool2d
        layers: List[nn.Module] = []
        for v in cfg:
            last_idx = len(layers) - 1
            if v == 'M':
                self.feature_info.append(dict(num_chs=prev_chs, reduction=net_stride, module=f'features.{last_idx}'))
                layers += [pool_layer(kernel_size=2, stride=2)]
                net_stride *= 2
            else:
                v = cast(int, v)
                conv2d = conv_layer(prev_chs, v, kernel_size=3, padding=1)
                if norm_layer is not None:
                    layers += [conv2d, norm_layer(v), act_layer(inplace=True)]
                else:
                    layers += [conv2d, act_layer(inplace=True)]
                prev_chs = v
        self.features = nn.Sequential(*layers)
        self.feature_info.append(dict(num_chs=prev_chs, reduction=net_stride, module=f'features.{len(layers) - 1}'))

        self.head = nn.Sequential(
            nn.Flatten(1),
            nn.Dropout(self.drop_rate),
            nn.Linear(self.num_features * 8, self.num_features), # 8 is output feature dimension
            nn.ReLU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(self.num_features, num_classes)
        )

        self._initialize_weights()

    @torch.jit.ignore
    def get_classifier(self):
        return self.head.fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.head(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def create_vgg(variant: str) -> VGG:
    model = VGG(cfgs[variant], num_classes = 10, norm_layer=nn.BatchNorm2d, drop_rate=0.3)
    return model

if __name__ == "__main__":
    print("creating vgg11")
    model = create_vgg('vgg11')
    print("creating vgg13")
    model = create_vgg('vgg13')
    print("creating vgg16")
    model = create_vgg('vgg16')
    print("creating vgg19")
    model = create_vgg('vgg19')