"""VGG
Adapted from https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vgg.py
"""

from typing import List, Any, cast

import torch
import torch.nn as nn
from torchvision import models

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
            pretrained = "",
            freeze_ratio = 0.5
    ) -> None:
        super(VGG, self).__init__()
        self.pretrained = pretrained 
        self.drop_rate = drop_rate
        self.num_classes = num_classes
        self.num_features = 4096
        self.freeze_ratio = freeze_ratio

        if not self.pretrained:
            assert output_stride == 32
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

            # technically u need one more linear layer but since it's just classification it should be fine
            # https://towardsdatascience.com/understanding-and-visualizing-resnets-442284831be8
            self.head = self.get_classifier()
            self._initialize_weights()           
        else:
            self.model = torch.hub.load("pytorch/vision", self.pretrained["model"], weights=self.pretrained["weights"])
            self.total_layers = self.get_layers()
            self.model.classifier = self.get_classifier()
            self.freeze_weights()

    def get_layers(self):
        return sum([1 for _, _ in self.model.named_parameters()])

    def get_classifier(self):
        return nn.Sequential(
                nn.Flatten(1),

                nn.Linear(512 * 7 * 7, 4096), # hard coded depending on resolution -> self.num_features (4096)
                nn.ReLU(True),
                nn.Dropout(self.drop_rate),
                nn.Linear(4096, 4096), 
                nn.ReLU(True),
                nn.Dropout(self.drop_rate),                
                nn.Linear(4096, self.num_classes)
            )
    
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.pretrained:
            x = self.features(x)
            x = self.head(x)
        else:
            x = self.model(x)
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

def create_vgg(variant: str, num_classes = 10, pretrained = "") -> VGG:
    if pretrained:
        model = VGG(cfgs[variant], num_classes = num_classes, norm_layer=nn.BatchNorm2d, 
                    drop_rate=0.3, pretrained=pretrained_cfgs[pretrained])
    else:
        model = VGG(cfgs[variant], num_classes = num_classes, norm_layer=nn.BatchNorm2d, drop_rate=0.3)
    return model

pretrained_cfgs = {
    'vgg11': dict(model="vgg11_bn", weights="VGG11_BN_Weights.DEFAULT"),
    'vgg13': dict(model="vgg13_bn", weights="VGG13_BN_Weights.DEFAULT"),
    'vgg16': dict(model="vgg16_bn", weights="VGG16_BN_Weights.DEFAULT"),
    'vgg19': dict(model="vgg19_bn", weights="VGG19_BN_Weights.DEFAULT"),
}

if __name__ == "__main__":
    # print("creating vgg11")
    # model = create_vgg('vgg11')
    # print("creating vgg13")
    # model = create_vgg('vgg13')
    # print("creating vgg16")
    # model = create_vgg('vgg16')
    # print("creating vgg19")
    # model = create_vgg('vgg19')

    torch.hub.set_dir("D:\Allen_2023\model_weights")

    print("creating pretrained vgg-11")
    model = create_vgg('vgg11', pretrained='vgg11')


    # sanity
    test = torch.zeros((5, 3, 256, 256))
    out = model(test)
    print(out.shape)

    # total parameters
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)