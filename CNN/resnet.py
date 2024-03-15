"""Resnet
Adapted from https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/resnet.py
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
from torchvision import models

def get_padding(kernel_size: int, stride: int, dilation: int = 1) -> int:
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            cardinality: int = 1,
            base_width: int = 64,
            reduce_first: int = 1,
            dilation: int = 1,
            first_dilation: Optional[int] = None,
            act_layer: Type[nn.Module] = nn.ReLU,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
    ):
        """
        Args:
            inplanes: Input channel dimensionality.
            planes: Used to determine output channel dimensionalities.
            stride: Stride used in convolution layers.
            downsample: Optional downsample layer for residual path.
            cardinality: Number of convolution groups.
            base_width: Base width used to determine output channel dimensionality.
            reduce_first: Reduction factor for first convolution output width of residual blocks.
            dilation: Dilation rate for convolution layers.
            first_dilation: Dilation rate for first convolution layer.
            act_layer: Activation layer.
            norm_layer: Normalization layer.
        """
        super(BasicBlock, self).__init__()

        assert cardinality == 1, 'BasicBlock only supports cardinality of 1'
        assert base_width == 64, 'BasicBlock does not support changing base width'
        first_planes = planes // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation

        self.conv1 = nn.Conv2d(
            inplanes, first_planes, kernel_size=3, stride=stride, padding=first_dilation,
            dilation=first_dilation, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(
            first_planes, outplanes, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = norm_layer(outplanes)

        self.act2 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def zero_init_last(self):
        if getattr(self.bn2, 'weight', None) is not None:
            nn.init.zeros_(self.bn2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act2(x)

        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            cardinality: int = 1,
            base_width: int = 64,
            reduce_first: int = 1,
            dilation: int = 1,
            first_dilation: Optional[int] = None,
            act_layer: Type[nn.Module] = nn.ReLU,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
    ):
        """
        Args:
            inplanes: Input channel dimensionality.
            planes: Used to determine output channel dimensionalities.
            stride: Stride used in convolution layers.
            downsample: Optional downsample layer for residual path.
            cardinality: Number of convolution groups.
            base_width: Base width used to determine output channel dimensionality.
            reduce_first: Reduction factor for first convolution output width of residual blocks.
            dilation: Dilation rate for convolution layers.
            first_dilation: Dilation rate for first convolution layer.
            act_layer: Activation layer.
            norm_layer: Normalization layer.
        """
        super(Bottleneck, self).__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation

        self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(
            first_planes, width, kernel_size=3, stride=stride,
            padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False)
        self.bn2 = norm_layer(width)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)

        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def zero_init_last(self):
        if getattr(self.bn3, 'weight', None) is not None:
            nn.init.zeros_(self.bn3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act3(x)

        return x


def downsample_conv(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        first_dilation: Optional[int] = None,
        norm_layer: Optional[Type[nn.Module]] = None,
) -> nn.Module:
    norm_layer = norm_layer or nn.BatchNorm2d
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    first_dilation = (first_dilation or dilation) if kernel_size > 1 else 1
    p = get_padding(kernel_size, stride, first_dilation)

    return nn.Sequential(*[
        nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=p, dilation=first_dilation, bias=False),
        norm_layer(out_channels)
    ])

def make_blocks(
        block_fn: Union[BasicBlock, Bottleneck],
        channels: List[int],
        block_repeats: List[int],
        inplanes: int,
        reduce_first: int = 1,
        output_stride: int = 32,
        down_kernel_size: int = 1,
        **kwargs,
) -> Tuple[List[Tuple[str, nn.Module]], List[Dict[str, Any]]]:
    stages = []
    feature_info = []
    net_block_idx = 0
    net_stride = 4
    dilation = prev_dilation = 1
    for stage_idx, (planes, num_blocks) in enumerate(zip(channels, block_repeats)):
        stage_name = f'layer{stage_idx + 1}'  # never liked this name, but weight compat requires it
        stride = 1 if stage_idx == 0 else 2
        if net_stride >= output_stride:
            dilation *= stride
            stride = 1
        else:
            net_stride *= stride

        downsample = None
        if stride != 1 or inplanes != planes * block_fn.expansion:
            down_kwargs = dict(
                in_channels=inplanes,
                out_channels=planes * block_fn.expansion,
                kernel_size=down_kernel_size,
                stride=stride,
                dilation=dilation,
                first_dilation=prev_dilation,
                norm_layer=kwargs.get('norm_layer'),
            )
            downsample = downsample_conv(**down_kwargs)

        block_kwargs = dict(reduce_first=reduce_first, dilation=dilation, **kwargs)
        blocks = []
        for block_idx in range(num_blocks):
            downsample = downsample if block_idx == 0 else None
            stride = stride if block_idx == 0 else 1
            blocks.append(block_fn(
                inplanes,
                planes,
                stride,
                downsample,
                first_dilation=prev_dilation,
                **block_kwargs,
            ))
            prev_dilation = dilation
            inplanes = planes * block_fn.expansion
            net_block_idx += 1

        stages.append((stage_name, nn.Sequential(*blocks)))
        feature_info.append(dict(num_chs=inplanes, reduction=net_stride, module=stage_name))

    return stages, feature_info


class ResNet(nn.Module):

    def __init__(
            self,
            block: Union[BasicBlock, Bottleneck],
            layers: List[int],
            num_classes: int = 1000,
            in_chans: int = 3,
            output_stride: int = 32,
            cardinality: int = 1,
            base_width: int = 64,
            block_reduce_first: int = 1,
            down_kernel_size: int = 1,
            act_layer = nn.ReLU,
            norm_layer = nn.BatchNorm2d,
            drop_rate: float = 0.0,
            zero_init_last: bool = True,
            block_args: Optional[Dict[str, Any]] = None,
            pretrained = "",
            freeze_ratio = 0.25
    ):
        """
        Args:
            block (nn.Module): class for the residual block. Options are BasicBlock, Bottleneck.
            layers (List[int]) : number of layers in each block
            num_classes (int): number of classification classes (default 1000)
            in_chans (int): number of input (color) channels. (default 3)
            output_stride (int): output stride of the network, 32, 16, or 8. (default 32)
            cardinality (int): number of convolution groups for 3x3 conv in Bottleneck. (default 1)
            base_width (int): bottleneck channels factor. `planes * base_width / 64 * cardinality` (default 64)
            stem_width (int): number of channels in stem convolutions (default 64)
            stem_type (str): The type of stem (default ''):
                * '', default - a single 7x7 conv with a width of stem_width
                * 'deep' - three 3x3 convolution layers of widths stem_width, stem_width, stem_width * 2
                * 'deep_tiered' - three 3x3 conv layers of widths stem_width//4 * 3, stem_width, stem_width * 2
            block_reduce_first (int): Reduction factor for first convolution output width of residual blocks,
                1 for all archs except senets, where 2 (default 1)
            down_kernel_size (int): kernel size of residual block downsample path,
                1x1 for most, 3x3 for senets (default: 1)
            act_layer (str, nn.Module): activation layer
            norm_layer (str, nn.Module): normalization layer
            drop_rate (float): Dropout probability before classifier, for training (default 0.)
            zero_init_last (bool): zero-init the last weight in residual path (usually last BN affine weight)
            block_args (dict): Extra kwargs to pass through to block module
        """
        super(ResNet, self).__init__()
        self.pretrained = pretrained
        self.drop_rate = drop_rate
        self.num_classes = num_classes
        self.freeze_ratio = freeze_ratio

        if not self.pretrained:
            block_args = block_args or dict()
            assert output_stride in (8, 16, 32)
            self.num_classes = num_classes
            self.drop_rate = drop_rate
            self.grad_checkpointing = False

            inplanes = 64
            self.conv1 = nn.Conv2d(in_chans, inplanes, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = norm_layer(inplanes)
            self.act1 = act_layer(inplace=True)
            self.feature_info = [dict(num_chs=inplanes, reduction=2, module='act1')]
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            # Feature Blocks
            channels = [64, 128, 256, 512]
            stage_modules, stage_feature_info = make_blocks(
                block,
                channels,
                layers,
                inplanes,
                cardinality=cardinality,
                base_width=base_width,
                output_stride=output_stride,
                reduce_first=block_reduce_first,
                down_kernel_size=down_kernel_size,
                act_layer=act_layer,
                norm_layer=norm_layer,
                **block_args,
            )
            for stage in stage_modules:
                self.add_module(*stage)  # layer1, layer2, etc
            self.feature_info.extend(stage_feature_info)

            # Head (Pooling and Classifier)
            self.num_features = 512 * block.expansion
            self.global_pool = nn.AdaptiveAvgPool2d(1)

            self.head = self.get_classifier()
            self.init_weights(zero_init_last=zero_init_last)
        else:
            self.model = torch.hub.load("pytorch/vision", self.pretrained["model"], weights=self.pretrained["weights"])
            self.total_layers = self.get_layers()
            self.num_features = self.model.fc.in_features
            self.model.fc = self.get_classifier()
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
                if "fc" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = True

    def get_classifier(self):
        return nn.Sequential(
            nn.Flatten(1),

            nn.Linear(self.num_features, 4096),
            nn.ReLU(True),
            nn.Dropout(self.drop_rate),               
            nn.Linear(4096, self.num_classes)
        )

    def init_weights(self, zero_init_last: bool = True):
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if zero_init_last:
            for m in self.modules():
                if hasattr(m, 'zero_init_last'):
                    m.zero_init_last()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.pretrained:
            x = self.forward_features(x)
            x = self.global_pool(x)
            x = self.head(x)
        else:
            x = self.model(x)
        return x


def create_resnet(variant, num_classes = 10, pretrained = "") -> ResNet:
    if pretrained:
        model = ResNet(cfgs[variant]['block'], cfgs[variant]['layers'], num_classes = num_classes, 
                       drop_rate=0.3, pretrained=pretrained_cfgs[pretrained])
    else:
        model = ResNet(cfgs[variant]['block'], cfgs[variant]['layers'], num_classes = num_classes, drop_rate=0.3)
    return model


cfgs = {
    'resnet18': dict(block=BasicBlock, layers=[2, 2, 2, 2]),
    'resnet34': dict(block=BasicBlock, layers=[3, 4, 6, 3]),
    'resnet50': dict(block=Bottleneck, layers=[3, 4, 6, 3]),
    'resnet101': dict(block=Bottleneck, layers=[3, 4, 23, 3]),
    'resnet152': dict(block=Bottleneck, layers=[3, 8, 36, 3]),
}

pretrained_cfgs = {
    'resnet18': dict(model="resnet18", weights="ResNet18_Weights.DEFAULT"),
    'resnet34': dict(model="resnet34", weights="ResNet34_Weights.DEFAULT"),
    'resnet50': dict(model="resnet50", weights="ResNet50_Weights.DEFAULT"),
    'resnet101': dict(model="resnet50", weights="ResNet101_Weights.DEFAULT"),
    'resnet152': dict(model="resnet152", weights="ResNet152_Weights.DEFAULT"),
}

if __name__ == "__main__":
    # try out original
    # print("creating resnet-18")
    # model = create_resnet('resnet18')
    # print("creating resnet-34")
    # model = create_resnet('resnet34')
    # print("creating resnet-50")
    # model = create_resnet('resnet50')
    # print("creating resnet-101")
    # model = create_resnet('resnet101')
    # print("creating resnet-152")
    # model = create_resnet('resnet152')

    # try out pretrained
    torch.hub.set_dir("D:\Allen_2023\model_weights")
    # print("creating pretrained resnet-18")
    # model = create_resnet('resnet18', pretrained='resnet18')
    # print("creating pretrained resnet-34")
    # model = create_resnet('resnet34', pretrained='resnet34')
    print("creating pretrained resnet-50")
    model = create_resnet('resnet50', pretrained='resnet50')
    # print("creating pretrained resnet-101")
    # model = create_resnet('resnet101', pretrained='resnet101')
    # print("creating pretrained resnet-152")
    # model = create_resnet('resnet152', pretrained='resnet152')

    # sanity
    # print(model)
    # print(model.total_layers)
    test = torch.zeros((5, 3, 256, 256))
    out = model(test)
    print(out.shape)

    # total parameters
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)
