import torch.nn as nn
from layers.efficientnet_utils import make_divisible
from layers.efficientnet_layers import DropPath, BatchNormAct2d, create_conv2d

def num_groups(group_size, channels):
    if not group_size:  # 0 or None
        return 1  # normal conv with 1 group
    else:
        # NOTE group_size == 1 -> depthwise conv
        assert channels % group_size == 0
        return channels // group_size


class SqueezeExcite(nn.Module):
    def __init__(
            self, in_chs, rd_ratio=0.25, rd_channels=None, act_layer=nn.ReLU,
            gate_layer=nn.Sigmoid, force_act_layer=None, rd_round_fn=None):
        super(SqueezeExcite, self).__init__()
        if rd_channels is None:
            rd_round_fn = rd_round_fn or round
            rd_channels = rd_round_fn(in_chs * rd_ratio)
        act_layer = force_act_layer or act_layer
        self.conv_reduce = nn.Conv2d(in_chs, rd_channels, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(rd_channels, in_chs, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class DepthwiseSeparableConv(nn.Module):
    def __init__(
            self, in_chs, out_chs, dw_kernel_size=3, stride=1, dilation=1, group_size=1, pad_type='',
            noskip=False, pw_kernel_size=1, pw_act=False, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
            se_layer=None, drop_path_rate=0.):
        super(DepthwiseSeparableConv, self).__init__()
        norm_act_layer = BatchNormAct2d
        groups = num_groups(group_size, in_chs)
        self.has_skip = (stride == 1 and in_chs == out_chs) and not noskip
        self.has_pw_act = pw_act  # activation after point-wise conv

        self.conv_dw = create_conv2d(
            in_chs, in_chs, dw_kernel_size, stride=stride, dilation=dilation, padding=pad_type, groups=groups)
        self.bn1 = norm_act_layer(in_chs, inplace=True, act_layer= act_layer)

        # Squeeze-and-excitation
        self.se = se_layer(in_chs, act_layer=act_layer) if se_layer else nn.Identity()

        self.conv_pw = create_conv2d(in_chs, out_chs, pw_kernel_size, padding=pad_type)
        self.bn2 = norm_act_layer(out_chs, inplace=True, apply_act=self.has_pw_act, act_layer= act_layer)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

    def feature_info(self, location):
        if location == 'expansion':  # after SE, input to PW
            return dict(module='conv_pw', hook_type='forward_pre', num_chs=self.conv_pw.in_channels)
        else:  # location == 'bottleneck', block output
            return dict(module='', num_chs=self.conv_pw.out_channels)

    def forward(self, x):
        shortcut = x
        x = self.conv_dw(x)
        x = self.bn1(x)
        x = self.se(x)
        x = self.conv_pw(x)
        x = self.bn2(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x


class InvertedResidual(nn.Module):
    def __init__(
            self, in_chs, out_chs, dw_kernel_size=3, stride=1, dilation=1, group_size=1, pad_type='',
            noskip=False, exp_ratio=1.0, exp_kernel_size=1, pw_kernel_size=1, act_layer=nn.ReLU,
            norm_layer=nn.BatchNorm2d, se_layer=None, conv_kwargs=None, drop_path_rate=0.):
        super(InvertedResidual, self).__init__()
        norm_act_layer = BatchNormAct2d
        conv_kwargs = conv_kwargs or {}
        mid_chs = make_divisible(in_chs * exp_ratio)
        groups = num_groups(group_size, mid_chs)
        self.has_skip = (in_chs == out_chs and stride == 1) and not noskip

        # Point-wise expansion
        self.conv_pw = create_conv2d(in_chs, mid_chs, exp_kernel_size, padding=pad_type, **conv_kwargs)
        self.bn1 = norm_act_layer(mid_chs, inplace=True, act_layer= act_layer)

        # Depth-wise convolution
        self.conv_dw = create_conv2d(
            mid_chs, mid_chs, dw_kernel_size, stride=stride, dilation=dilation,
            groups=groups, padding=pad_type, **conv_kwargs)
        self.bn2 = norm_act_layer(mid_chs, inplace=True, act_layer= act_layer)

        # Squeeze-and-excitation
        self.se = se_layer(mid_chs, act_layer=act_layer) if se_layer else nn.Identity()

        # Point-wise linear projection
        self.conv_pwl = create_conv2d(mid_chs, out_chs, pw_kernel_size, padding=pad_type, **conv_kwargs)
        self.bn3 = norm_act_layer(out_chs, apply_act=False, act_layer= act_layer)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

    def feature_info(self, location):
        if location == 'expansion':  # after SE, input to PWL
            return dict(module='conv_pwl', hook_type='forward_pre', num_chs=self.conv_pwl.in_channels)
        else:  # location == 'bottleneck', block output
            return dict(module='', num_chs=self.conv_pwl.out_channels)

    def forward(self, x):
        shortcut = x
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.se(x)
        x = self.conv_pwl(x)
        x = self.bn3(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x
