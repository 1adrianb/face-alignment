"""RetinaFace network with MobileNet0.25 backbone, FPN, and SSH.

Adapted from https://github.com/biubug6/Pytorch_Retinaface (MIT License).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models._utils as _utils

from .mobilenet import MobileNetV1, conv_bn


def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )


def conv_bn1x1(inp, oup, stride, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )


class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        leaky = 0.1 if out_channel <= 64 else 0
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel // 2, stride=1)
        self.conv5X5_1 = conv_bn(in_channel, out_channel // 4, stride=1, leaky=leaky)
        self.conv5X5_2 = conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)
        self.conv7X7_2 = conv_bn(out_channel // 4, out_channel // 4, stride=1, leaky=leaky)
        self.conv7x7_3 = conv_bn_no_relu(out_channel // 4, out_channel // 4, stride=1)

    def forward(self, x):
        conv3X3 = self.conv3X3(x)
        conv5X5_1 = self.conv5X5_1(x)
        conv5X5 = self.conv5X5_2(conv5X5_1)
        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)
        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        return F.relu(out)


class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        leaky = 0.1 if out_channels <= 64 else 0
        self.output1 = conv_bn1x1(in_channels_list[0], out_channels, stride=1, leaky=leaky)
        self.output2 = conv_bn1x1(in_channels_list[1], out_channels, stride=1, leaky=leaky)
        self.output3 = conv_bn1x1(in_channels_list[2], out_channels, stride=1, leaky=leaky)
        self.merge1 = conv_bn(out_channels, out_channels, leaky=leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky=leaky)

    def forward(self, x):
        x = list(x.values())
        output1 = self.output1(x[0])
        output2 = self.output2(x[1])
        output3 = self.output3(x[2])

        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)
        return [output1, output2, output3]


class ClassHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=2):
        super(ClassHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 2, kernel_size=1)

    def forward(self, x):
        out = self.conv1x1(x)
        return out.permute(0, 2, 3, 1).contiguous().view(out.shape[0], -1, 2)


class BboxHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=2):
        super(BboxHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 4, kernel_size=1)

    def forward(self, x):
        out = self.conv1x1(x)
        return out.permute(0, 2, 3, 1).contiguous().view(out.shape[0], -1, 4)


class LandmarkHead(nn.Module):
    def __init__(self, inchannels=512, num_anchors=2):
        super(LandmarkHead, self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 10, kernel_size=1)

    def forward(self, x):
        out = self.conv1x1(x)
        return out.permute(0, 2, 3, 1).contiguous().view(out.shape[0], -1, 10)


# MobileNet0.25 config for inference
cfg_mnet = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'pretrain': False,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64,
}


class RetinaFace(nn.Module):
    def __init__(self, cfg=None):
        super(RetinaFace, self).__init__()
        if cfg is None:
            cfg = cfg_mnet
        backbone = MobileNetV1()
        self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])

        in_channels_list = [cfg['in_channel'] * 2, cfg['in_channel'] * 4, cfg['in_channel'] * 8]
        out_channels = cfg['out_channel']
        self.fpn = FPN(in_channels_list, out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.ClassHead = nn.ModuleList([ClassHead(out_channels) for _ in range(3)])
        self.BboxHead = nn.ModuleList([BboxHead(out_channels) for _ in range(3)])
        self.LandmarkHead = nn.ModuleList([LandmarkHead(out_channels) for _ in range(3)])

    def forward(self, inputs):
        out = self.body(inputs)
        fpn = self.fpn(out)
        features = [self.ssh1(fpn[0]), self.ssh2(fpn[1]), self.ssh3(fpn[2])]

        bbox_regressions = torch.cat([self.BboxHead[i](f) for i, f in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](f) for i, f in enumerate(features)], dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](f) for i, f in enumerate(features)], dim=1)

        return bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions
