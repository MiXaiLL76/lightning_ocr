import math
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1x1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv3x3 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1x1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv3x3(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetABI(nn.Module):
    def __init__(
        self,
        in_channels=3,
        stem_channels=32,
        base_channels=32,
        arch_settings=[3, 4, 6, 6, 3],
        strides=[2, 1, 2, 1, 1],
    ):
        super(ResNetABI, self).__init__()

        self.base_channels = base_channels

        self.conv1 = nn.Conv2d(
            in_channels, stem_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(stem_channels)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(32, arch_settings[0], stride=strides[0])
        self.layer2 = self._make_layer(64, arch_settings[1], stride=strides[1])
        self.layer3 = self._make_layer(128, arch_settings[2], stride=strides[2])
        self.layer4 = self._make_layer(256, arch_settings[3], stride=strides[3])
        self.layer5 = self._make_layer(512, arch_settings[4], stride=strides[4])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.base_channels != channels * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.base_channels,
                    channels * BasicBlock.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(channels * BasicBlock.expansion),
            )

        layers = []
        layers.append(BasicBlock(self.base_channels, channels, stride, downsample))
        self.base_channels = channels * BasicBlock.expansion
        for i in range(1, blocks):
            layers.append(BasicBlock(self.base_channels, channels))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x
