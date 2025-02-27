import math
import torch.nn as nn
from lightning_ocr.modules.backbones.resnet_abi import BasicBlock


# https://github.com/open-mmlab/mmocr/blob/main/mmocr/models/textrecog/backbones/resnet31_ocr.py
class ResNet31OCR(nn.Module):
    def __init__(
        self,
        in_channels=3,
        channels=[64, 128, 256, 256, 512, 512, 512],
        layers=[1, 2, 5, 3],  # layers
        last_stage_pool=False,
        stage4_pool_cfg=dict(kernel_size=(2, 1), stride=(2, 1)),
    ):
        super().__init__()

        self.last_stage_pool = last_stage_pool

        # conv 1 (Conv, Conv)
        self.conv1_1 = nn.Conv2d(
            in_channels, channels[0], kernel_size=3, stride=1, padding=1
        )
        self.bn1_1 = nn.BatchNorm2d(channels[0])
        self.relu1_1 = nn.ReLU(inplace=True)

        self.conv1_2 = nn.Conv2d(
            channels[0], channels[1], kernel_size=3, stride=1, padding=1
        )
        self.bn1_2 = nn.BatchNorm2d(channels[1])
        self.relu1_2 = nn.ReLU(inplace=True)

        # conv 2 (Max-pooling, Residual block, Conv)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
        self.block2 = self._make_layer(channels[1], channels[2], layers[0])
        self.conv2 = nn.Conv2d(
            channels[2], channels[2], kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(channels[2])
        self.relu2 = nn.ReLU(inplace=True)

        # conv 3 (Max-pooling, Residual block, Conv)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
        self.block3 = self._make_layer(channels[2], channels[3], layers[1])
        self.conv3 = nn.Conv2d(
            channels[3], channels[3], kernel_size=3, stride=1, padding=1
        )
        self.bn3 = nn.BatchNorm2d(channels[3])
        self.relu3 = nn.ReLU(inplace=True)

        # conv 4 (Max-pooling, Residual block, Conv)
        self.pool4 = nn.MaxPool2d(padding=0, ceil_mode=True, **stage4_pool_cfg)
        self.block4 = self._make_layer(channels[3], channels[4], layers[2])
        self.conv4 = nn.Conv2d(
            channels[4], channels[4], kernel_size=3, stride=1, padding=1
        )
        self.bn4 = nn.BatchNorm2d(channels[4])
        self.relu4 = nn.ReLU(inplace=True)

        # conv 5 ((Max-pooling), Residual block, Conv)
        self.pool5 = None
        if self.last_stage_pool:
            self.pool5 = nn.MaxPool2d(
                kernel_size=2, stride=2, padding=0, ceil_mode=True
            )  # 1/16
        self.block5 = self._make_layer(channels[4], channels[5], layers[3])
        self.conv5 = nn.Conv2d(
            channels[5], channels[5], kernel_size=3, stride=1, padding=1
        )
        self.bn5 = nn.BatchNorm2d(channels[5])
        self.relu5 = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, input_channels, output_channels, blocks):
        layers = []
        for _ in range(blocks):
            downsample = None
            if input_channels != output_channels:
                downsample = nn.Sequential(
                    nn.Conv2d(
                        input_channels,
                        output_channels * BasicBlock.expansion,
                        kernel_size=1,
                        stride=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(output_channels * BasicBlock.expansion),
                )
            layers.append(
                BasicBlock(
                    input_channels * BasicBlock.expansion,
                    output_channels,
                    downsample=downsample,
                )
            )
            input_channels = output_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.relu1_1(x)

        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.relu1_2(x)

        outs = []
        for i in range(4):
            layer_index = i + 2
            pool_layer = getattr(self, f"pool{layer_index}")
            block_layer = getattr(self, f"block{layer_index}")
            conv_layer = getattr(self, f"conv{layer_index}")
            bn_layer = getattr(self, f"bn{layer_index}")
            relu_layer = getattr(self, f"relu{layer_index}")

            if pool_layer is not None:
                x = pool_layer(x)
            x = block_layer(x)
            x = conv_layer(x)
            x = bn_layer(x)
            x = relu_layer(x)

            outs.append(x)

        return x
