import torch
import torch.nn as nn


class ChannelReductionEncoder(nn.Module):
    def __init__(self, in_channels: int = 512, out_channels: int = 128) -> None:
        super().__init__()
        self.layer = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.layer(feat)
