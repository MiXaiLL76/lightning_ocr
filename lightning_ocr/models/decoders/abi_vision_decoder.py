import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from lightning_ocr.models.modules.transformer_module import PositionalEncoding


class ABIVisionDecoder(nn.Module):
    """Converts visual features into text characters.

    Implementation of VisionEncoder in
        `ABINet <https://arxiv.org/abs/2103.06495>`_.

    Args:
        num_classes (int): Number of chars in Dictionary.
        in_channels (int): Number of channels :math:`E` of input vector.
            Defaults to 512.
        num_channels (int): Number of channels of hidden vectors in mini U-Net.
            Defaults to 64.
        attn_height (int): Height :math:`H` of input image features. Defaults
            to 8.
        attn_width (int): Width :math:`W` of input image features. Defaults to
            32.
        attn_mode (str): Upsampling mode for :obj:`torch.nn.Upsample` in mini
            U-Net. Defaults to 'nearest'.
        max_seq_len (int): Maximum sequence length. The
            sequence is usually generated from decoder. Defaults to 40.
    """

    def __init__(
        self,
        num_classes: int = 37,
        in_channels: int = 512,
        num_channels: int = 64,
        attn_height: int = 8,
        attn_width: int = 32,
        attn_mode: str = "nearest",
        max_seq_len: int = 40,
    ) -> None:
        super().__init__()

        self.max_seq_len = max_seq_len

        # For mini-Unet
        self.k_encoder = nn.Sequential(
            self._encoder_layer(in_channels, num_channels, stride=(1, 2)),
            self._encoder_layer(num_channels, num_channels, stride=(2, 2)),
            self._encoder_layer(num_channels, num_channels, stride=(2, 2)),
            self._encoder_layer(num_channels, num_channels, stride=(2, 2)),
        )

        self.k_decoder = nn.Sequential(
            self._decoder_layer(
                num_channels, num_channels, scale_factor=2, mode=attn_mode
            ),
            self._decoder_layer(
                num_channels, num_channels, scale_factor=2, mode=attn_mode
            ),
            self._decoder_layer(
                num_channels, num_channels, scale_factor=2, mode=attn_mode
            ),
            self._decoder_layer(
                num_channels,
                in_channels,
                size=(attn_height, attn_width),
                mode=attn_mode,
            ),
        )

        self.pos_encoder = PositionalEncoding(in_channels, max_seq_len)
        self.project = nn.Linear(in_channels, in_channels)
        self.cls = nn.Linear(in_channels, num_classes)

    def forward(self, out_enc: torch.Tensor) -> Dict:
        # Position Attention
        N, E, H, W = out_enc.size()
        k, v = out_enc, out_enc  # (N, E, H, W)

        # Apply mini U-Net on k
        features = []
        for i in range(len(self.k_encoder)):
            k = self.k_encoder[i](k)
            features.append(k)
        for i in range(len(self.k_decoder) - 1):
            k = self.k_decoder[i](k)
            k = k + features[len(self.k_decoder) - 2 - i]
        k = self.k_decoder[-1](k)

        # q = positional encoding
        zeros = out_enc.new_zeros((N, self.max_seq_len, E))  # (N, T, E)
        q = self.pos_encoder(zeros)  # (N, T, E)
        q = self.project(q)  # (N, T, E)

        # Attention encoding
        attn_scores = torch.bmm(q, k.flatten(2, 3))  # (N, T, (H*W))
        attn_scores = attn_scores / (E**0.5)
        attn_scores = torch.softmax(attn_scores, dim=-1)
        v = v.permute(0, 2, 3, 1).view(N, -1, E)  # (N, (H*W), E)
        attn_vecs = torch.bmm(attn_scores, v)  # (N, T, E)

        out_enc = self.cls(attn_vecs)
        result = {
            "feature": attn_vecs,
            "logits": out_enc,
            "attn_scores": attn_scores.view(N, -1, H, W),
        }
        return result

    def _encoder_layer(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
    ) -> nn.Sequential:
        """Generate encoder layer.

        Args:
            in_channels (int): Input channels.
            out_channels (int): Output channels.
            kernel_size (int, optional): Kernel size. Defaults to 3.
            stride (int, optional): Stride. Defaults to 2.
            padding (int, optional): Padding. Defaults to 1.

        Returns:
            nn.Sequential: Encoder layer.
        """

        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def _decoder_layer(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        mode: str = "nearest",
        scale_factor: Optional[int] = None,
        size: Optional[Tuple[int, int]] = None,
    ):
        """Generate decoder layer.

        Args:
            in_channels (int): Input channels.
            out_channels (int): Output channels.
            kernel_size (int): Kernel size. Defaults to 3.
            stride (int): Stride. Defaults to 1.
            padding (int): Padding. Defaults to 1.
            mode (str): Interpolation mode. Defaults to 'nearest'.
            scale_factor (int, optional): Scale factor for upsampling.
            size (Tuple[int, int], optional): Output size. Defaults to None.
        """
        align_corners = None if mode == "nearest" else True
        return nn.Sequential(
            nn.Upsample(
                size=size,
                scale_factor=scale_factor,
                mode=mode,
                align_corners=align_corners,
            ),
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            ),
        )
