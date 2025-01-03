import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Fixed positional encoding with sine and cosine functions."""

    def __init__(self, d_hid=512, n_position=200, dropout=0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Not a parameter
        # Position table of shape (1, n_position, d_hid)
        self.register_buffer(
            "position_table", self._get_sinusoid_encoding_table(n_position, d_hid)
        )

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """Sinusoid position encoding table."""
        denominator = torch.Tensor(
            [1.0 / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
        )
        denominator = denominator.view(1, -1)
        pos_tensor = torch.arange(n_position).unsqueeze(-1).float()
        sinusoid_table = pos_tensor * denominator
        sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])

        return sinusoid_table.unsqueeze(0)

    def forward(self, x):
        """
        Args:
            x (Tensor): Tensor of shape (batch_size, pos_len, d_hid, ...)
        """
        self.device = x.device
        x = x + self.position_table[:, : x.size(1)].clone().detach()
        return self.dropout(x)
