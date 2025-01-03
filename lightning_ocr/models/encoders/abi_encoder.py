import torch
import torch.nn as nn
import torch.nn.functional as F
from ..modules.transformer_module import PositionalEncoding

class ABIEncoder(nn.Module):
    """Implement transformer encoder for text recognition, modified from
    `<https://github.com/FangShancheng/ABINet>`.

    Args:
        n_layers (int): Number of attention layers. Defaults to 2.
        n_head (int): Number of parallel attention heads. Defaults to 8.
        d_model (int): Dimension :math:`D_m` of the input from previous model.
            Defaults to 512.
        d_inner (int): Hidden dimension of feedforward layers. Defaults to
            2048.
        dropout (float): Dropout rate. Defaults to 0.1.
        max_len (int): Maximum output sequence length :math:`T`. Defaults to
            8 * 32.
    """

    def __init__(self,
                 n_layers: int = 2,
                 n_head: int = 8,
                 d_model: int = 512,
                 d_inner: int = 2048,
                 dropout: float = 0.1,
                 max_len: int = 8 * 32
            ):
        super(ABIEncoder, self).__init__()
        assert d_model % n_head == 0, 'd_model must be divisible by n_head'

        self.pos_encoder = PositionalEncoding(d_model, n_position=max_len)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_head,
            dim_feedforward=d_inner,
            activation=F.relu,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=n_layers,
        )
    
    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        n, c, h, w = feature.shape
        feature = feature.view(n, c, -1).transpose(1, 2)  # (n, h*w, c)
        feature = self.pos_encoder(feature)  # (n, h*w, c)
        feature = feature.transpose(0, 1)  # (h*w, n, c)
        feature = self.transformer(feature)
        feature = feature.permute(1, 2, 0).view(n, c, h, w)
        return feature