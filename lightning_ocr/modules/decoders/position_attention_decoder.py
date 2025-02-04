from typing import Optional
from lightning_ocr.modules.layers.position_aware_layer import PositionAwareLayer
from lightning_ocr.modules.layers.dot_product_attention_layer import (
    DotProductAttentionLayer,
)

import torch
import torch.nn as nn


class PositionAttentionDecoder(nn.Module):
    """Position attention decoder for RobustScanner.

    RobustScanner: `RobustScanner: Dynamically Enhancing Positional Clues for
    Robust Text Recognition <https://arxiv.org/abs/2007.07542>`_

    Args:
        num_classes (int): Number of chars in tokenizer.
        rnn_layers (int): Number of RNN layers. Defaults to 2.
        dim_input (int): Dimension :math:`D_i` of input vector ``feat``.
            Defaults to 512.
        dim_model (int): Dimension :math:`D_m` of the model. Should also be the
            same as encoder output vector ``out_enc``. Defaults to 128.
        max_seq_len (int): Maximum output sequence length :math:`T`. Defaults
            to 40.
        return_feature (bool): Return feature or logits as the result. Defaults
            to True.
        encode_value (bool): Whether to use the output of encoder ``out_enc``
            as `value` of attention layer. If False, the original feature
            ``feat`` will be used. Defaults to False.
    """

    def __init__(
        self,
        num_classes: int = 37,
        rnn_layers: int = 2,
        dim_input: int = 512,
        dim_model: int = 128,
        max_seq_len: int = 40,
        return_feature: bool = True,
        encode_value: bool = False,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.dim_input = dim_input
        self.dim_model = dim_model
        self.return_feature = return_feature
        self.encode_value = encode_value

        self.embedding = nn.Embedding(self.max_seq_len + 1, self.dim_model)

        self.position_aware_module = PositionAwareLayer(self.dim_model, rnn_layers)

        self.attention_layer = DotProductAttentionLayer()

        self.prediction = None
        if not self.return_feature:
            self.prediction = nn.Linear(
                dim_model if encode_value else dim_input, num_classes
            )
        self.softmax = nn.Softmax(dim=-1)

    def _get_position_index(
        self, length: int, batch_size: int, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Get position index for position attention.

        Args:
            length (int): Length of the sequence.
            batch_size (int): Batch size.
            device (torch.device, optional): Device. Defaults to None.

        Returns:
            torch.Tensor: Position index.
        """
        position_index = torch.arange(0, length, device=device)
        position_index = position_index.repeat([batch_size, 1])
        position_index = position_index.long()
        return position_index

    def forward(self, feat: torch.Tensor, out_enc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat (Tensor): Tensor of shape :math:`(N, D_i, H, W)`.
            out_enc (Tensor): Encoder output of shape
                :math:`(N, D_m, H, W)`.
            data_samples (list[TextRecogDataSample], optional): Batch of
                TextRecogDataSample, containing gt_text information. Defaults
                to None.

        Returns:
            Tensor: A raw logit tensor of shape :math:`(N, T, C)` if
            ``return_feature=False``. Otherwise it will be the hidden feature
            before the prediction projection layer, whose shape is
            :math:`(N, T, D_m)`.
        """

        #
        n, c_enc, h, w = out_enc.size()
        assert c_enc == self.dim_model
        _, c_feat, _, _ = feat.size()
        assert c_feat == self.dim_input
        position_index = self._get_position_index(self.max_seq_len, n, feat.device)

        position_out_enc = self.position_aware_module(out_enc)

        query = self.embedding(position_index)
        query = query.permute(0, 2, 1).contiguous()
        key = position_out_enc.view(n, c_enc, h * w)
        if self.encode_value:
            value = out_enc.view(n, c_enc, h * w)
        else:
            value = feat.view(n, c_feat, h * w)

        attn_out = self.attention_layer(query, key, value, None)
        attn_out = attn_out.permute(0, 2, 1).contiguous()  # [n, max_seq_len, dim_v]

        if self.return_feature:
            return attn_out

        return self.prediction(attn_out)

    #   return self.softmax(self.prediction(attn_out))
