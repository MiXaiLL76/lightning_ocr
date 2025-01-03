import torch
import torch.nn as nn
import lightning as L


class EncoderDecoderRecognizer(L.LightningModule):
    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        loss_fn: nn.Module,
    ):
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        self.loss_fn = loss_fn

    def forward(self, inputs: torch.Tensor):
        feat = self.backbone(inputs)
        out_enc = self.encoder(feat)
        return self.decoder(feat, out_enc)

    def training_step(self, batch, batch_idx):
        inputs, data_samples = batch

        out_enc = self.forward(inputs)

        losses = self.loss_fn(out_enc, data_samples)
        total_loss = torch.sum(torch.stack(list(losses.values())))

        losses["total_loss"] = total_loss

        self.log_dict(losses, on_step=True)

        return total_loss
