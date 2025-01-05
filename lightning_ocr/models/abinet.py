import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import lightning as L
from lightning_ocr.models.backbones.resnet_abi import ResNetABI
from lightning_ocr.models.encoders.abi_encoder import ABIEncoder
from lightning_ocr.models.decoders.abi_vision_decoder import ABIVisionDecoder
from lightning_ocr.models.module_losses.abi_module_loss import ABIModuleLoss
from lightning_ocr.dictionary.dictionary import Dictionary
from lightning_ocr.datasets.recog_text_dataset import (
    RecogTextDataset,
    visualize_dataset,
)
from lightning_ocr.models.postprocessors.attn_postprocessor import (
    AttentionPostprocessor,
)
from lightning_ocr.metrics.recog_metric import WordMetric, OneMinusNEDMetric, CharMetric
from torch.utils.tensorboard import SummaryWriter


class ABINetVision(L.LightningModule):
    def __init__(self, config: dict = {}):
        super().__init__()
        self.dictionary = Dictionary(**config.get("dictionary", {}))

        self.backbone = ResNetABI(**config.get("backbone", {}))
        self.encoder = ABIEncoder(**config.get("encoder", {}))
        self.decoder = ABIVisionDecoder(
            num_classes=self.dictionary.num_classes, **config.get("decoder", {})
        )
        self.postprocessor = AttentionPostprocessor(
            dictionary=self.dictionary,
            max_seq_len=self.decoder.max_seq_len,
            **config.get("postprocessor", {}),
        )
        self.loss_fn = ABIModuleLoss(
            dictionary=self.dictionary,
            max_seq_len=self.decoder.max_seq_len,
            **config.get("loss_fn", {}),
        )

        self.metrics = [
            WordMetric(mode=["exact", "ignore_case", "ignore_case_symbol"]),
            CharMetric(),
            OneMinusNEDMetric(),
        ]

        assert (
            self.dictionary.end_idx is not None
        ), "Dictionary must contain an end token! (with_end=True)"

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=1e-4,
        )

        scheduler1 = {
            "scheduler": torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.001, total_iters=2
            ),
            "interval": "step",
            "frequency": 1,
        }
        scheduler2 = {
            "scheduler": torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[16, 18], last_epoch=self.trainer.max_epochs
            ),
            "interval": "epoch",
            "frequency": 1,
        }

        return [optimizer], [scheduler1, scheduler2]

    def forward(self, inputs: torch.Tensor):
        feat = self.backbone(inputs)
        out_enc = self.encoder(feat)
        return dict(out_vis=self.decoder(out_enc))

    def forward_test(self, inputs: torch.Tensor) -> torch.Tensor:
        raw_result = self.forward(inputs)

        if "out_fusers" in raw_result and len(raw_result["out_fusers"]) > 0:
            ret = raw_result["out_fusers"][-1]["logits"]
        elif "out_langs" in raw_result and len(raw_result["out_langs"]) > 0:
            ret = raw_result["out_langs"][-1]["logits"]
        else:
            ret = raw_result["out_vis"]["logits"]
        return torch.nn.functional.softmax(ret, dim=-1)

    def training_step(self, batch, batch_idx):
        inputs, data_samples = batch
        out_enc = self.forward(inputs)
        losses = self.loss_fn(out_enc, data_samples)
        total_loss = torch.sum(torch.stack(list(losses.values())))

        losses["total"] = total_loss

        self.log_dict(
            {f"loss/{key}": val for key, val in losses.items()},
            on_step=True,
            on_epoch=True,
            batch_size=len(data_samples),
        )

        return total_loss

    def validation_step(self, batch, batch_idx):
        inputs, data_samples = batch
        out_enc = self.forward_test(inputs)
        return out_enc

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        inputs, data_samples = batch
        data_samples = self.postprocessor(outputs, data_samples)

        for metric in self.metrics:
            metric.process(None, data_samples)
            eval_res = metric.evaluate(size=len(data_samples))
            self.log_dict(
                eval_res,
                on_epoch=True,
                prog_bar=True,
                batch_size=len(data_samples),
            )

        for data_sample in data_samples:
            fig = visualize_dataset(data_sample, return_fig=True)

            tensorboard: SummaryWriter = self.logger.experiment
            tensorboard.add_figure(
                f"data_samples/{data_sample['index']}", fig, self.global_step
            )
            break


# https://github.com/open-mmlab/mmocr/blob/main/configs/textrecog/abinet/_base_abinet-vision.py#L42
def load_train_pipeline():
    train_pipeline = [
        A.Resize(32, 128),
        A.Compose(
            [  # RandomApply
                A.OneOf(
                    [  # RandomChoice
                        A.Rotate(limit=(-15, 15), always_apply=True),
                        A.Affine(
                            scale=(0.2, 2.0),
                            rotate=(-15, 15),
                            translate_percent=(0.3, 0.3),
                            shear=(-15, 15),
                            always_apply=True,
                        ),
                        A.Perspective(
                            scale=(0.05, 0.1), fit_output=True, always_apply=True
                        ),
                    ],
                    1,
                )
            ],
            p=0.5,
        ),
        A.Compose(
            [  # RandomApply
                A.GaussNoise(var_limit=(20, 20), p=0.5),
                A.MotionBlur(blur_limit=7, p=0.5),
            ],
            p=0.25,
        ),
        A.ColorJitter(
            brightness=(0.5, 1.5),
            contrast=(0.5, 1.5),
            saturation=(0.5, 1.5),
            hue=(-0.1, 0.1),
            p=0.25,
        ),
        A.Normalize(
            mean=[123.675 / 255, 116.28 / 255, 103.53 / 255],
            std=[58.395 / 255, 57.12 / 255, 57.375 / 255],
            max_pixel_value=255.0,
            p=1.0,
        ),
        ToTensorV2(),
    ]
    return train_pipeline


def load_test_pipeline():
    test_pipeline = [
        A.Resize(32, 128),
        A.Normalize(
            mean=[123.675 / 255, 116.28 / 255, 103.53 / 255],
            std=[58.395 / 255, 57.12 / 255, 57.375 / 255],
            max_pixel_value=255.0,
            p=1.0,
        ),
        ToTensorV2(),
    ]
    return test_pipeline


if __name__ == "__main__":
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import TensorBoardLogger
    from torch.utils.data import DataLoader

    batch_size = 8

    tb_logger = TensorBoardLogger(save_dir="logs/")

    train_dataset = RecogTextDataset(
        data_root="/home/mixaill76/text_datasets/data_collection/005-CV",
        ann_file="ann_file.json",
        pipeline=load_train_pipeline(),
    )

    log_every_n_steps = 50
    if len(train_dataset) // batch_size < 50:
        log_every_n_steps = 5

    checkpoint_callback = ModelCheckpoint(
        dirpath="./checkpoints/abinet",
        filename="model-{epoch:02d}-loss-{loss/total_epoch:.2f}",
        monitor="loss/total_epoch",
        save_weights_only=True,
        auto_insert_metric_name=False,
        every_n_epochs=1,
    )

    trainer = L.Trainer(
        precision="16",
        logger=tb_logger,
        log_every_n_steps=log_every_n_steps,
        callbacks=[checkpoint_callback],
        max_epochs=20,
    )
    dictionary = dict(
        dict_list=list("0123456789."),
        with_start=True,
        with_end=True,
        same_start_end=True,
        with_padding=False,
        with_unknown=False,
    )
    model = ABINetVision(dict(dictionary=dictionary, decoder={"max_seq_len": 12}))
    train_dataset.data_list = model.loss_fn.get_targets(train_dataset.data_list)

    dataset, test_dataset = torch.utils.data.random_split(train_dataset, [0.8, 0.2])

    trainer.fit(
        model,
        train_dataloaders=DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=batch_size,
            collate_fn=train_dataset.collate_fn,
        ),
        val_dataloaders=DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=batch_size,
            collate_fn=train_dataset.collate_fn,
        ),
    )
