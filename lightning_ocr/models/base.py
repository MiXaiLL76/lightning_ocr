import torch
import lightning as L
import typing
import albumentations as A
from lightning_ocr.metrics import WordMetric, OneMinusNEDMetric, CharMetric
from torch.utils.tensorboard import SummaryWriter
from matplotlib.figure import Figure
from lightning_ocr.datasets import RecogTextDataModule
from collections import defaultdict


class BaseOcrModel(L.LightningModule):
    def __init__(
        self,
        config: dict,
        base_pretrained_model: str,
        image_height: int,
        image_width: int,
    ):
        self.pretrained_model = config.get("pretrained_model", base_pretrained_model)
        self.init_from_pretrained_model = config.get("init_from_pretrained_model", True)
        self.max_token_length = config.get("max_seq_len", 32)
        self.base_config = config
        self.image_size = {"height": image_height, "width": image_width}
        self.base_config = config

        self.metrics = [
            WordMetric(mode=["exact", "ignore_case", "ignore_case_symbol"]),
            CharMetric(),
            OneMinusNEDMetric(),
        ]

        super().__init__()

    def predict(self, images):
        raise NotImplementedError

    def dump_config(self, output_folder: str):
        raise NotImplementedError

    @classmethod
    def load_from_folder(cls, folder: str, model_file: typing.Optional[str] = "latest"):
        raise NotImplementedError

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.base_config.get("lr", 1e-04),
        )

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs
                * len(self.trainer.fit_loop._data_source.instance),
                eta_min=1e-7,
            ),
            "interval": "step",
            "frequency": 1,
            "name": "CosineAnnealingLR",
        }

        return [optimizer], [scheduler]

    def log_figure(
        self,
        tag: str,
        figure: typing.Union["Figure", typing.List["Figure"]],
        global_step: int,
    ):
        tensorboard: SummaryWriter = self.logger.experiment
        tensorboard.add_figure(tag, figure, global_step)

    def on_validation_batch_end(self, data_samples, batch, batch_idx):
        if isinstance(data_samples, list):
            if len(data_samples) > 0 and isinstance(data_samples[0], dict):
                splited_data_samples = defaultdict(list)
                for item in data_samples:
                    splited_data_samples[item.get("_dataset", "dataset")].append(item)

                for metric in self.metrics:
                    for dataset_name, sub_data_samples in splited_data_samples.items():
                        metric.process(None, sub_data_samples)
                        eval_res = metric.evaluate(
                            size=len(sub_data_samples), prefix=dataset_name
                        )
                        self.log_dict(
                            eval_res,
                            on_epoch=True,
                            batch_size=len(sub_data_samples),
                        )

                for item in data_samples:
                    prefix = item.get("_dataset", "dataset")
                    fig = RecogTextDataModule.visualize_dataset(item, return_fig=True)
                    if fig is not None:
                        self.log_figure(
                            f"{prefix}/{item['index']}", fig, self.global_step
                        )
                        break

    def load_train_pipeline(self):
        train_pipeline = [
            A.Resize(self.image_size["height"], self.image_size["width"]),
            A.Compose(
                [  # RandomApply
                    A.OneOf(
                        [  # RandomChoice
                            A.Rotate(limit=(-15, 15), p=1.0),
                            A.Affine(
                                scale=(0.2, 2.0),
                                rotate=(-15, 15),
                                translate_percent=(0.3, 0.3),
                                shear=(-15, 15),
                                p=1.0,
                            ),
                            A.Perspective(scale=(0.05, 0.1), fit_output=True, p=1.0),
                            A.RandomRotate90(p=1.0),
                        ],
                        1,
                    )
                ],
                p=0.5,
            ),
            A.Compose(
                [  # RandomApply
                    A.GaussNoise(std_range=(0.1, 0.2), p=0.5),
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
        ]
        return train_pipeline

    def load_test_pipeline(self):
        test_pipeline = [
            A.Resize(self.image_size["height"], self.image_size["width"]),
        ]
        return test_pipeline
