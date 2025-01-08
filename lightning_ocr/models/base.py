import torch
import lightning as L
import typing
import albumentations as A
from lightning_ocr.metrics.recog_metric import WordMetric, OneMinusNEDMetric, CharMetric
from torch.utils.tensorboard import SummaryWriter
from matplotlib.figure import Figure

class BaseOcrModel(L.LightningModule):
    def __init__(self, 
                 config: dict, 
                 base_pretrained_model : str,
                 image_height : int,
                 image_width : int,
    ):
        self.pretrained_model = config.get("pretrained_model", base_pretrained_model)
        self.init_from_pretrained_model = config.get("init_from_pretrained_model", True)
        self.max_token_length = config.get("max_seq_len", 40)
        self.base_config = config
        self.image_size = {'height': image_height, 'width': image_width}
        self.base_config = config

        self.metrics = [
            WordMetric(mode=["exact", "ignore_case", "ignore_case_symbol"]),
            CharMetric(),
            OneMinusNEDMetric(),
        ]
        
        super().__init__()

    def predict(self, images):
        raise NotImplementedError
    
    def dump_config(self, output_folder : str):
        raise NotImplementedError
    
    @classmethod
    def load_from_folder(cls, folder : str, model_file : typing.Optional[str] = "latest"):
        raise NotImplementedError
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.base_config.get("lr", 2e-05),
        )

        scheduler1 = {
            "scheduler": torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.9, total_iters=2 * len(self.trainer.fit_loop._data_source.instance)
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

    def log_figure(self, tag : str, figure: typing.Union["Figure", typing.List["Figure"]], global_step : int):
        tensorboard: SummaryWriter = self.logger.experiment
        tensorboard.add_figure(tag, figure, global_step)
    
    def load_train_pipeline(self):
        train_pipeline = [
            A.Resize(self.image_size["height"], self.image_size["width"]),
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
        ]
        return train_pipeline


    def load_test_pipeline(self):
        test_pipeline = [
            A.Resize(self.image_size["height"], self.image_size["width"]),
        ]
        return test_pipeline

