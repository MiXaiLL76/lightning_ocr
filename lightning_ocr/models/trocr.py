import torch
import lightning as L
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    VisionEncoderDecoderConfig,
)
import albumentations as A
from lightning_ocr.datasets.recog_text_dataset import (
    RecogTextDataset,
    RecogTextDataModule,
    visualize_dataset,
)
from lightning_ocr.dictionary.fast_tokenizers import FastTokenizer
from lightning_ocr.metrics.recog_metric import WordMetric, OneMinusNEDMetric, CharMetric
from torch.utils.tensorboard import SummaryWriter


# https://huggingface.co/docs/transformers/model_doc/trocr#transformers.TrOCRForCausalLM.forward.example
class TrOCR(L.LightningModule):
    def __init__(self, config: dict = {}):
        super().__init__()
        pretrained_model = config.get(
            "pretrained_model", "microsoft/trocr-small-printed"
        )

        self.processor = TrOCRProcessor.from_pretrained(pretrained_model)
        self.processor.tokenizer = FastTokenizer(**config.get("tokenizer", {}))

        self.cfg = VisionEncoderDecoderConfig.from_pretrained(pretrained_model)

        self.max_token_length = config.get("max_seq_len", self.cfg.decoder.max_length)

        # DECODER CFG
        self.cfg.decoder.vocab_size = self.processor.tokenizer.vocab_size
        self.cfg.decoder.pad_token_id = self.processor.tokenizer.pad_token_id
        self.cfg.decoder.bos_token_id = self.processor.tokenizer.bos_token_id
        self.cfg.decoder.eos_token_id = self.processor.tokenizer.eos_token_id
        self.cfg.decoder.max_length = self.max_token_length

        # ENCODER CFG
        self.cfg.encoder.max_length = self.max_token_length

        # MODEL CFG
        self.cfg.decoder_start_token_id = self.processor.tokenizer.eos_token_id
        self.cfg.eos_token_id = self.processor.tokenizer.eos_token_id
        self.cfg.pad_token_id = self.processor.tokenizer.pad_token_id
        self.cfg.vocab_size = self.processor.tokenizer.vocab_size

        self.model = VisionEncoderDecoderModel.from_pretrained(
            pretrained_model, config=self.cfg, ignore_mismatched_sizes=True
        )

        self.metrics = [
            WordMetric(mode=["exact", "ignore_case", "ignore_case_symbol"]),
            CharMetric(),
            OneMinusNEDMetric(),
        ]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=2e-05,
        )

        scheduler1 = {
            "scheduler": torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.9, total_iters=2
            ),
            "interval": "epoch",
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

    def tokenizer_encode(self, data_samples):
        def tokenize(samples, tokenizer_class):
            tokens = tokenizer_class(
                samples,
                return_tensors="pt",
                pad_to_multiple_of=self.max_token_length,
                padding=True,
                max_length=self.max_token_length,
                truncation=True,
            )
            labels = tokens["input_ids"]
            labels[tokens["attention_mask"] == 0] = tokenizer_class.eos_token_id
            return labels

        samples = [item["gt_text"] for item in data_samples]
        return tokenize(samples, self.processor.tokenizer)

    def training_step(self, batch, batch_idx):
        inputs, data_samples = batch

        pixel_values = self.processor(inputs, return_tensors="pt").pixel_values.to(
            self.model.device
        )
        labels = self.tokenizer_encode(data_samples).to(self.model.device)
        outputs = self.model(pixel_values, labels=labels)
        losses = {
            "total": outputs.loss,
        }

        self.log_dict(
            {f"loss/{key}": val for key, val in losses.items()},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(data_samples),
        )

        lr = self.optimizers().param_groups[0]["lr"]  # Get current learning rate
        self.log("learning_rate", lr, on_step=True, prog_bar=True, logger=True)

        return outputs.loss

    def validation_step(self, batch, batch_idx):
        inputs, data_samples = batch

        pixel_values = self.processor(inputs, return_tensors="pt").pixel_values.to(
            self.model.device
        )
        generated_ids = self.model.generate(
            pixel_values, max_new_tokens=self.max_token_length
        )
        return generated_ids

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        inputs, data_samples = batch

        def batch_decode(input_ids):
            return [
                "".join(self.processor.batch_decode(input_id, skip_special_tokens=True))
                for input_id in input_ids
            ]

        generated_ids = batch_decode(outputs)

        for idx, data_sample in enumerate(data_samples):
            data_sample["pred_text"] = generated_ids[idx]

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


def load_train_pipeline():
    train_pipeline = [
        A.Resize(384, 384),
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


def load_test_pipeline():
    test_pipeline = [
        A.Resize(384, 384),
    ]
    return test_pipeline


if __name__ == "__main__":
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import TensorBoardLogger
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    batch_size = 8

    tb_logger = TensorBoardLogger(save_dir="logs/TrOCR")

    train_dataset = RecogTextDataset(
        data_root="/home/mixaill76/text_datasets/data_collection/005-CV",
        ann_file="ann_file.json",
        pipeline=load_train_pipeline(),
    )

    log_every_n_steps = 50
    if len(train_dataset) // batch_size < 50:
        log_every_n_steps = 5

    checkpoint_callback = ModelCheckpoint(
        dirpath="./checkpoints/TrOCR",
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
        # accumulate_grad_batches=batch_size,
    )

    # https://huggingface.co/microsoft/trocr-small-printed/tree/main
    small_cfg = {
        "tokenizer": {
            "dict_list": list("0123456789."),
        },
        "pretrained_model": "microsoft/trocr-small-printed",
    }
    model = TrOCR(small_cfg)

    from sklearn.model_selection import train_test_split
    import copy

    TRAIN, TEST = train_test_split(
        train_dataset.data_list, test_size=0.2, random_state=42
    )

    test_dataset = copy.deepcopy(train_dataset)
    test_dataset.data_list = TEST
    test_dataset.transform = A.Compose(load_test_pipeline())
    train_dataset.data_list = TRAIN

    trainer.fit(
        model,
        datamodule=RecogTextDataModule(
            train_datasets=[train_dataset],
            eval_datasets=[test_dataset],
            batch_size=batch_size,
        ),
    )
