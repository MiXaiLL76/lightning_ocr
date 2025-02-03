import os
import torch
import json
import lightning as L
from lightning_ocr.datasets import HuggingFaceOCRDataset, RecogTextDataModule
import typing
from transformers import MgpstrProcessor, MgpstrForSceneTextRecognition, MgpstrConfig
from lightning_ocr.tokenizer import MgpstrTokenizer
from lightning_ocr.models.base import BaseOcrModel


# https://huggingface.co/docs/transformers/v4.33.2/en/model_doc/mgp-str
class MGP_STR(BaseOcrModel):
    def __init__(self, config: dict = {}):
        super().__init__(
            config=config,
            base_pretrained_model="alibaba-damo/mgp-str-base",
            image_height=32,
            image_width=128,
        )

        self.processor = MgpstrProcessor.from_pretrained(self.pretrained_model)

        if config.get("tokenizer") is not None:
            self.processor.char_tokenizer = MgpstrTokenizer(
                **config.get("tokenizer", {})
            )

        self.cfg = MgpstrConfig.from_pretrained(self.pretrained_model)
        self.cfg.num_character_labels = len(self.processor.char_tokenizer.vocab)
        self.cfg.max_token_length = self.max_token_length

        # https://github.com/huggingface/transformers/blob/v4.33.2/src/transformers/models/mgp_str/processing_mgp_str.py#L143
        self.processor.bpe_tokenizer.pad_token_id = (
            self.processor.bpe_tokenizer.eos_token_id
        )
        self.processor.wp_tokenizer.eos_token_id = 102

        if self.init_from_pretrained_model:
            self.model = MgpstrForSceneTextRecognition.from_pretrained(
                self.pretrained_model, config=self.cfg, ignore_mismatched_sizes=True
            )
        else:
            self.model = MgpstrForSceneTextRecognition(self.cfg)

    def predict(self, images):
        pixel_values = self.processor(
            images=images, return_tensors="pt"
        ).pixel_values.to(self.model.device)
        outputs = self.model(pixel_values)
        return self.processor.batch_decode(outputs.logits)["generated_text"]

    def dump_config(self, output_folder):
        os.makedirs(output_folder, exist_ok=True)

        with open(f"{output_folder}/base_config.json", "w") as f:
            json.dump(self.base_config, f, indent=4)

        self.processor.save_pretrained(output_folder)
        self.model.save_pretrained(output_folder, state_dict={})

        self.processor.bpe_tokenizer.save_pretrained(f"{output_folder}/bpe")
        self.processor.char_tokenizer.save_pretrained(f"{output_folder}/char")

        self.processor.wp_tokenizer.save_pretrained(f"{output_folder}/wp")
        with open(f"{output_folder}/wp/vocab.json", "w") as fd:
            json.dump(self.processor.wp_tokenizer.vocab, fd)

        if os.path.exists(f"{output_folder}/model.safetensors"):
            os.remove(f"{output_folder}/model.safetensors")

    @classmethod
    def load_from_folder(cls, folder, model_file: typing.Optional[str] = "latest"):
        with open(f"{folder}/base_config.json", "r") as f:
            config = json.load(f)

        config["pretrained_model"] = folder
        config["init_from_pretrained_model"] = False

        model = cls(config)

        if model_file == "latest":
            model_file = sorted(
                [file for file in os.listdir(f"{folder}/") if ".ckpt" in file]
            )[-1]

        model.load_state_dict(
            torch.load(
                f"{folder}/{model_file}",
                map_location=torch.device("cpu"),
                weights_only=True,
            )["state_dict"]
        )
        return model

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

    def tokenizer_encode(self, data_samples):
        def tokenize(samples, tokenizer_class):
            if tokenizer_class.bos_token is not None:
                samples = [f"{tokenizer_class.bos_token}{text}" for text in samples]

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
        char_labels = tokenize(samples, self.processor.char_tokenizer)
        bpe_labels = tokenize(samples, self.processor.bpe_tokenizer)
        wp_labels = tokenize(samples, self.processor.wp_tokenizer)
        return char_labels, bpe_labels, wp_labels

    def calc_loss(self, pred_logits, target_labels):
        return torch.nn.functional.cross_entropy(
            pred_logits.view(-1, pred_logits.size(-1)), target_labels.view(-1)
        )

    def training_step(self, batch, batch_idx):
        inputs, data_samples = batch

        pixel_values = self.processor(
            images=inputs, return_tensors="pt"
        ).pixel_values.to(self.model.device)
        outputs = self.model(pixel_values)

        char_labels, bpe_labels, wp_labels = self.tokenizer_encode(data_samples)
        char_loss = self.calc_loss(outputs.logits[0], char_labels.to(self.model.device))
        bpe_loss = self.calc_loss(outputs.logits[1], bpe_labels.to(self.model.device))
        wp_loss = self.calc_loss(outputs.logits[2], wp_labels.to(self.model.device))
        loss = char_loss + bpe_loss + wp_loss

        losses = {
            "char_loss": char_loss,
            "bpe_loss": bpe_loss,
            "wp_loss": wp_loss,
            "total": loss,
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

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, data_samples = batch

        pixel_values = self.processor(
            images=inputs, return_tensors="pt"
        ).pixel_values.to(self.model.device)
        outputs = self.model(pixel_values)

        generated_text = self.processor.batch_decode(outputs.logits)["generated_text"]

        for idx, data_sample in enumerate(data_samples):
            data_sample["pred_text"] = generated_text[idx]

        return data_samples

    def forward(self, inputs: torch.Tensor):
        outputs = self.model(inputs)
        return outputs.logits

    def to_onnx(self, file_path: str, **kwargs: typing.Any) -> None:
        input_sample = torch.randn(
            2, 3, self.image_size["height"], self.image_size["width"]
        )
        # logits, hidden_states, attentions, a3_attentions
        torch.onnx.export(
            self,
            input_sample,
            file_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
            **kwargs,
        )


if __name__ == "__main__":
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import TensorBoardLogger
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    batch_size = 8

    config = {
        "tokenizer": {
            "dict_list": list("0123456789.-;:"),
        },
    }

    model = MGP_STR(config)

    tb_logger = TensorBoardLogger(save_dir="logs/MGP_STR")

    train_datasets = [
        HuggingFaceOCRDataset(
            "MiXaiLL76/7SEG_OCR", split="train", pipeline=model.load_train_pipeline()
        ),
        HuggingFaceOCRDataset(
            "MiXaiLL76/SVHN_OCR", split="train", pipeline=model.load_train_pipeline()
        ),
        HuggingFaceOCRDataset(
            "MiXaiLL76/CTW1500_OCR",
            split="train_numbers",
            pipeline=model.load_train_pipeline(),
        ),
        HuggingFaceOCRDataset(
            "MiXaiLL76/ICDAR2013_OCR",
            split="train_numbers",
            pipeline=model.load_train_pipeline(),
        ),
        HuggingFaceOCRDataset(
            "MiXaiLL76/ICDAR2015_OCR",
            split="train_numbers",
            pipeline=model.load_train_pipeline(),
        ),
        HuggingFaceOCRDataset(
            "MiXaiLL76/TextOCR_OCR",
            split="train_numbers",
            pipeline=model.load_train_pipeline(),
        ),
        HuggingFaceOCRDataset(
            "MiXaiLL76/CTW1500_OCR",
            split="train_numbers",
            pipeline=model.load_train_pipeline(),
        ),
    ]

    eval_datasets = [
        HuggingFaceOCRDataset(
            "MiXaiLL76/SVHN_OCR", split="test", pipeline=model.load_test_pipeline()
        ),
        HuggingFaceOCRDataset(
            "MiXaiLL76/CTW1500_OCR",
            split="test_numbers",
            pipeline=model.load_test_pipeline(),
        ),
        HuggingFaceOCRDataset(
            "MiXaiLL76/ICDAR2013_OCR",
            split="test_numbers",
            pipeline=model.load_test_pipeline(),
        ),
        HuggingFaceOCRDataset(
            "MiXaiLL76/ICDAR2015_OCR",
            split="test_numbers",
            pipeline=model.load_test_pipeline(),
        ),
        HuggingFaceOCRDataset(
            "MiXaiLL76/TextOCR_OCR",
            split="test_numbers",
            pipeline=model.load_test_pipeline(),
        ),
        HuggingFaceOCRDataset(
            "MiXaiLL76/CTW1500_OCR",
            split="test_numbers",
            pipeline=model.load_test_pipeline(),
        ),
    ]

    log_every_n_steps = 100

    checkpoint_callback = ModelCheckpoint(
        dirpath="./checkpoints/MGP_STR",
        filename="model-{epoch:02d}-loss-{loss/total_epoch:.2f}",
        monitor="loss/total_epoch",
        save_weights_only=True,
        auto_insert_metric_name=False,
        every_n_epochs=1,
    )

    trainer = L.Trainer(
        precision="16-mixed",
        logger=tb_logger,
        log_every_n_steps=log_every_n_steps,
        callbacks=[checkpoint_callback],
        max_epochs=20,
    )

    model.dump_config(checkpoint_callback.dirpath)

    trainer.fit(
        model,
        datamodule=RecogTextDataModule(
            train_datasets=train_datasets,
            eval_datasets=eval_datasets,
            batch_size=batch_size,
        ),
    )
