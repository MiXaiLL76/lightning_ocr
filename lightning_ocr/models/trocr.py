import torch
import os
import json
import lightning as L
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    VisionEncoderDecoderConfig,
)
import typing
from lightning_ocr.datasets import HuggingFaceOCRDataset, RecogTextDataModule
from lightning_ocr.tokenizer import FastTokenizer
from lightning_ocr.models.base import BaseOcrModel


# https://huggingface.co/docs/transformers/model_doc/trocr#transformers.TrOCRForCausalLM.forward.example
class TrOCR(BaseOcrModel):
    def __init__(self, config: dict = {}):
        super().__init__(
            config=config,
            base_pretrained_model="microsoft/trocr-small-printed",
            image_height=384,
            image_width=384,
        )

        self.processor = TrOCRProcessor.from_pretrained(self.pretrained_model)
        self.processor.image_processor.size = self.image_size

        if config.get("tokenizer") is not None:
            self.processor.tokenizer = FastTokenizer(**config.get("tokenizer", {}))
            self.processor.tokenizer_class = "PreTrainedTokenizerFast"

        self.cfg = VisionEncoderDecoderConfig.from_pretrained(self.pretrained_model)

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

        if self.init_from_pretrained_model:
            self.model = VisionEncoderDecoderModel.from_pretrained(
                self.pretrained_model, config=self.cfg, ignore_mismatched_sizes=True
            )
        else:
            self.model = VisionEncoderDecoderModel(self.cfg)

    def predict(self, images):
        pixel_values = self.processor(images, return_tensors="pt").pixel_values.to(
            self.model.device
        )
        generated_ids = self.model.generate(
            pixel_values, max_new_tokens=self.max_token_length
        )
        return self.processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )

    def dump_config(self, output_folder):
        os.makedirs(output_folder, exist_ok=True)

        with open(f"{output_folder}/base_config.json", "w") as f:
            json.dump(self.base_config, f, indent=4)

        self.processor.save_pretrained(output_folder)
        self.processor.tokenizer.save_pretrained(output_folder)
        with open(f"{output_folder}/vocab.json", "w") as fd:
            json.dump(self.processor.tokenizer.vocab, fd)

        self.model.save_pretrained(output_folder, state_dict={})

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

        generated_texts = self.processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        for idx, data_sample in enumerate(data_samples):
            data_sample["pred_text"] = generated_texts[idx]

        return data_samples

    def forward(self, inputs: torch.Tensor):
        generated_ids = self.model.generate(
            inputs, max_new_tokens=self.max_token_length
        )
        return generated_ids


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

    model = TrOCR(config)

    tb_logger = TensorBoardLogger(save_dir="logs/TrOCR")

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
        dirpath="./checkpoints/TrOCR",
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
