import torch
import json
import os
import lightning as L
import typing
from lightning_ocr.modules.backbones import ResNetABI
from lightning_ocr.modules.encoders import ABIEncoder
from lightning_ocr.modules.decoders import ABIVisionDecoder
from lightning_ocr.datasets import HuggingFaceOCRDataset, RecogTextDataModule
from lightning_ocr.models.base import BaseOcrModel
from lightning_ocr.tokenizer import FastTokenizer
from lightning_ocr.mmocr_compatible.abinet import load_mmocr_state_dict
from transformers import ViTImageProcessor


BASE_PROCESSOR_CFG = {
    "do_convert_rgb": None,
    "do_normalize": False,
    "do_rescale": True,
    "do_resize": True,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "resample": 3,
    "rescale_factor": 0.00392156862745098,
    "size": {"height": 32, "width": 128},
}


class ABINetVision(BaseOcrModel):
    def __init__(self, config: dict = {}):
        super().__init__(
            config=config,
            base_pretrained_model="https://download.openmmlab.com/mmocr/textrecog/abinet/abinet-vision_20e_st-an_mj/abinet-vision_20e_st-an_mj_20220915_152445-85cfb03d.pth",
            image_height=32,
            image_width=128,
        )
        if ("mmocr" in self.pretrained_model) or ("openmmlab" in self.pretrained_model):
            self.load_mmocr_model(self.pretrained_model)

        config["lr"] = config.get("lr", 1e-4)
        self.max_token_length = config.get("max_seq_len", 26)

        if config.get("tokenizer") is not None:
            self.tokenizer = FastTokenizer(**config.get("tokenizer", {}))
        else:
            raise ValueError(
                "Tokenizer not found in config, abinet have not pretrained tokenizer."
            )

        self.backbone = ResNetABI(**config.get("backbone", {}))
        self.encoder = ABIEncoder(**config.get("encoder", {}))
        self.decoder = ABIVisionDecoder(
            num_classes=self.tokenizer.vocab_size,
            max_seq_len=self.max_token_length,
            **config.get("decoder", {}),
        )
        vit_processor_cfg = dict(
            BASE_PROCESSOR_CFG,
            **{"size": self.image_size},
        )
        self.processor = ViTImageProcessor(
            **dict(vit_processor_cfg, **config.get("processor", {}))
        )

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

    def load_mmocr_model(self, model_path):
        if "http" in model_path:
            import os
            import urllib.request

            _model_path = os.path.basename(model_path)
            if not os.path.exists(_model_path):
                urllib.request.urlretrieve(model_path, _model_path)
        else:
            _model_path = model_path

        state_dict = load_mmocr_state_dict(_model_path)
        self.load_state_dict(state_dict, strict=False)

    def predict(self, images):
        if not isinstance(images, list):
            images = [images]

        inputs = self.processor.preprocess(images, return_tensors="pt")["pixel_values"]
        outputs = self.forward(inputs.to(self.device))
        tokens, _ = self.logits_postprocessor(outputs[0])
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

    def forward(self, inputs: torch.Tensor, softmax: bool = True):
        attn_vecs, out_enc, attn_scores = self.decoder(
            self.encoder(self.backbone(inputs))
        )
        if softmax:
            attn_vecs = torch.nn.functional.softmax(attn_vecs, dim=-1)
        return attn_vecs, out_enc, attn_scores

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
        labels = tokenize(samples, self.tokenizer)
        return labels

    def calc_loss(self, pred_logits, target_labels):
        return torch.nn.functional.cross_entropy(
            pred_logits.view(-1, pred_logits.size(-1)), target_labels.view(-1)
        )

    def training_step(self, batch, batch_idx):
        inputs, data_samples = batch
        inputs = self.processor.preprocess(inputs, return_tensors="pt")["pixel_values"]
        outputs = self.forward(inputs.to(self.device), softmax=False)
        labels = self.tokenizer_encode(data_samples).to(self.device)
        total_loss = self.calc_loss(outputs[0], labels)

        losses = {
            "total": total_loss,
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

        return total_loss

    def logits_postprocessor(self, preds_max_prob):
        # preds_max_prob = torch.nn.functional.softmax(logits, dim=-1)
        out_tokens, conf_scores = [], []

        scores, tokens = preds_max_prob.max(dim=-1)
        tokens = tokens.detach().cpu().tolist()
        scores = scores.detach().cpu().tolist()

        for index in range(preds_max_prob.size(0)):
            try:
                pred_eos = tokens[index].index(self.tokenizer.eos_token_id)
            except ValueError:
                pred_eos = len(tokens[index])

            out_tokens.append(tokens[index][:pred_eos])
            conf_scores.append(scores[index][:pred_eos])

        return out_tokens, conf_scores

    def validation_step(self, batch, batch_idx):
        inputs, data_samples = batch
        inputs = self.processor.preprocess(inputs, return_tensors="pt")["pixel_values"]
        outputs = self.forward(inputs.to(self.device))

        tokens, scores = self.logits_postprocessor(outputs[0])

        for i in range(len(data_samples)):
            data_samples[i]["pred_text"] = self.tokenizer.decode(
                tokens[i], skip_special_tokens=True
            )
            data_samples[i]["pred_score"] = scores

        return data_samples

    def dump_config(self, output_folder):
        os.makedirs(output_folder, exist_ok=True)

        with open(f"{output_folder}/base_config.json", "w") as f:
            json.dump(self.base_config, f, indent=4)

        self.processor.save_pretrained(output_folder)
        self.tokenizer.save_pretrained(output_folder)

        with open(f"{output_folder}/vocab.json", "w") as fd:
            json.dump(self.tokenizer.vocab, fd)

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

    def to_onnx(self, file_path: str, **kwargs: typing.Any) -> None:
        input_sample = torch.randn(
            2, 3, self.image_size["height"], self.image_size["width"]
        )

        torch.onnx.export(
            self,
            input_sample,
            file_path,
            input_names=["input"],
            output_names=["output", "out_enc", "attn_scores"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
                "out_enc": {0: "batch_size"},
                "attn_scores": {0: "batch_size"},
            },
            **kwargs,
        )


if __name__ == "__main__":
    import os
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import TensorBoardLogger

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    batch_size = 32

    config = {
        "max_seq_len": 12,
        "tokenizer": {
            "dict_list": list("0123456789.-;:"),
        },
    }
    model = ABINetVision(config)
    tb_logger = TensorBoardLogger(save_dir="logs/abinet/")

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
        dirpath="./checkpoints/abinet",
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
        max_epochs=50,
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
