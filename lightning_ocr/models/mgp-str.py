import torch
import lightning as L
import albumentations as A
from lightning_ocr.datasets.recog_text_dataset import (
    RecogTextDataset,
    RecogTextDataModule,
    visualize_dataset,
)
from lightning_ocr.metrics.recog_metric import WordMetric, OneMinusNEDMetric, CharMetric
from torch.utils.tensorboard import SummaryWriter
from transformers import MgpstrProcessor, MgpstrForSceneTextRecognition, MgpstrConfig
from lightning_ocr.dictionary.tokenization_mgp_str import MgpstrTokenizer

# https://huggingface.co/docs/transformers/v4.33.2/en/model_doc/mgp-str
class MGP_STR(L.LightningModule):
    def __init__(self, config: dict = {}):
        super().__init__()
        
        char_tokenizer = MgpstrTokenizer(**config.get("tokenizer", {}))

        pretrained_model = config.get("pretrained_model", 'alibaba-damo/mgp-str-base')
        
        self.cfg = MgpstrConfig.from_pretrained(pretrained_model)
        self.cfg.num_character_labels = len(char_tokenizer.vocab)
        self.cfg.max_token_length = config.get("max_seq_len", self.cfg.max_token_length)

        self.processor = MgpstrProcessor.from_pretrained(pretrained_model)
        self.processor.char_tokenizer = char_tokenizer

        # https://github.com/huggingface/transformers/blob/v4.33.2/src/transformers/models/mgp_str/processing_mgp_str.py#L143
        self.processor.bpe_tokenizer.pad_token_id = self.processor.bpe_tokenizer.eos_token_id
        self.processor.wp_tokenizer.eos_token_id = 102

        self.model = MgpstrForSceneTextRecognition.from_pretrained(pretrained_model, config=self.cfg, ignore_mismatched_sizes=True)
        
        self.metrics = [
            WordMetric(mode=["exact", "ignore_case", "ignore_case_symbol"]),
            CharMetric(),
            OneMinusNEDMetric(),
        ]
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=1e-4,
        )

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=self.trainer.max_epochs * len(self.trainer.fit_loop._data_source.instance), 
                eta_min=1e-7, 
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]

    def tokenizer_encode(self, data_samples):
        def tokenize(samples, tokenizer_class):

            if tokenizer_class.bos_token is not None:
                samples = [f"{tokenizer_class.bos_token}{text}" for text in samples]

            tokens = tokenizer_class(
                samples, 
                return_tensors="pt", 
                pad_to_multiple_of=self.cfg.max_token_length,
                padding=True,
                max_length=self.cfg.max_token_length,
                truncation=True,
            )
            labels = tokens['input_ids']
            labels[tokens['attention_mask'] == 0] = tokenizer_class.eos_token_id
            return labels
        
        samples = [item['gt_text'] for item in data_samples]
        char_labels = tokenize(samples, self.processor.char_tokenizer)
        bpe_labels = tokenize(samples, self.processor.bpe_tokenizer)
        wp_labels = tokenize(samples, self.processor.wp_tokenizer)
        return char_labels, bpe_labels, wp_labels
    
    def calc_loss(self, pred_logits, target_labels):
        return torch.nn.functional.cross_entropy(pred_logits.view(-1, pred_logits.size(-1)), target_labels.view(-1))
    
    def training_step(self, batch, batch_idx):
        inputs, data_samples = batch

        pixel_values = self.processor(images=inputs, return_tensors="pt").pixel_values.to(self.model.device)
        outputs = self.model(pixel_values)

        char_labels, bpe_labels, wp_labels = self.tokenizer_encode(data_samples)
        char_loss = self.calc_loss(outputs.logits[0], char_labels.to(self.model.device))
        bpe_loss = self.calc_loss(outputs.logits[1], bpe_labels.to(self.model.device))
        wp_loss = self.calc_loss(outputs.logits[2], wp_labels.to(self.model.device))
        loss = char_loss + bpe_loss + wp_loss

        losses = {
            "char_loss" : char_loss,
            "bpe_loss" : bpe_loss,
            "wp_loss" : wp_loss,
            "total" : loss,
        }
        
        self.log_dict(
            {f"loss/{key}": val for key, val in losses.items()},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(data_samples),
        )

        lr = self.optimizers().param_groups[0]['lr']  # Get current learning rate
        self.log('learning_rate', lr, on_step=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, data_samples = batch

        pixel_values = self.processor(images=inputs, return_tensors="pt").pixel_values.to(self.model.device)
        outputs = self.model(pixel_values)
        return outputs

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        inputs, data_samples = batch

        generated_text = self.processor.batch_decode(outputs.logits)['generated_text']

        for idx, data_sample in enumerate(data_samples):
            data_sample['pred_text'] = generated_text[idx]
        
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
    ]
    return train_pipeline


def load_test_pipeline():
    test_pipeline = [
        A.Resize(32, 128),
    ]
    return test_pipeline


if __name__ == "__main__":
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import TensorBoardLogger
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    batch_size = 8

    tb_logger = TensorBoardLogger(save_dir="logs/MGP_STR")

    train_dataset = RecogTextDataset(
        data_root="/home/mixaill76/text_datasets/data_collection/005-CV",
        ann_file="ann_file.json",
        pipeline=load_train_pipeline(),
    )

    log_every_n_steps = 50
    if len(train_dataset) // batch_size < 50:
        log_every_n_steps = 5

    checkpoint_callback = ModelCheckpoint(
        dirpath="./checkpoints/MGP_STR",
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

    cfg = {
        "tokenizer" : {
            "dict_list" : list("0123456789."),
        },
    }

    model = MGP_STR(cfg)
    
    from sklearn.model_selection import train_test_split
    import copy
    TRAIN, TEST = train_test_split(train_dataset.data_list, test_size=0.2, random_state=42)
    
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
