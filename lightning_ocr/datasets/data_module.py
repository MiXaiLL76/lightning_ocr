import torch
import lightning as L
import numpy as np
from torch.utils.data import DataLoader
import cv2
import matplotlib.pyplot as plt
from typing import List, Optional, Union
from lightning_ocr.datasets.recog_text_dataset import RecogTextDataset
from lightning_ocr.datasets.hf_ocr_dataset import HuggingFaceOCRDataset

_valid_datasets = Union[RecogTextDataset, HuggingFaceOCRDataset]


class RecogTextDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_datasets: Union[List[_valid_datasets], _valid_datasets],
        eval_datasets: Union[List[_valid_datasets], _valid_datasets],
        batch_size: Optional[int] = 8,
        eval_batch_size: Optional[int] = None,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.eval_batch_size = (
            eval_batch_size if eval_batch_size is not None else batch_size
        )

        if not isinstance(train_datasets, list):
            train_datasets = [train_datasets]

        if not isinstance(eval_datasets, list):
            eval_datasets = [eval_datasets]

        self.train_datasets = torch.utils.data.ConcatDataset(train_datasets)
        self.eval_datasets = torch.utils.data.ConcatDataset(eval_datasets)

        weights = []
        for dataset in train_datasets:
            dataset_len = len(dataset)
            weights.append(np.linspace(1, 3, dataset_len))

        weights = np.concatenate(tuple(weights))

        self.sampler = torch.utils.data.WeightedRandomSampler(
            weights, len(self.train_datasets), replacement=False
        )

    def __len__(self):
        return len(self.train_datasets) // self.batch_size

    def train_dataloader(self):
        return DataLoader(
            self.train_datasets,
            batch_size=self.batch_size,
            # shuffle=True,
            num_workers=self.batch_size,
            collate_fn=RecogTextDataModule.collate_fn,
            sampler=self.sampler,
        )

    def val_dataloader(self):
        return DataLoader(
            self.eval_datasets,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.eval_batch_size,
            collate_fn=RecogTextDataModule.collate_fn,
        )

    @staticmethod
    def collate_fn(data_samples):
        inputs = [item["image"] for item in data_samples]

        if len(inputs) > 0:
            if isinstance(inputs[0], np.ndarray):
                inputs = np.stack([item for item in inputs], axis=0)
            else:
                inputs = torch.stack([item for item in inputs], dim=0)

        return inputs, data_samples

    @staticmethod
    def visualize_dataset(
        data_sample: dict, show: bool = False, return_fig: bool = False
    ) -> np.ndarray:
        if data_sample.get("filename"):
            data = cv2.imread(data_sample["filename"])
        elif data_sample.get("gt_image"):
            data = data_sample["gt_image"]
        else:
            return None

        fig, ax = plt.subplots()
        ax.imshow(data)
        title = [f"GT: {data_sample['gt_text']}"]
        title_kargs = {}
        if "pred_text" in data_sample:
            title.append(f"DT: {data_sample['pred_text']}")
            if data_sample["pred_text"].strip() == data_sample["gt_text"].strip():
                title_kargs["color"] = "green"
            else:
                title_kargs["color"] = "red"

        ax.set_title("\n".join(title), **title_kargs)

        fig.canvas.draw()  # Draw the canvas, cache the renderer

        if show:
            plt.show()
            return

        if return_fig:
            return fig

        # Convert the canvas to a raw RGB buffer
        buf = fig.canvas.buffer_rgba()
        ncols, nrows = fig.canvas.get_width_height()
        image = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 4)
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        return image


if __name__ == "__main__":
    import albumentations as A

    pipeline = [A.Resize(32, 128)]
    data_module = RecogTextDataModule(
        train_datasets=[
            HuggingFaceOCRDataset(
                "MiXaiLL76/7SEG_OCR", split="train", pipeline=pipeline
            ),
            HuggingFaceOCRDataset(
                "MiXaiLL76/SVHN_OCR", split="train", pipeline=pipeline
            ),
            HuggingFaceOCRDataset(
                "MiXaiLL76/CTW1500_OCR", split="train_numbers", pipeline=pipeline
            ),
            HuggingFaceOCRDataset(
                "MiXaiLL76/ICDAR2013_OCR", split="train_numbers", pipeline=pipeline
            ),
            HuggingFaceOCRDataset(
                "MiXaiLL76/ICDAR2015_OCR", split="train_numbers", pipeline=pipeline
            ),
            HuggingFaceOCRDataset(
                "MiXaiLL76/TextOCR_OCR", split="train_numbers", pipeline=pipeline
            ),
            HuggingFaceOCRDataset(
                "MiXaiLL76/CTW1500_OCR", split="train_numbers", pipeline=pipeline
            ),
        ],
        eval_datasets=[
            HuggingFaceOCRDataset(
                "MiXaiLL76/SVHN_OCR", split="test", pipeline=pipeline
            ),
            HuggingFaceOCRDataset(
                "MiXaiLL76/CTW1500_OCR", split="test_numbers", pipeline=pipeline
            ),
            HuggingFaceOCRDataset(
                "MiXaiLL76/ICDAR2013_OCR", split="test_numbers", pipeline=pipeline
            ),
            HuggingFaceOCRDataset(
                "MiXaiLL76/ICDAR2015_OCR", split="test_numbers", pipeline=pipeline
            ),
            HuggingFaceOCRDataset(
                "MiXaiLL76/TextOCR_OCR", split="test_numbers", pipeline=pipeline
            ),
            HuggingFaceOCRDataset(
                "MiXaiLL76/CTW1500_OCR", split="test_numbers", pipeline=pipeline
            ),
        ],
        batch_size=8,
        eval_batch_size=1,
    )
    i = 0
    for _, data_samples in data_module.train_dataloader():
        print(i, len(data_samples))
        for item in data_samples:
            print(item["_dataset"], item["gt_text"])

        i += 1

        if i > 5:
            break
