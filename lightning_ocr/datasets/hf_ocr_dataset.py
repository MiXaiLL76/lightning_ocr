from typing import Callable, List
import torch
import numpy as np
import albumentations as A
from datasets import load_dataset
import os


class HuggingFaceOCRDataset(torch.utils.data.Dataset):
    def __init__(
        self, dataset_path, split: str = None, pipeline: List[Callable] = [], **kwargs
    ) -> None:
        self.dataset_path = dataset_path
        self.dataset = load_dataset(dataset_path, split=split, **kwargs)
        self.transform = A.Compose(pipeline)

    def __getitem__(self, index):
        item = {
            "index": index,
            "_dataset": os.path.basename(self.dataset_path),
            "gt_image": self.dataset[index]["image"].convert("RGB"),
            "gt_text": self.dataset[index]["text"],
        }
        item["image"] = self.transform(image=np.array(item["gt_image"]))["image"]
        return item

    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    dataset = HuggingFaceOCRDataset("MiXaiLL76/CTW1500_OCR", split="train_numbers")
    print(len(dataset))
    print(dataset[0])
