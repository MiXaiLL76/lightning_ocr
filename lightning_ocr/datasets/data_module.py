import torch
import lightning as L
from torch.utils.data import DataLoader
from typing import List, Optional, Union
from lightning_ocr.datasets.recog_text_dataset import RecogTextDataset

class RecogTextDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_datasets: Union[List[RecogTextDataset], RecogTextDataset],
        eval_datasets: Union[List[RecogTextDataset], RecogTextDataset],
        batch_size: Optional[int] = 8,
        eval_batch_size: Optional[int] = None,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.eval_batch_size = (
            eval_batch_size if eval_batch_size is not None else batch_size
        )

        if isinstance(train_datasets, RecogTextDataset):
            train_datasets = [train_datasets]

        if isinstance(eval_datasets, RecogTextDataset):
            eval_datasets = [eval_datasets]

        self.train_datasets = torch.utils.data.ConcatDataset(train_datasets)
        self.eval_datasets = torch.utils.data.ConcatDataset(eval_datasets)

    def __len__(self):
        return len(self.train_datasets) // self.batch_size

    def train_dataloader(self):
        return DataLoader(
            self.train_datasets,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.batch_size,
            collate_fn=RecogTextDataset.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.eval_datasets,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.eval_batch_size,
            collate_fn=RecogTextDataset.collate_fn,
        )
