import os.path as osp
from typing import Callable, List, Optional, Union
import pandas as pd
import torch
import numpy as np
from PIL import Image
import albumentations as A
import cv2
import matplotlib.pyplot as plt
import lightning as L
from torch.utils.data import DataLoader


class RecogTextDataset(torch.utils.data.Dataset):
    r"""RecogTextDataset for text recognition.

    The annotation file is in jsonl format, it should be a list of dicts.

    The annotation formats are shown as follows.
    - jsonl format
    .. code-block:: none

        ``{"filename": "test_img1.jpg", "text": "OpenMMLab"}``
        ``{"filename": "test_img2.jpg", "text": "MMOCR"}``

    Args:
        ann_file (str): Annotation file path. Defaults to ''.
        parse_cfg (dict, optional): Config of parser for parsing annotations.Defaults to
            ``{
                     "dtype" : {'text':str, 'filename' : str},
                     "join_path" : 'filename',
                }``.
        data_root (str): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to ''.
        data_prefix (dict): Prefix for training data. Defaults to
            ``dict(img_path='')``.
        pipeline (list, optional): Processing pipeline. Defaults to [].
    """

    def __init__(
        self,
        ann_file: str = "",
        parser_cfg: Optional[dict] = {
            "dtype": {"text": str, "filename": str},
        },
        data_root: Optional[str] = "",
        data_prefix: dict = dict(img_path=""),
        pipeline: List[Callable] = [],
        gt_text_row: str = "text",
        image_row: str = "filename",
    ) -> None:
        self.ann_file = ann_file
        self.parser_cfg = parser_cfg
        self.data_root = data_root
        self.data_prefix = data_prefix

        self.gt_text_row = gt_text_row
        self.image_row = image_row

        self.data_list: List[dict] = self.load_data_list()
        self.transform = A.Compose(pipeline)

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """

        def join_path(row):
            row[self.image_row] = osp.join(
                self.data_root, self.data_prefix["img_path"], row[self.image_row]
            )
            return row

        data_frame = pd.read_json(
            osp.join(self.data_root, self.ann_file), lines=True, **self.parser_cfg
        )
        data_frame = data_frame.apply(join_path, axis=1)

        columns = []
        for col in data_frame.columns:
            if col == self.gt_text_row:
                columns.append("gt_text")
            elif col == self.image_row:
                columns.append("filename")
            else:
                columns.append(col)
        data_frame.columns = columns

        return data_frame.to_dict("records")

    def __getitem__(self, index):
        item: dict = self.data_list[index]
        item["index"] = index
        pillow_image = Image.open(item["filename"]).convert("RGB")
        item["image"] = self.transform(image=np.array(pillow_image))["image"]
        return item

    def __len__(self):
        return len(self.data_list)

    @staticmethod
    def collate_fn(data_samples):
        inputs = [item["image"] for item in data_samples]

        if len(inputs) > 0:
            if isinstance(inputs[0], np.ndarray):
                inputs = np.stack([item for item in inputs], axis=0)
            else:
                inputs = torch.stack([item for item in inputs], dim=0)

        return inputs, data_samples


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


def visualize_dataset(
    data_sample: dict, show: bool = False, return_fig: bool = False
) -> np.ndarray:
    data = cv2.imread(data_sample["filename"])
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

    if return_fig:
        return fig

    # Convert the canvas to a raw RGB buffer
    buf = fig.canvas.buffer_rgba()
    ncols, nrows = fig.canvas.get_width_height()
    image = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 4)
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    return image
