import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pytorch_lightning as pl
import torch
from PIL import Image
from torch import FloatTensor, LongTensor
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms

from .vocab import CROHMEVocab

vocab = CROHMEVocab()

MAX_SIZE = 32e4  # change here accroading to your GPU memory

@dataclass
class Data:
    fpath: str
    strokeImgs: FloatTensor # [stroke_num, h, w]
    positions: FloatTensor # [4]
    label: List[str]


import math
# load data
def data_iterator(
    data: List[Data],
    batch_size: int,
):
    data.sort(key=lambda x: len(x.strokeImgs))
    return [data[i * batch_size: (i + 1) * batch_size] for i in range(math.ceil(len(data) / batch_size))]


import numpy as np
def extract_data(paths: List[str]) -> List[Data]:
    """Extract all data need for a dataset from zip archive

    Args:
        archive (ZipFile):
        dir_name (str): dir name in archive zip (eg: train, test_2014......)

    Returns:
        Data: list of tuple of image and formula
    """
    import glob

    data = []
    for img_dir in paths:
        files = glob.glob(img_dir, recursive=True)
        if len(files) == 0:
            print(f"WARN: {img_dir} is empty")
        for fpath in files:
            rawdata = np.load(fpath, allow_pickle=True).tolist()
            label = rawdata['label']
            positions = FloatTensor(rawdata['positions'])
            strokeImgs = FloatTensor(rawdata['strokeImgs'])
            data.append(Data(fpath, strokeImgs, positions, label))

    print(f"Extract data from: {paths}, with data size: {len(data)}")
    return data


@dataclass
class Batch:
    img_bases: List[str]  # [b,]
    strokeImgs: FloatTensor  # [b, stroke_num, H, W]
    strokeMasks: FloatTensor # [b, stroke_num]
    positions: FloatTensor # [batch_size, stroke_num, 4(left, top, right, bottom)]
    wordLabels: List[List[int]]  # [b, l]

    def __len__(self) -> int:
        return len(self.img_bases)

    def to(self, device) -> "Batch":
        return Batch(
            img_bases=self.img_bases,
            strokeImgs=self.strokeImgs.to(device),
            strokeMasks=self.strokeMasks.to(device),
            positions=self.positions.to(device),
            wordLabels=self.wordLabels,
        )

import  torch.nn.functional  as F

def collate_fn(batch: List[Data]):
    assert len(batch) == 1
    batch = batch[0] 
    labels = [d.label for d in batch]


    stroke_nums = [d.strokeImgs.size(0) for d in batch]
    maxSnum = max(stroke_nums)

    strokeImgss = torch.stack([F.pad(d.strokeImgs, (0,0,0,0,0, maxSnum - d.strokeImgs.size(0)), 'constant', 0) for d in batch])
    positionss = torch.stack([F.pad(d.positions, (0,0,0, maxSnum - d.positions.size(0)), 'constant', 0) for d in batch])

    fnames = [d.fpath for d in batch]
    seqs_y = [vocab.words2indices(x) for x in labels]

    strokeMask = torch.zeros((len(batch), maxSnum), dtype=torch.bool)
    for idx, d in enumerate(strokeImgss):
        strokeMask[idx, len(d):] = 1

    return Batch(fnames, strokeImgss, strokeMask, positionss, seqs_y)


def build_dataset(img_dir_paths: List[str], batch_size: int):
    data = extract_data(img_dir_paths)
    return data_iterator(data, batch_size)


class StrokeDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        train_dir_paths: List[str],
        test_dir_paths: List[str],
        batch_size: int = 8,
        num_workers: int = 5,
    ) -> None:
        super().__init__()
        self.train_dir_paths = train_dir_paths
        self.test_dir_paths = test_dir_paths
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = build_dataset(self.train_dir_paths, self.batch_size)
            self.val_dataset = build_dataset(self.test_dir_paths, 1)
        if stage == "test" or stage is None:
            self.test_dataset = build_dataset(self.test_dir_paths, 1)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )


