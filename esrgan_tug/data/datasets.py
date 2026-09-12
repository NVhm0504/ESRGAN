"""PyTorch datasets."""
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..utils import imread_rgb, list_images
from .degradation import degrade, degrade_high_order


def to_tensor(img_u8):
    return torch.from_numpy(np.ascontiguousarray(img_u8.transpose(2, 0, 1))).float().div_(255.0)


def to_u8(t):
    return (t.detach().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy())


def _augment(img, rng):
    if rng.random() < 0.5:
        img = img[:, ::-1]
    if rng.random() < 0.5:
        img = img[::-1]
    return np.rot90(img, int(rng.integers(4)))


class SRTrainDataset(Dataset):
    """HR files -> random HR crop -> augmentation -> LR by Eq. (1) re-sampled per call."""

    def __init__(self, files, patch=192, scale=4, degradation="eq1", deg_cfg=None, augment=True):
        if not files:
            raise FileNotFoundError("no HR training images found")
        self.files, self.patch, self.scale = files, patch, scale
        self.degradation, self.deg_cfg, self.augment = degradation, deg_cfg, augment

    @staticmethod
    def collect(dirs):
        return [os.path.join(d, f) for d in dirs if d and os.path.isdir(d) for f in list_images(d)]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        rng = np.random.default_rng(int(torch.randint(0, 2**62, (1,)).item()))
        hr = imread_rgb(self.files[i])
        p = self.patch
        h, w = hr.shape[:2]
        if h < p or w < p:
            hr = np.pad(hr, ((0, max(0, p - h)), (0, max(0, p - w)), (0, 0)), mode="reflect")
            h, w = hr.shape[:2]
        y, x = rng.integers(0, h - p + 1), rng.integers(0, w - p + 1)
        hr = hr[y:y + p, x:x + p]
        if self.augment:
            hr = np.ascontiguousarray(_augment(hr, rng))
        if self.degradation == "high_order":
            lr = degrade_high_order(hr, self.scale, rng)
        else:
            lr, _ = degrade(hr, self.scale, rng=rng, cfg=self.deg_cfg)
        return to_tensor(lr), to_tensor(hr)


class PairedFolder(Dataset):
    """LR/HR pairs matched by file name (HR optional)."""

    def __init__(self, lr_dir, hr_dir=None, limit=None):
        self.lr_dir, self.hr_dir = lr_dir, hr_dir
        self.names = list_images(lr_dir)[:limit] if limit else list_images(lr_dir)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        n = self.names[i]
        item = {"name": n, "lr": to_tensor(imread_rgb(os.path.join(self.lr_dir, n)))}
        if self.hr_dir:
            item["hr_u8"] = imread_rgb(os.path.join(self.hr_dir, n))
        return item


class LabelledImages(Dataset):
    """Images + land-cover labels from a manifest (id,label[,split])."""

    def __init__(self, img_dir, manifest, split=None, augment=False):
        df = pd.read_csv(manifest)
        if split is not None and "split" in df:
            df = df[df.split == split]
        self.img_dir, self.augment = img_dir, augment
        self.ids, self.labels = df.id.tolist(), df.label.astype(int).tolist()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img = imread_rgb(os.path.join(self.img_dir, self.ids[i] + ".png"))
        if self.augment:
            rng = np.random.default_rng(int(torch.randint(0, 2**62, (1,)).item()))
            img = _augment(img, rng)
        return to_tensor(img), self.labels[i]
