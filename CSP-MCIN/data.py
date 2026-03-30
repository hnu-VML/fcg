#!/usr/bin/env python3
"""
Data loading utilities for FusionNet training and inference.

This module builds paired SAR/optical datasets and the corresponding
``DataLoader`` objects so the training loop can stay focused on model logic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

LOGGER = logging.getLogger(__name__)
Image.MAX_IMAGE_PIXELS = None


@dataclass
class SamplePaths:
    sar: Path
    opt: Path
    name: str


class PairedImageDataset(Dataset):
    """Loads paired SAR/optical images that share the same stem name."""

    IMG_EXTENSIONS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    def __init__(
        self,
        sar_dir: Path,
        opt_dir: Path,
        image_size: int,
        inchannel: tuple = (1, 3),
    ) -> None:
        self.sar_dir = sar_dir.expanduser().resolve()
        self.opt_dir = opt_dir.expanduser().resolve()
        self.image_size = image_size
        self.inchannel = inchannel
        self.resize = transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR)
        self.to_tensor = transforms.ToTensor()
        self.samples: List[SamplePaths] = self._build_samples()
        LOGGER.info("Loaded %d paired samples.", len(self.samples))

    def _build_samples(self) -> List[SamplePaths]:
        sar_files = self._scan_dir(self.sar_dir)
        opt_files = self._scan_dir(self.opt_dir)
        common = sorted(set(sar_files) & set(opt_files))
        samples: List[SamplePaths] = []
        for stem in common:
            sar_path = sar_files[stem]
            opt_path = opt_files[stem]
            samples.append(SamplePaths(sar=sar_path, opt=opt_path, name=stem))
        return samples

    @staticmethod
    def _scan_dir(directory: Path) -> dict:
        files = {}
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix.lower() in PairedImageDataset.IMG_EXTENSIONS:
                files[path.stem] = path
        return files

    def __len__(self) -> int:
        return len(self.samples)

    def _load_sar_image(self, path: Path) -> torch.Tensor:
        if not path or not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        ori_image = Image.open(path)
        if self.inchannel[0] == 1:
            image = ori_image.convert("L")
        else:
            image = ori_image.convert("RGB")
        image = self.resize(image)
        sar_3ch = ori_image.convert("RGB")
        sar_3ch = self.resize(sar_3ch)
        return self.to_tensor(image), self.to_tensor(sar_3ch)

    def _load_opt_image(self, path: Path) -> torch.Tensor:
        if not path or not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        ori_image = Image.open(path)
        if self.inchannel[1] == 1:
            image = ori_image.convert("L")
        else:
            image = ori_image.convert("RGB")
        image = self.resize(image)
        opt_ycbcr = ori_image.convert("YCbCr")
        opt_ycbcr = self.resize(opt_ycbcr)
        opt_rgb = ori_image.convert("RGB")
        opt_rgb = self.resize(opt_rgb)
        return self.to_tensor(image), self.to_tensor(opt_ycbcr), self.to_tensor(opt_rgb)

    def __getitem__(self, idx: int):
        "sar,opt是模型输入进行前向的，按照模型架构动态调整"
        sample = self.samples[idx]
        sar, sar_3ch = self._load_sar_image(sample.sar)
        opt, opt_ycbcr, opt_rgb = self._load_opt_image(sample.opt)
        return {"sar": sar, "sar_3ch": sar_3ch, "opt": opt, "opt_ycbcr": opt_ycbcr, "opt_rgb": opt_rgb, "name": sample.name}


def create_train_and_val_dataloaders(
    train_dir: Path,
    val_dir: Optional[Path],
    image_size: int,
    inchannel: tuple,
    train_batch_size: int,
    val_batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Build train/val dataloaders for paired SAR/optical images.

    Args:
        train_dir: Directory containing ``sar`` and ``opt`` folders for training.
        val_dir: Directory containing ``sar`` and ``opt`` folders for validation.
        image_size: Square resize dimension for all images.
        inchannel: Tuple of channels for SAR and optical images.
        train_batch_size: Batch size for training.
        val_batch_size: Batch size for validation.
        num_workers: Number of dataloader workers.
        pin_memory: Whether to enable ``pin_memory``.
    """
    train_dataset = PairedImageDataset(
        sar_dir=train_dir / "sar",
        opt_dir=train_dir / "opt",
        image_size=image_size,
        inchannel=inchannel,
    )
    val_dataset = None
    if val_dir is not None:
        val_dataset = PairedImageDataset(
            sar_dir=val_dir / "sar",
            opt_dir=val_dir / "opt",
            image_size=image_size,
            inchannel=inchannel,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
        if val_dataset is not None
        else None
    )
    return train_loader, val_loader
