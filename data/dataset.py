"""
Dataset transformation and DataLoader pipeline for CIFAR-10.
"""

import os
from typing import Optional, Tuple
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def get_cifar10_transforms() -> transforms.Compose:
    """
    Standard preprocessing transforms:
    - Resize to 32x32
    - Random Horizontal Flip (p=0.5)
    - Convert to Tensor
    - Normalize to [-1, 1] via (x - 0.5) / 0.5
    """
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def get_cifar10_dataloader(
    root: str = "./data",
    batch_size: int = 64,
    num_workers: int = 2,
    shuffle: bool = True,
) -> DataLoader:
    """
    Downloads and prepares CIFAR-10 training DataLoader.
    """
    transform = get_cifar10_transforms()
    dataset = datasets.CIFAR10(
        root=root,
        train=True,
        download=True,
        transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
