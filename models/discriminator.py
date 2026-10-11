"""
Discriminator Network with Minibatch Standard Deviation for StyleGAN.

Downsamples 32x32 -> 16x16 -> 8x8 -> 4x4.
Appends MinibatchStdDev before the final 4x4 classification block.
"""

import torch
import torch.nn as nn
from models.blocks import MinibatchStdDev


class StyleDiscriminator(nn.Module):
    """
    Convolutional Discriminator with Minibatch Standard Deviation.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        self.in_channels = in_channels

        # 32x32 -> 16x16
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 16x16 -> 8x8
        self.block2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 8x8 -> 4x4
        self.block3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Minibatch Standard Deviation: appends 1 channel -> (base_channels * 4 + 1)
        self.mbstd = MinibatchStdDev(group_size=4)

        # Final 4x4 classification block
        self.final_block = nn.Sequential(
            nn.Conv2d(base_channels * 4 + 1, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(base_channels * 4 * 4 * 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: RGB image tensor of shape (B, 3, 32, 32).
        Returns:
            Scalar classification logits of shape (B, 1).
        """
        h = self.block1(x)
        h = self.block2(h)
        h = self.block3(h)
        h = self.mbstd(h)
        logits = self.final_block(h)
        return logits
