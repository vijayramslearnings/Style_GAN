"""
Building blocks for StyleGAN: StyleConvBlock, NoiseInjection, and MinibatchStdDev.
"""

from typing import Optional
import torch
import torch.nn as nn
from models.adain import AdaIN


class NoiseInjection(nn.Module):
    """
    Per-pixel Gaussian noise injection with learnable channel-wise scaling.
    Captures stochastic variation (fine textures, edges) without distorting macro structure.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            batch_size, _, height, width = x.shape
            noise = torch.randn(batch_size, 1, height, width, device=x.device, dtype=x.dtype)
        return x + self.weight * noise


class StyleConvBlock(nn.Module):
    """
    Style-modulated convolutional block.
    Applies Conv2d -> Noise Injection -> LeakyReLU -> AdaIN(w).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        style_dim: int = 128,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.noise = NoiseInjection(out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.adain = AdaIN(channels=out_channels, style_dim=style_dim)

    def forward(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.conv(x)
        h = self.noise(h, noise)
        h = self.act(h)
        h = self.adain(h, w)
        return h


class MinibatchStdDev(nn.Module):
    """
    Minibatch Standard Deviation layer.
    Computes cross-sample standard deviation across the batch, averages it,
    and appends it as an extra channel to penalize mode collapse and enforce diversity.
    """

    def __init__(self, group_size: int = 4, epsilon: float = 1e-8):
        super().__init__()
        self.group_size = group_size
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        group_size = min(batch_size, self.group_size)
        if batch_size % group_size != 0:
            group_size = batch_size

        num_groups = batch_size // group_size
        y = x.view(num_groups, group_size, channels, height, width)

        # Variance across group members
        mean = y.mean(dim=1, keepdim=True)
        var = ((y - mean) ** 2).mean(dim=1, keepdim=True)
        std = torch.sqrt(var + self.epsilon)

        # Average over channels and spatial dimensions
        stat = std.mean(dim=(2, 3, 4), keepdim=True)

        # Broadcast back to (B, 1, H, W)
        stat = stat.squeeze(2).expand(num_groups, group_size, 1, height, width)
        stat = stat.contiguous().view(batch_size, 1, height, width)

        # Append as extra channel -> (B, C + 1, H, W)
        return torch.cat([x, stat], dim=1)
