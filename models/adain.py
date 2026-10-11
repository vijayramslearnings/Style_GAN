"""
Adaptive Instance Normalization (AdaIN).

Modulates normalized feature map channels via learnable affine projections
(scale and shift) driven by the style vector w:
    AdaIN(x_i, w) = y_{s,i}(w) * ((x_i - mu(x_i)) / sigma(x_i)) + y_{b,i}(w)
"""

import torch
import torch.nn as nn


class AdaIN(nn.Module):
    """
    Adaptive Instance Normalization block.
    Computes per-channel mean and variance across spatial dimensions,
    normalizes the feature map, and modulates with style-derived scale and bias.
    """

    def __init__(self, channels: int, style_dim: int = 128, epsilon: float = 1e-8):
        super().__init__()
        self.channels = channels
        self.epsilon = epsilon
        # Linear layer mapping w -> scale (y_s) and shift (y_b)
        self.linear = nn.Linear(style_dim, channels * 2)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature map tensor of shape (B, C, H, W).
            w: Style vector of shape (B, style_dim).
        Returns:
            Style-modulated tensor of shape (B, C, H, W).
        """
        batch_size, channels, _, _ = x.shape

        # Instance Normalization across spatial dimensions H, W
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.epsilon)

        # Affine projections: y_s (scale) and y_b (shift)
        style = self.linear(w).view(batch_size, 2, channels, 1, 1)
        scale = style[:, 0] + 1.0  # Centered at 1.0
        shift = style[:, 1]         # Centered at 0.0

        return scale * x_norm + shift
