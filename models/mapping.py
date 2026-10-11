"""
Mapping Network for StyleGAN.

A 4-layer Multi-Layer Perceptron (MLP) mapping random Gaussian noise
vector z in R^128 into an intermediate disentangled style space w in R^128.
"""

import torch
import torch.nn as nn


class PixelNorm(nn.Module):
    """Normalizes feature vectors along channel dimension to unit norm."""

    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)


class MappingNetwork(nn.Module):
    """
    4-Layer MLP Mapping Network (z -> w).
    Normalizes z using PixelNorm, then passes it through 4 Linear layers
    with LeakyReLU activations.
    """

    def __init__(self, z_dim: int = 128, w_dim: int = 128, num_layers: int = 4):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim

        layers = [PixelNorm()]
        for i in range(num_layers):
            in_dim = z_dim if i == 0 else w_dim
            layers.append(nn.Linear(in_dim, w_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
