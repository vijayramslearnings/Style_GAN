"""
StyleGenerator Synthesis Network for StyleGAN.

Features:
- Learned constant input base tensor C in R^{1 x 128 x 4 x 4} initialized to ones.
- Progressive upsampling: 4x4 -> 8x8 -> 16x16 -> 32x32.
- StyleConvBlock with AdaIN and per-pixel stochastic noise injection.
- Style mixing regularization (w_early, w_late).
- Final Tanh projection to 3-channel RGB image.
"""

from typing import List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mapping import MappingNetwork
from models.blocks import StyleConvBlock, NoiseInjection
from models.adain import AdaIN


class StyleGenerator(nn.Module):
    """
    Complete Style-Based Generator.
    Combines the 4-layer Mapping Network and the 4-stage Synthesis Network.
    """

    def __init__(
        self,
        z_dim: int = 128,
        w_dim: int = 128,
        base_channels: int = 128,
        out_channels: int = 3,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.base_channels = base_channels

        # 4-layer Mapping Network
        self.mapping = MappingNetwork(z_dim=z_dim, w_dim=w_dim, num_layers=4)

        # Learned constant base tensor initialized to ones
        self.const = nn.Parameter(torch.ones(1, base_channels, 4, 4))

        # 4x4 stage: Initial noise + AdaIN + StyleConv
        self.init_noise = NoiseInjection(base_channels)
        self.init_act = nn.LeakyReLU(0.2, inplace=True)
        self.init_adain = AdaIN(base_channels, w_dim)
        self.conv_4x4 = StyleConvBlock(base_channels, base_channels, style_dim=w_dim)

        # 8x8 stage
        self.conv_8x8 = StyleConvBlock(base_channels, base_channels, style_dim=w_dim)

        # 16x16 stage
        self.conv_16x16 = StyleConvBlock(base_channels, base_channels // 2, style_dim=w_dim)

        # 32x32 stage
        self.conv_32x32 = StyleConvBlock(base_channels // 2, base_channels // 4, style_dim=w_dim)

        # ToRGB output projection
        self.to_rgb = nn.Conv2d(base_channels // 4, out_channels, kernel_size=1)

    def forward(
        self,
        z: Optional[torch.Tensor] = None,
        w: Optional[torch.Tensor] = None,
        w_late: Optional[torch.Tensor] = None,
        mix_threshold: int = 2,  # 0..3: switch from w to w_late at this stage
    ) -> torch.Tensor:
        """
        Args:
            z: Latent vector in Z-space of shape (B, z_dim).
            w: Latent vector in W-space of shape (B, w_dim). If provided, z is ignored.
            w_late: Optional second style vector for style mixing.
            mix_threshold: Stage index where style switches from w to w_late.
        Returns:
            Generated RGB image tensor of shape (B, 3, 32, 32) in range [-1, 1].
        """
        if w is None:
            assert z is not None, "Either z or w must be provided."
            w = self.mapping(z)

        # If style mixing is requested, select style per stage
        w0 = w
        w1 = w_late if (w_late is not None and mix_threshold <= 1) else w
        w2 = w_late if (w_late is not None and mix_threshold <= 2) else w
        w3 = w_late if (w_late is not None and mix_threshold <= 3) else w

        batch_size = w.shape[0]

        # 1. 4x4 resolution base: Learned constant C
        x = self.const.repeat(batch_size, 1, 1, 1)
        x = self.init_noise(x)
        x = self.init_act(x)
        x = self.init_adain(x, w0)
        x = self.conv_4x4(x, w0)

        # 2. 8x8 resolution stage (Upsample 2x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.conv_8x8(x, w1)

        # 3. 16x16 resolution stage (Upsample 2x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.conv_16x16(x, w2)

        # 4. 32x32 resolution stage (Upsample 2x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.conv_32x32(x, w3)

        # Projection to RGB mapped to [-1, 1] via Tanh
        rgb = torch.tanh(self.to_rgb(x))
        return rgb
