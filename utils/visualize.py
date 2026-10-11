"""
Visualization utilities for StyleGAN.

Contains:
- plot_style_mixing: 2x2 Clean Style Mixing Grid Matrix Output.
- plot_latent_interpolation: 4-Step Latent Space LERP Interpolation Sequence.
"""

import os
from typing import Optional
import torch
import numpy as np
import matplotlib.pyplot as plt


def denormalize_img(tensor: torch.Tensor) -> np.ndarray:
    """
    Converts a PyTorch image tensor (C, H, W) in range [-1, 1]
    to a NumPy array (H, W, C) in range [0, 1].
    """
    img = tensor.detach().cpu().clamp(-1.0, 1.0)
    img = (img + 1.0) / 2.0
    img = img.permute(1, 2, 0).numpy()
    return img


@torch.no_grad()
def plot_style_mixing(
    generator: torch.nn.Module,
    device: torch.device,
    save_path: str = "outputs/exp9_style_mixing.png",
    mix_threshold: int = 2,
    seed: Optional[int] = 42,
) -> None:
    """
    Generates a 2x2 Clean Style Mixing Grid Matrix (Max 4 images):
    - Top row: Pure source styles (w1 and w2)
    - Bottom row: Cross-mixed generations (Style A + Style B and Style B + Style A)
    """
    generator.eval()
    if seed is not None:
        torch.manual_seed(seed)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    # Sample two source latents in Z-space
    z1 = torch.randn(1, generator.z_dim, device=device)
    z2 = torch.randn(1, generator.z_dim, device=device)

    # Map to disentangled style space W
    w1 = generator.mapping(z1)
    w2 = generator.mapping(z2)

    # 1. Pure Source A
    img_a = generator(w=w1)[0]
    # 2. Pure Source B
    img_b = generator(w=w2)[0]
    # 3. Mixed: Early layers from A, Late layers from B
    img_ab = generator(w=w1, w_late=w2, mix_threshold=mix_threshold)[0]
    # 4. Mixed: Early layers from B, Late layers from A
    img_ba = generator(w=w2, w_late=w1, mix_threshold=mix_threshold)[0]

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    fig.suptitle("StyleGAN Style Mixing (4 Sample Grid)", fontsize=16, fontweight="bold", y=0.98)

    # Top-Left: Source Style A
    axes[0, 0].imshow(denormalize_img(img_a))
    axes[0, 0].set_title("Source Style A (w1)", fontsize=13, fontweight="semibold", pad=8)
    axes[0, 0].axis("off")

    # Top-Right: Source Style B
    axes[0, 1].imshow(denormalize_img(img_b))
    axes[0, 1].set_title("Source Style B (w2)", fontsize=13, fontweight="semibold", pad=8)
    axes[0, 1].axis("off")

    # Bottom-Left: Mixed A + B
    axes[1, 0].imshow(denormalize_img(img_ab))
    axes[1, 0].set_title("Mixed: Style A + Style B", fontsize=13, fontweight="semibold", pad=8)
    axes[1, 0].axis("off")

    # Bottom-Right: Mixed B + A
    axes[1, 1].imshow(denormalize_img(img_ba))
    axes[1, 1].set_title("Mixed: Style B + Style A", fontsize=13, fontweight="semibold", pad=8)
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Style mixing matrix saved to: {save_path}")


@torch.no_grad()
def plot_latent_interpolation(
    generator: torch.nn.Module,
    device: torch.device,
    save_path: str = "outputs/exp9_latent_interpolation.png",
    steps: int = 4,
    seed: Optional[int] = 42,
) -> None:
    """
    Generates a 4-step Latent Space LERP Interpolation Sequence in W-space:
    alpha = [0.00, 0.33, 0.67, 1.00].
    """
    generator.eval()
    if seed is not None:
        torch.manual_seed(seed)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    # Sample two endpoint latents in Z-space and map to W
    z1 = torch.randn(1, generator.z_dim, device=device)
    z2 = torch.randn(1, generator.z_dim, device=device)
    w1 = generator.mapping(z1)
    w2 = generator.mapping(z2)

    alphas = np.linspace(0.0, 1.0, steps)

    fig, axes = plt.subplots(1, steps, figsize=(12, 3.2))
    fig.suptitle("Latent Space LERP Interpolation (4 Steps)", fontsize=15, fontweight="bold", y=1.05)

    for i, alpha in enumerate(alphas):
        # Linear interpolation (LERP) in W-space
        w_interp = (1.0 - alpha) * w1 + alpha * w2
        img = generator(w=w_interp)[0]

        axes[i].imshow(denormalize_img(img))
        axes[i].set_title(f"$\\alpha = {alpha:.2f}$", fontsize=13, fontweight="semibold", pad=8)
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Latent interpolation plot saved to: {save_path}")
