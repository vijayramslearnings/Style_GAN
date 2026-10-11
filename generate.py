"""
Inference & Figure Generation Script for StyleGAN.

Loads a saved generator checkpoint and generates:
1. 2x2 Clean Style Mixing Grid Matrix (outputs/exp9_style_mixing.png)
2. 4-Step Latent Space LERP Interpolation Sequence (outputs/exp9_latent_interpolation.png)
"""

import argparse
import os
import torch

from models.generator import StyleGenerator
from utils.visualize import plot_style_mixing, plot_latent_interpolation


def parse_args():
    parser = argparse.ArgumentParser(description="Generate StyleGAN evaluation artifacts.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/netG.pth",
                        help="Path to trained generator checkpoint.")
    parser.add_argument("--out_dir", type=str, default="outputs",
                        help="Directory to save generated artifacts.")
    parser.add_argument("--z_dim", type=int, default=128, help="Dimension of latent space Z.")
    parser.add_argument("--w_dim", type=int, default=128, help="Dimension of style space W.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use ('auto', 'cuda', 'mps', 'cpu').")
    return parser.parse_args()


def get_device(pref: str) -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"[*] Running inference on device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)

    generator = StyleGenerator(z_dim=args.z_dim, w_dim=args.w_dim).to(device)

    if os.path.isfile(args.checkpoint):
        print(f"[*] Loading weights from: {args.checkpoint}")
        generator.load_state_dict(torch.load(args.checkpoint, map_location=device))
    else:
        print(f"[!] Checkpoint '{args.checkpoint}' not found. Generating sample plots with initialized weights.")

    generator.eval()

    # 1. 2x2 Style Mixing Grid Matrix
    mix_path = os.path.join(args.out_dir, "exp9_style_mixing.png")
    plot_style_mixing(generator, device, save_path=mix_path, seed=args.seed)

    # 2. 4-Step Latent Vector LERP Interpolation Sequence
    interp_path = os.path.join(args.out_dir, "exp9_latent_interpolation.png")
    plot_latent_interpolation(generator, device, save_path=interp_path, steps=4, seed=args.seed)

    print(f"[+] All artifacts generated and saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
