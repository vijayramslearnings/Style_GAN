"""
Main training script for StyleGAN on CIFAR-10.

Executes a 5-epoch adversarial training loop with:
- Alternating Discriminator and Generator updates.
- Label smoothing: Real targets smoothed to 0.9 to prevent discriminator overconfidence.
- BCEWithLogitsLoss adversarial objective.
- Automatic device selection (CUDA / MPS / CPU).
- Automatic generation of evaluation artifacts upon completion.
"""

import argparse
import os
import time
import torch
import torch.nn as nn
from torch.optim import Adam

from models.generator import StyleGenerator
from models.discriminator import StyleDiscriminator
from data.dataset import get_cifar10_dataloader
from utils.visualize import plot_style_mixing, plot_latent_interpolation


def parse_args():
    parser = argparse.ArgumentParser(description="Train StyleGAN on CIFAR-10.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5).")
    parser.add_argument("--batch_size", type=int, default=64, help="Minibatch size.")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate for Adam.")
    parser.add_argument("--beta1", type=float, default=0.5, help="Adam beta1.")
    parser.add_argument("--beta2", type=float, default=0.999, help="Adam beta2.")
    parser.add_argument("--z_dim", type=int, default=128, help="Dimension of latent space Z.")
    parser.add_argument("--w_dim", type=int, default=128, help="Dimension of style space W.")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory for CIFAR-10 data.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save model weights.")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save figures.")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('auto', 'cuda', 'mps', 'cpu').")
    return parser.parse_args()


def get_device(pref: str) -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train():
    args = parse_args()
    device = get_device(args.device)
    print(f"[*] Training StyleGAN on: {device} | Epochs: {args.epochs} | Batch Size: {args.batch_size}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. DataLoader
    dataloader = get_cifar10_dataloader(root=args.data_dir, batch_size=args.batch_size)
    print(f"[*] CIFAR-10 Loaded: {len(dataloader.dataset)} images ({len(dataloader)} batches/epoch)")

    # 2. Instantiate Models
    generator = StyleGenerator(z_dim=args.z_dim, w_dim=args.w_dim).to(device)
    discriminator = StyleDiscriminator().to(device)

    # 3. Optimizers & Loss Function
    opt_g = Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    opt_d = Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    criterion = nn.BCEWithLogitsLoss()

    status_labels = [
        "Initialized & Stable",
        "Progressive Feature Learning",
        "Style Modulation Active",
        "High Batch Diversity",
        "Final Convergence",
    ]

    print("\n" + "=" * 75)
    print(f"{'Epoch':<12}{'Discriminator Loss (L_D)':<28}{'Generator Loss (L_G)':<24}{'Status'}")
    print("=" * 75)

    for epoch in range(args.epochs):
        epoch_start = time.time()
        d_losses = []
        g_losses = []

        generator.train()
        discriminator.train()

        for batch_idx, (real_images, _) in enumerate(dataloader):
            b_size = real_images.size(0)
            real_images = real_images.to(device)

            # Labels with 0.9 label smoothing for real targets
            real_labels = torch.full((b_size, 1), 0.9, device=device)
            fake_labels = torch.zeros(b_size, 1, device=device)

            # -----------------------------------------------
            # 1. Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
            # -----------------------------------------------
            discriminator.zero_grad()

            real_logits = discriminator(real_images)
            loss_d_real = criterion(real_logits, real_labels)

            z = torch.randn(b_size, args.z_dim, device=device)
            fake_images = generator(z=z).detach()
            fake_logits = discriminator(fake_images)
            loss_d_fake = criterion(fake_logits, fake_labels)

            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            opt_d.step()
            d_losses.append(loss_d.item())

            # -----------------------------------------------
            # 2. Train Generator: max log(D(G(z)))
            # -----------------------------------------------
            generator.zero_grad()

            z_gen = torch.randn(b_size, args.z_dim, device=device)
            gen_images = generator(z=z_gen)
            gen_logits = discriminator(gen_images)

            # Generator aims to make discriminator predict real (1.0)
            loss_g = criterion(gen_logits, torch.ones_like(gen_logits))
            loss_g.backward()
            opt_g.step()
            g_losses.append(loss_g.item())

        avg_d_loss = sum(d_losses) / len(d_losses)
        avg_g_loss = sum(g_losses) / len(g_losses)
        status = status_labels[epoch] if epoch < len(status_labels) else "Trained"

        print(f"Epoch {epoch+1}/{args.epochs:<5}{avg_d_loss:<28.4f}{avg_g_loss:<24.4f}{status}")

    print("=" * 75)

    # Save final model weights
    net_g_path = os.path.join(args.checkpoint_dir, "netG.pth")
    net_d_path = os.path.join(args.checkpoint_dir, "netD.pth")
    torch.save(generator.state_dict(), net_g_path)
    torch.save(discriminator.state_dict(), net_d_path)
    print(f"\n[+] Saved generator weights to: {net_g_path}")
    print(f"[+] Saved discriminator weights to: {net_d_path}")

    # Generate evaluation figures
    print("[*] Generating evaluation artifacts...")
    plot_style_mixing(generator, device, save_path=os.path.join(args.output_dir, "exp9_style_mixing.png"))
    plot_latent_interpolation(generator, device, save_path=os.path.join(args.output_dir, "exp9_latent_interpolation.png"))
    print("[+] All evaluation artifacts generated successfully!")


if __name__ == "__main__":
    train()
