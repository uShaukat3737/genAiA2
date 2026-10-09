import os
import sys
import json
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import get_dataloaders, download_and_extract_bsds500
from models import build_dae_model
from metrics import compute_mse, compute_psnr, compute_ssim, CombinedLoss, SSIMLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Train Convolutional Denoising Autoencoder on BSDS500")
    # Dataset arguments
    parser.add_argument("--data_dir", type=str, default="./data/BSDS500", help="Directory for BSDS500 dataset")
    parser.add_argument("--download", action="store_true", help="Download BSDS500 if missing")
    parser.add_argument("--crop_size", type=int, default=128, help="Patch crop size for training")
    
    # Noise arguments
    parser.add_argument("--noise_type", type=str, default="gaussian", choices=["gaussian", "salt_and_pepper"],
                        help="Noise corruption type to inject")
    parser.add_argument("--noise_param", type=float, default=0.10,
                        help="Noise intensity: std for gaussian (e.g. 0.1, 0.2), corruption ratio for S&P (e.g. 0.05, 0.1)")
    
    # Model architecture arguments
    parser.add_argument("--base_channels", type=int, default=32, help="Base channel count for ConvDAE")
    parser.add_argument("--bottleneck", type=int, default=32, help="Bottleneck channel dimension (for bottleneck studies)")
    
    # Optimization arguments
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and evaluation")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for optimizer")
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "l1", "mse_ssim"],
                        help="Reconstruction loss function")
    
    # Runtime & checkpointing
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"],
                        help="Hardware accelerator to use")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save weights & logs")
    parser.add_argument("--exp_name", type=str, default=None, help="Custom experiment directory name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader worker count")
    
    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_arg)


def get_loss_fn(loss_name: str) -> nn.Module:
    if loss_name == "mse":
        return nn.MSELoss()
    elif loss_name == "l1":
        return nn.L1Loss()
    elif loss_name == "mse_ssim":
        return CombinedLoss(alpha=0.8)
    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total_samples = 0
    
    for noisy, clean in loader:
        noisy = noisy.to(device)
        clean = clean.to(device)
        
        optimizer.zero_grad()
        recon = model(noisy)
        loss = criterion(recon, clean)
        loss.backward()
        optimizer.step()
        
        batch_size = noisy.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size
        
    return running_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_psnr = 0.0
    running_ssim = 0.0
    total_samples = 0
    
    for noisy, clean in loader:
        noisy = noisy.to(device)
        clean = clean.to(device)
        
        recon = model(noisy)
        loss = criterion(recon, clean)
        
        batch_size = noisy.size(0)
        running_loss += loss.item() * batch_size
        
        # Batch metrics
        for i in range(batch_size):
            psnr_val = compute_psnr(recon[i], clean[i])
            ssim_val = compute_ssim(recon[i].unsqueeze(0), clean[i].unsqueeze(0))
            running_psnr += psnr_val
            running_ssim += ssim_val
            
        total_samples += batch_size
        
    avg_loss = running_loss / max(total_samples, 1)
    avg_psnr = running_psnr / max(total_samples, 1)
    avg_ssim = running_ssim / max(total_samples, 1)
    
    return avg_loss, avg_psnr, avg_ssim


def plot_curves(history: dict, save_path: str):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    
    # Loss plot
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="royalblue", lw=2)
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", color="crimson", lw=2, linestyle="--")
    axes[0].set_title("Reconstruction Loss", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle=":", alpha=0.6)
    
    # PSNR plot
    axes[1].plot(epochs, history["val_psnr"], label="Val PSNR", color="forestgreen", lw=2)
    axes[1].set_title("Validation PSNR (dB)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("PSNR (dB)")
    axes[1].legend()
    axes[1].grid(True, linestyle=":", alpha=0.6)
    
    # SSIM plot
    axes[2].plot(epochs, history["val_ssim"], label="Val SSIM", color="darkorange", lw=2)
    axes[2].set_title("Validation SSIM", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("SSIM")
    axes[2].legend()
    axes[2].grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def train(args):
    set_seed(args.seed)
    device = select_device(args.device)
    print(f"===========================================================")
    print(f" Starting Denoising Autoencoder Training")
    print(f" Device:         {device}")
    print(f" Noise Type:     {args.noise_type} (param={args.noise_param})")
    print(f" Bottleneck Ch:  {args.bottleneck}")
    print(f" Loss Function:  {args.loss}")
    print(f" Epochs:         {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr}")
    print(f"===========================================================")
    
    # Experiment directory
    if args.exp_name is None:
        noise_str = f"{args.noise_type}_p{args.noise_param}"
        args.exp_name = f"dae_{noise_str}_btn{args.bottleneck}_{args.loss}"
    exp_dir = Path(args.save_dir) / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Check data
    if args.download or not Path(args.data_dir).exists():
        download_and_extract_bsds500(args.data_dir)
        
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        crop_size=args.crop_size,
        noise_type=args.noise_type,
        noise_param=args.noise_param,
        num_workers=args.num_workers,
        seed=args.seed
    )
    
    # Model
    model = build_dae_model(
        in_channels=3,
        out_channels=3,
        base_channels=args.base_channels,
        bottleneck_channels=args.bottleneck
    ).to(device)
    
    latent_info = model.get_latent_dimension((args.crop_size, args.crop_size))
    print(f"[Model] Latent shape: {latent_info['latent_shape']}, Parameters: {latent_info['total_params']:,}")
    
    criterion = get_loss_fn(args.loss).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_psnr": [],
        "val_ssim": [],
        "lr": []
    }
    
    best_val_psnr = -float("inf")
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_psnr, val_ssim = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step()
        
        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_psnr"].append(val_psnr)
        history["val_ssim"].append(val_ssim)
        history["lr"].append(current_lr)
        
        epoch_time = time.time() - epoch_start
        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Train Loss: {train_loss:.5f} | "
            f"Val Loss: {val_loss:.5f} | "
            f"Val PSNR: {val_psnr:.2f} dB | "
            f"Val SSIM: {val_ssim:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )
        
        # Save best model checkpoint
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            best_ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
                "val_psnr": val_psnr,
                "val_ssim": val_ssim,
                "val_loss": val_loss
            }
            torch.save(best_ckpt, exp_dir / "best_model.pth")
            
        # Save latest model
        latest_ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "history": history
        }
        torch.save(latest_ckpt, exp_dir / "latest_model.pth")
        
        # Periodically save training history and curves
        with open(exp_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        plot_curves(history, str(exp_dir / "training_curves.png"))
        
    total_time = time.time() - start_time
    print("-----------------------------------------------------------")
    print(f"Training completed in {total_time / 60:.2f} minutes.")
    print(f"Best Validation PSNR: {best_val_psnr:.2f} dB")
    print(f"Saved artifacts to:   {exp_dir}")
    print("-----------------------------------------------------------")
    
    return exp_dir / "best_model.pth"


if __name__ == "__main__":
    cli_args = parse_args()
    train(cli_args)
