import os
import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from dataset import BSDS500Dataset
from models import build_dae_model
from metrics import compute_mse, compute_psnr, compute_ssim


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Denoising Autoencoder on BSDS500 Test Split")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model .pth checkpoint")
    parser.add_argument("--data_dir", type=str, default="./data/BSDS500", help="Directory of BSDS500 dataset")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                        help="Dataset split to evaluate")
    parser.add_argument("--noise_type", type=str, default=None,
                        help="Override noise type (default: from checkpoint args)")
    parser.add_argument("--noise_param", type=float, default=None,
                        help="Override noise parameter (default: from checkpoint args)")
    parser.add_argument("--crop_size", type=int, default=128, help="Crop dimension for evaluation")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for test evaluation")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of visualization samples to render")
    parser.add_argument("--save_dir", type=str, default="./results", help="Directory to save evaluation artifacts")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--seed", type=int, default=123, help="Seed for reproducible test noise")
    return parser.parse_args()


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_arg)


@torch.no_grad()
def evaluate_dataset(model, loader, device):
    """
    Evaluates the full dataset split, computing average MSE, PSNR, and SSIM
    both before denoising (noisy input vs clean) and after denoising (model output vs clean).
    """
    model.eval()
    
    in_mse_list, in_psnr_list, in_ssim_list = [], [], []
    out_mse_list, out_psnr_list, out_ssim_list = [], [], []
    
    for noisy, clean in loader:
        noisy = noisy.to(device)
        clean = clean.to(device)
        recon = model(noisy)
        
        batch_size = noisy.size(0)
        for i in range(batch_size):
            # Input noisy vs clean
            in_mse = compute_mse(noisy[i], clean[i])
            in_psnr = compute_psnr(noisy[i], clean[i])
            in_ssim = compute_ssim(noisy[i].unsqueeze(0), clean[i].unsqueeze(0))
            in_mse_list.append(in_mse)
            in_psnr_list.append(in_psnr)
            in_ssim_list.append(in_ssim)
            
            # Model reconstructed vs clean
            out_mse = compute_mse(recon[i], clean[i])
            out_psnr = compute_psnr(recon[i], clean[i])
            out_ssim = compute_ssim(recon[i].unsqueeze(0), clean[i].unsqueeze(0))
            out_mse_list.append(out_mse)
            out_psnr_list.append(out_psnr)
            out_ssim_list.append(out_ssim)
            
    metrics = {
        "input_mse": float(np.mean(in_mse_list)),
        "input_psnr": float(np.mean(in_psnr_list)),
        "input_ssim": float(np.mean(in_ssim_list)),
        "output_mse": float(np.mean(out_mse_list)),
        "output_psnr": float(np.mean(out_psnr_list)),
        "output_ssim": float(np.mean(out_ssim_list)),
        "delta_psnr": float(np.mean(out_psnr_list) - np.mean(in_psnr_list)),
        "delta_ssim": float(np.mean(out_ssim_list) - np.mean(in_ssim_list)),
        "total_evaluated_samples": len(out_mse_list)
    }
    return metrics


def visualize_and_save(model, dataset, device, num_samples: int, save_path: str):
    """
    Visualizes side-by-side triplets:
    Row 1: Original Clean Image
    Row 2: Corrupted Noisy Image (with input PSNR)
    Row 3: Reconstructed Denoised Image (with output PSNR and delta)
    """
    model.eval()
    num_samples = min(num_samples, len(dataset))
    
    fig, axes = plt.subplots(3, num_samples, figsize=(3.6 * num_samples, 10.5))
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=1)
        
    for col in range(num_samples):
        noisy_t, clean_t = dataset[col]
        with torch.no_grad():
            noisy_in = noisy_t.unsqueeze(0).to(device)
            recon_t = model(noisy_in).squeeze(0).cpu()
            
        clean_np = clean_t.permute(1, 2, 0).clamp(0, 1).numpy()
        noisy_np = noisy_t.permute(1, 2, 0).clamp(0, 1).numpy()
        recon_np = recon_t.permute(1, 2, 0).clamp(0, 1).numpy()
        
        in_psnr = compute_psnr(noisy_t, clean_t)
        in_ssim = compute_ssim(noisy_t.unsqueeze(0), clean_t.unsqueeze(0))
        
        out_psnr = compute_psnr(recon_t, clean_t)
        out_ssim = compute_ssim(recon_t.unsqueeze(0), clean_t.unsqueeze(0))
        delta_psnr = out_psnr - in_psnr
        
        # Row 0: Original Clean
        axes[0, col].imshow(clean_np)
        axes[0, col].set_title(f"Clean Ground Truth\nSample #{col+1}", fontsize=11, fontweight="bold")
        axes[0, col].axis("off")
        
        # Row 1: Noisy Input
        axes[1, col].imshow(noisy_np)
        axes[1, col].set_title(f"Noisy Input\nPSNR: {in_psnr:.2f} dB | SSIM: {in_ssim:.3f}", fontsize=10, color="crimson")
        axes[1, col].axis("off")
        
        # Row 2: Reconstructed Output
        axes[2, col].imshow(recon_np)
        sign_psnr = f"+{delta_psnr:.2f}" if delta_psnr >= 0 else f"{delta_psnr:.2f}"
        axes[2, col].set_title(f"Reconstructed (DAE)\nPSNR: {out_psnr:.2f} dB ({sign_psnr} dB)\nSSIM: {out_ssim:.3f}",
                               fontsize=10, color="forestgreen", fontweight="bold")
        axes[2, col].axis("off")
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Evaluation] Visualization grid saved to: {save_path}")


def main():
    args = parse_args()
    device = select_device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")
        
    checkpoint = torch.load(ckpt_path, map_location=device)
    saved_args = checkpoint.get("args", {})
    
    # Resolve noise parameters
    noise_type = args.noise_type or saved_args.get("noise_type", "gaussian")
    noise_param = args.noise_param if args.noise_param is not None else saved_args.get("noise_param", 0.1)
    bottleneck = saved_args.get("bottleneck", 32)
    base_channels = saved_args.get("base_channels", 32)
    
    print(f"===========================================================")
    print(f" Evaluating Denoising Autoencoder on BSDS500 [{args.split.upper()}]")
    print(f" Checkpoint:      {ckpt_path.name}")
    print(f" Architecture:    Bottleneck={bottleneck}, BaseChannels={base_channels}")
    print(f" Noise Type:      {noise_type} (param={noise_param})")
    print(f" Device:          {device}")
    print(f"===========================================================")
    
    # Build Model & Load Weights
    model = build_dae_model(
        in_channels=3,
        out_channels=3,
        base_channels=base_channels,
        bottleneck_channels=bottleneck
    ).to(device)
    
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    
    # Dataset and DataLoader
    dataset = BSDS500Dataset(
        data_dir=args.data_dir,
        split=args.split,
        crop_size=args.crop_size,
        noise_type=noise_type,
        noise_param=noise_param,
        is_train=False,
        seed=args.seed
    )
    
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Compute full quantitative metrics
    metrics = evaluate_dataset(model, loader, device)
    metrics["checkpoint"] = str(ckpt_path)
    metrics["split"] = args.split
    metrics["noise_type"] = noise_type
    metrics["noise_param"] = noise_param
    metrics["bottleneck"] = bottleneck
    
    print("\n------------------- Evaluation Results --------------------")
    print(f" Total Evaluated Images:  {metrics['total_evaluated_samples']}")
    print(f" Input (Noisy) MSE:       {metrics['input_mse']:.5f}")
    print(f" Input (Noisy) PSNR:      {metrics['input_psnr']:.2f} dB")
    print(f" Input (Noisy) SSIM:      {metrics['input_ssim']:.4f}")
    print(f" Output (Denoised) MSE:   {metrics['output_mse']:.5f}")
    print(f" Output (Denoised) PSNR:  {metrics['output_psnr']:.2f} dB")
    print(f" Output (Denoised) SSIM:  {metrics['output_ssim']:.4f}")
    print(f" PSNR Improvement:        {metrics['delta_psnr']:+.2f} dB")
    print(f" SSIM Improvement:        {metrics['delta_ssim']:+.4f}")
    print("-----------------------------------------------------------")
    
    # Save Metrics JSON
    metrics_file = save_dir / f"metrics_{ckpt_path.stem}_{noise_type}_{noise_param}.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Evaluation] Metrics saved to: {metrics_file}")
    
    # Generate & Save Visualization
    vis_path = save_dir / f"visualization_{ckpt_path.stem}_{noise_type}_{noise_param}.png"
    visualize_and_save(model, dataset, device, args.num_samples, str(vis_path))


if __name__ == "__main__":
    main()
