import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Union


def compute_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Computes Mean Squared Error (MSE) between two image tensors in [0, 1].
    """
    return F.mse_loss(pred, target, reduction='mean').item()


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0, eps: float = 1e-10) -> float:
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).
    
    Formula:
        PSNR = 10 * log10(max_val^2 / (MSE + eps))
    
    Args:
        pred: Reconstructed image tensor.
        target: Reference clean ground truth image tensor.
        max_val: Dynamic range of pixel values (1.0 for normalized tensors).
        eps: Small epsilon to prevent division by zero.
    """
    mse = F.mse_loss(pred, target, reduction='mean').item()
    if mse < eps:
        return 100.0  # Perfect reconstruction ceiling
    return float(10.0 * math.log10((max_val ** 2) / mse))


def _gaussian_kernel(window_size: int, sigma: float) -> torch.Tensor:
    """Generates 1D Gaussian kernel."""
    gauss = torch.tensor([
        math.exp(-((x - window_size // 2) ** 2) / float(2 * sigma ** 2))
        for x in range(window_size)
    ])
    return gauss / gauss.sum()


def _create_window(window_size: int, channel: int, sigma: float = 1.5) -> torch.Tensor:
    """Generates 2D Gaussian window for depthwise convolution."""
    _1d_window = _gaussian_kernel(window_size, sigma).unsqueeze(1)
    _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2d_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim_tensor(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    size_average: bool = True,
    val_range: float = 1.0
) -> torch.Tensor:
    """
    Differentiable Structural Similarity Index (SSIM) on PyTorch tensors.
    
    Args:
        img1, img2: Tensors of shape (B, C, H, W) normalized to [0, val_range].
        window_size: Spatial size of Gaussian sliding window.
        sigma: Standard deviation of Gaussian window (default: 1.5).
        size_average: If True, returns mean scalar; otherwise returns per-sample tensor.
        val_range: Dynamic range of input images (default: 1.0).
    """
    if img1.ndim == 3:
        img1 = img1.unsqueeze(0)
    if img2.ndim == 3:
        img2 = img2.unsqueeze(0)
        
    B, C, H, W = img1.shape
    window = _create_window(window_size, C, sigma).to(img1.device).type_as(img1)
    
    # Constants from Wang et al. (2004)
    C1 = (0.01 * val_range) ** 2
    C2 = (0.03 * val_range) ** 2
    
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=C)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=C)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=C) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(dim=[1, 2, 3])


def compute_ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> float:
    """
    Computes Structural Similarity Index (SSIM) returning standard float.
    """
    return ssim_tensor(pred, target, window_size=window_size).item()


class SSIMLoss(nn.Module):
    """SSIM loss module (1.0 - SSIM) for training models."""
    def __init__(self, window_size: int = 11, size_average: bool = True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        return 1.0 - ssim_tensor(img1, img2, window_size=self.window_size, size_average=self.size_average)


class CombinedLoss(nn.Module):
    """Combined MSE + SSIM loss for sharper edge reconstruction."""
    def __init__(self, alpha: float = 0.85, window_size: int = 11):
        super().__init__()
        self.alpha = alpha
        self.ssim_loss = SSIMLoss(window_size=window_size)
        self.mse_loss = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (1.0 - self.alpha) * self.mse_loss(pred, target) + self.alpha * self.ssim_loss(pred, target)


def evaluate_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """
    Computes MSE, PSNR, and SSIM between reconstructed and ground truth tensors.
    
    Returns:
        Dictionary with keys 'mse', 'psnr', 'ssim'.
    """
    mse_val = compute_mse(pred, target)
    psnr_val = compute_psnr(pred, target)
    ssim_val = compute_ssim(pred, target)
    return {
        "mse": mse_val,
        "psnr": psnr_val,
        "ssim": ssim_val
    }


if __name__ == "__main__":
    x = torch.rand(4, 3, 128, 128)
    # Add slight distortion
    y = torch.clamp(x + torch.randn_like(x) * 0.05, 0.0, 1.0)
    
    metrics = evaluate_metrics(y, x)
    print("Metrics on test tensors:")
    for k, v in metrics.items():
        print(f"  {k.upper()}: {v:.4f}")
