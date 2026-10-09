import os
import io
import sys
import glob
import zipfile
import urllib.request
import argparse
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


def add_gaussian_noise(tensor: torch.Tensor, std: float = 0.1, mean: float = 0.0) -> torch.Tensor:
    """
    Inject additive zero-mean Gaussian noise into image tensor.
    
    Args:
        tensor: Clean image tensor with values in range [0, 1]. Shape (C, H, W) or (B, C, H, W).
        std: Standard deviation of the Gaussian distribution (noise intensity).
        mean: Mean of the Gaussian distribution (default: 0.0).
        
    Returns:
        Noisy image tensor clamped to [0.0, 1.0].
    """
    noise = torch.randn_like(tensor) * std + mean
    noisy = tensor + noise
    return torch.clamp(noisy, 0.0, 1.0)


def add_salt_and_pepper_noise(tensor: torch.Tensor, amount: float = 0.05, salt_vs_pepper: float = 0.5) -> torch.Tensor:
    """
    Inject impulse Salt-and-Pepper noise into image tensor.
    
    Args:
        tensor: Clean image tensor with values in range [0, 1]. Shape (C, H, W) or (B, C, H, W).
        amount: Proportion of corrupted pixels (e.g. 0.05 = 5% of pixels corrupted).
        salt_vs_pepper: Ratio of salt (1.0) vs pepper (0.0). Default is 0.5 (equal split).
        
    Returns:
        Corrupted image tensor clamped to [0.0, 1.0].
    """
    noisy = tensor.clone()
    
    # Generate mask across spatial dimensions (or per-channel)
    # Applying across all channels at identical pixel location simulates sensor pixel defect
    if tensor.ndim == 3:
        mask_shape = (1, tensor.shape[1], tensor.shape[2])
    elif tensor.ndim == 4:
        mask_shape = (tensor.shape[0], 1, tensor.shape[2], tensor.shape[3])
    else:
        mask_shape = tensor.shape
        
    rand_matrix = torch.rand(mask_shape, device=tensor.device)
    
    # Salt noise (white pixels = 1.0)
    salt_thresh = amount * salt_vs_pepper
    salt_mask = rand_matrix < salt_thresh
    
    # Pepper noise (black pixels = 0.0)
    pepper_thresh = amount
    pepper_mask = (rand_matrix >= salt_thresh) & (rand_matrix < pepper_thresh)
    
    noisy = torch.where(salt_mask, torch.ones_like(noisy), noisy)
    noisy = torch.where(pepper_mask, torch.zeros_like(noisy), noisy)
    return torch.clamp(noisy, 0.0, 1.0)


def inject_noise(tensor: torch.Tensor, noise_type: str = "gaussian", noise_param: float = 0.1) -> torch.Tensor:
    """
    Unified noise injection helper.
    
    Args:
        tensor: Normalized image tensor in [0, 1].
        noise_type: 'gaussian' or 'salt_and_pepper' (or 'sp').
        noise_param: std for gaussian, corrupted fraction for salt_and_pepper.
    """
    noise_type = noise_type.lower()
    if noise_type in ["gaussian", "gauss"]:
        return add_gaussian_noise(tensor, std=noise_param)
    elif noise_type in ["salt_and_pepper", "sp", "salt_pepper"]:
        return add_salt_and_pepper_noise(tensor, amount=noise_param)
    elif noise_type in ["clean", "none"]:
        return tensor.clone()
    else:
        raise ValueError(f"Unknown noise_type '{noise_type}'. Supported: 'gaussian', 'salt_and_pepper'.")


def download_and_extract_bsds500(target_dir: str) -> str:
    """
    Downloads and extracts the Berkeley Segmentation Dataset (BSDS500) if not already present.
    Extracts the official train, val, and test image sets.
    """
    target_path = Path(target_dir).resolve()
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Check if images are already present
    train_dir = target_path / "train"
    val_dir = target_path / "val"
    test_dir = target_path / "test"
    
    if train_dir.exists() and val_dir.exists() and test_dir.exists():
        train_count = len(list(train_dir.glob("*.jpg")))
        val_count = len(list(val_dir.glob("*.jpg")))
        test_count = len(list(test_dir.glob("*.jpg")))
        if train_count > 0 and val_count > 0 and test_count > 0:
            print(f"[Dataset] Found existing BSDS500 images at: {target_path}")
            print(f"          Train: {train_count}, Val: {val_count}, Test: {test_count} images.")
            return str(target_path)
            
    # Also check if BSDS500 is nested in BSDS500/data/images
    nested_path = target_path / "BSDS500" / "data" / "images"
    if (nested_path / "train").exists():
        import shutil
        for split in ["train", "val", "test"]:
            src_split = nested_path / split
            dst_split = target_path / split
            if src_split.exists() and not dst_split.exists():
                shutil.copytree(src_split, dst_split)
        return str(target_path)
        
    import ssl
    print(f"[Dataset] Downloading BSDS500 archive from GitHub mirror...")
    zip_url = "https://github.com/BIDS/BSDS500/archive/refs/heads/master.zip"
    
    try:
        # Create SSL context that does not fail on macOS systems missing CA certificates
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        req = urllib.request.Request(
            zip_url,
            headers={"User-Agent": "Mozilla/5.0 (DAE-Denoising-Dataset-Downloader)"}
        )
        with urllib.request.urlopen(req, context=ssl_context) as response:
            zip_bytes = response.read()
        print(f"[Dataset] Download complete ({len(zip_bytes) / 1024 / 1024:.2f} MB). Extracting...")
        
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            # Locate all image entries inside the zip
            for member in z.namelist():
                for split in ["train", "val", "test"]:
                    prefix = f"BSDS500-master/BSDS500/data/images/{split}/"
                    if member.startswith(prefix) and member.endswith(".jpg"):
                        filename = os.path.basename(member)
                        dest_folder = target_path / split
                        dest_folder.mkdir(parents=True, exist_ok=True)
                        dest_file = dest_folder / filename
                        with z.open(member) as src, open(dest_file, "wb") as dst:
                            dst.write(src.read())
                            
        print(f"[Dataset] Extracted BSDS500 images to: {target_path}")
        train_count = len(list((target_path / "train").glob("*.jpg")))
        val_count = len(list((target_path / "val").glob("*.jpg")))
        test_count = len(list((target_path / "test").glob("*.jpg")))
        print(f"          Counts -> Train: {train_count}, Val: {val_count}, Test: {test_count}")
        return str(target_path)
        
    except Exception as e:
        print(f"[Dataset] urllib download failed ({e}), attempting git clone fallback...")
        import subprocess
        tmp_clone = target_path / "_git_bsds500"
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", "https://github.com/BIDS/BSDS500.git", str(tmp_clone)],
                check=True
            )
            import shutil
            for split in ["train", "val", "test"]:
                src = tmp_clone / "BSDS500" / "data" / "images" / split
                dst = target_path / split
                if src.exists():
                    dst.mkdir(parents=True, exist_ok=True)
                    for f in src.glob("*.jpg"):
                        shutil.copy(f, dst / f.name)
            shutil.rmtree(tmp_clone, ignore_errors=True)
            print(f"[Dataset] Git clone extraction succeeded at: {target_path}")
            return str(target_path)
        except Exception as git_err:
            print(f"[Dataset] Error downloading BSDS500 via git clone: {git_err}")
            raise e


class BSDS500Dataset(Dataset):
    """
    PyTorch Dataset for Berkeley Segmentation Dataset (BSDS500) paired with noisy input generation.
    Each sample returns a tuple of (noisy_image, clean_image).
    """
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        crop_size: int = 128,
        noise_type: str = "gaussian",
        noise_param: float = 0.1,
        is_train: bool = True,
        seed: int = None
    ):
        """
        Args:
            data_dir: Root directory containing train/val/test subdirectories.
            split: 'train', 'val', or 'test'.
            crop_size: Spatial dimension for square patch crop (default: 128).
            noise_type: 'gaussian' or 'salt_and_pepper'.
            noise_param: Noise level (std for gaussian, corrupted fraction for S&P).
            is_train: If True, uses random cropping & flips. Otherwise center crop/fixed crop.
            seed: Optional random seed for reproducible noise generation.
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.crop_size = crop_size
        self.noise_type = noise_type
        self.noise_param = noise_param
        self.is_train = is_train
        self.seed = seed
        
        split_dir = self.data_dir / split
        if not split_dir.exists():
            # Check nested
            nested = self.data_dir / "BSDS500" / "data" / "images" / split
            if nested.exists():
                split_dir = nested
                
        self.image_paths = sorted(glob.glob(str(split_dir / "*.jpg")) + glob.glob(str(split_dir / "*.png")))
        
        if len(self.image_paths) == 0:
            raise FileNotFoundError(
                f"No image files found in {split_dir}. "
                f"Please ensure BSDS500 dataset is downloaded by calling download_and_extract_bsds500."
            )
            
    def __len__(self) -> int:
        return len(self.image_paths)
        
    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        # Geometry transforms
        if self.is_train:
            # Data augmentation: Random Crop + Random Horizontal & Vertical Flips
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(self.crop_size, self.crop_size))
            image = TF.crop(image, i, j, h, w)
            if torch.rand(1).item() > 0.5:
                image = TF.hflip(image)
            if torch.rand(1).item() > 0.5:
                image = TF.vflip(image)
        else:
            # Validation / Testing: Center Crop for deterministic evaluation
            image = TF.center_crop(image, (self.crop_size, self.crop_size))
            
        # Convert to Tensor normalized to [0, 1]
        clean_tensor = TF.to_tensor(image)
        
        # Noise injection
        if self.seed is not None and not self.is_train:
            # Deterministic noise for val/test
            rng_state = torch.get_rng_state()
            torch.manual_seed(self.seed + idx)
            noisy_tensor = inject_noise(clean_tensor, self.noise_type, self.noise_param)
            torch.set_rng_state(rng_state)
        else:
            noisy_tensor = inject_noise(clean_tensor, self.noise_type, self.noise_param)
            
        return noisy_tensor, clean_tensor


def get_dataloaders(
    data_dir: str,
    batch_size: int = 16,
    crop_size: int = 128,
    noise_type: str = "gaussian",
    noise_param: float = 0.1,
    num_workers: int = 2,
    seed: int = 42
):
    """
    Constructs DataLoaders for train, validation, and test splits.
    """
    train_dataset = BSDS500Dataset(
        data_dir=data_dir,
        split="train",
        crop_size=crop_size,
        noise_type=noise_type,
        noise_param=noise_param,
        is_train=True
    )
    
    val_dataset = BSDS500Dataset(
        data_dir=data_dir,
        split="val",
        crop_size=crop_size,
        noise_type=noise_type,
        noise_param=noise_param,
        is_train=False,
        seed=seed
    )
    
    test_dataset = BSDS500Dataset(
        data_dir=data_dir,
        split="test",
        crop_size=crop_size,
        noise_type=noise_type,
        noise_param=noise_param,
        is_train=False,
        seed=seed + 1000
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BSDS500 Dataset Loader and Noise Injection CLI")
    parser.add_argument("--download", action="store_true", help="Download and extract BSDS500 dataset")
    parser.add_argument("--data_dir", type=str, default="./data/BSDS500", help="Directory for BSDS500 dataset")
    parser.add_argument("--noise_type", type=str, default="gaussian", choices=["gaussian", "salt_and_pepper"])
    parser.add_argument("--noise_param", type=float, default=0.1, help="Noise std (gaussian) or corrupt ratio (S&P)")
    parser.add_argument("--crop_size", type=int, default=128, help="Crop patch dimension")
    parser.add_argument("--save_sample", type=str, default=None, help="Save a sample visualization comparison to path")
    args = parser.parse_args()
    
    if args.download or not Path(args.data_dir).exists():
        print(f"Downloading BSDS500 to {args.data_dir}...")
        download_and_extract_bsds500(args.data_dir)
        
    print(f"Testing DataLoader on {args.data_dir}...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=4,
        crop_size=args.crop_size,
        noise_type=args.noise_type,
        noise_param=args.noise_param
    )
    
    noisy_b, clean_b = next(iter(train_loader))
    print(f"Loaded train batch successfully! Noisy shape: {noisy_b.shape}, Clean shape: {clean_b.shape}")
    print(f"Noisy min: {noisy_b.min():.3f}, max: {noisy_b.max():.3f}")
    print(f"Clean min: {clean_b.min():.3f}, max: {clean_b.max():.3f}")
    
    if args.save_sample:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
        
        # Test both noises on clean image
        clean_img = clean_b[0].permute(1, 2, 0).numpy()
        noisy_gauss = add_gaussian_noise(clean_b[0], std=0.15).permute(1, 2, 0).numpy()
        noisy_sp = add_salt_and_pepper_noise(clean_b[0], amount=0.10).permute(1, 2, 0).numpy()
        
        axes[0].imshow(clean_img)
        axes[0].set_title("Clean Image")
        axes[0].axis("off")
        
        axes[1].imshow(noisy_gauss)
        axes[1].set_title("Gaussian Noise (std=0.15)")
        axes[1].axis("off")
        
        axes[2].imshow(noisy_sp)
        axes[2].set_title("Salt & Pepper Noise (p=0.10)")
        axes[2].axis("off")
        
        plt.tight_layout()
        plt.savefig(args.save_sample, dpi=200)
        print(f"Saved sample comparison to {args.save_sample}")
