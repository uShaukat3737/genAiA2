import torch
import torch.nn as nn
from typing import Tuple, Dict, Any


class ConvDAE(nn.Module):
    """
    Convolutional Denoising Autoencoder (ConvDAE)
    
    Architecture Overview:
    ----------------------
    The network compresses a noisy RGB image x_noisy in [0, 1]^(3 x H x W) through
    a series of strided convolutional feature-extraction blocks into a compact
    latent bottleneck representation z in R^(bottleneck_channels x H/4 x W/4).
    The decoder then reconstructs the clean RGB image x_clean via transposed
    convolutions and a Sigmoid output activation.
    
    1. Encoder:
       - Block 1: Conv2d(3, base_channels, 3x3) -> BN -> LeakyReLU(0.2) -> Conv2d(base_channels, base_channels, stride=2) -> BN -> LeakyReLU(0.2)
       - Block 2: Conv2d(base_channels, base_channels*2, 3x3) -> BN -> LeakyReLU(0.2) -> Conv2d(base_channels*2, base_channels*2, stride=2) -> BN -> LeakyReLU(0.2)
       - Bottleneck Projection: Conv2d(base_channels*2, bottleneck_channels, 3x3) -> BN -> LeakyReLU(0.2)
       
    2. Bottleneck:
       - Latent feature map of shape [B, bottleneck_channels, H/4, W/4].
       - Provides spatial compression (downsampled by 4x) and channel throttling.
       
    3. Decoder:
       - Bottleneck Expansion: Conv2d(bottleneck_channels, base_channels*2, 3x3) -> BN -> LeakyReLU(0.2)
       - Upsample Block 1: ConvTranspose2d(base_channels*2, base_channels, 4x4, stride=2) -> BN -> LeakyReLU(0.2) -> Conv2d(base_channels, base_channels) -> BN -> LeakyReLU(0.2)
       - Upsample Block 2: ConvTranspose2d(base_channels, base_channels, 4x4, stride=2) -> BN -> LeakyReLU(0.2) -> Conv2d(base_channels, base_channels) -> BN -> LeakyReLU(0.2)
       - Output Projection: Conv2d(base_channels, 3, 3x3) -> Sigmoid()
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 32,
        bottleneck_channels: int = 32
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.bottleneck_channels = bottleneck_channels
        
        # --- ENCODER ---
        self.encoder_block1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.encoder_block2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Bottleneck projection
        self.bottleneck_proj = nn.Sequential(
            nn.Conv2d(base_channels * 2, bottleneck_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # --- DECODER ---
        self.bottleneck_expand = nn.Sequential(
            nn.Conv2d(bottleneck_channels, base_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.decoder_block1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.decoder_block2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Output Reconstruction Layer
        self.output_conv = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid()  # Restricts values to strictly [0.0, 1.0]
        )
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder to get bottleneck representation."""
        h1 = self.encoder_block1(x)
        h2 = self.encoder_block2(h1)
        z = self.bottleneck_proj(h2)
        return z
        
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass through decoder to reconstruct clean image."""
        h = self.bottleneck_expand(z)
        h = self.decoder_block1(h)
        h = self.decoder_block2(h)
        out = self.output_conv(h)
        return out
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full autoencoder forward pass: x_noisy -> latent -> x_denoised."""
        z = self.encode(x)
        out = self.decode(z)
        return out

    def get_latent_dimension(self, input_shape: Tuple[int, int] = (128, 128)) -> Dict[str, Any]:
        """Calculates total latent units and compression ratio for a given input resolution."""
        H, W = input_shape
        latent_h = H // 4
        latent_w = W // 4
        latent_elements = self.bottleneck_channels * latent_h * latent_w
        input_elements = self.in_channels * H * W
        compression_ratio = input_elements / latent_elements
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "bottleneck_channels": self.bottleneck_channels,
            "latent_shape": (self.bottleneck_channels, latent_h, latent_w),
            "latent_elements": latent_elements,
            "input_elements": input_elements,
            "compression_ratio": compression_ratio,
            "total_params": total_params,
            "trainable_params": trainable_params
        }


def build_dae_model(
    in_channels: int = 3,
    out_channels: int = 3,
    base_channels: int = 32,
    bottleneck_channels: int = 32
) -> ConvDAE:
    """Builder function for ConvDAE."""
    return ConvDAE(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
        bottleneck_channels=bottleneck_channels
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ConvDAE Model Summary")
    parser.add_argument("--bottleneck", type=int, default=32, help="Bottleneck channel dimension")
    args = parser.parse_args()
    
    model = build_dae_model(bottleneck_channels=args.bottleneck)
    sample_input = torch.randn(2, 3, 128, 128)
    sample_output = model(sample_input)
    
    info = model.get_latent_dimension((128, 128))
    print(f"Model instantiated successfully!")
    print(f"Input shape:  {sample_input.shape}")
    print(f"Latent shape: {info['latent_shape']} ({info['latent_elements']} elements)")
    print(f"Output shape: {sample_output.shape}")
    print(f"Compression ratio: {info['compression_ratio']:.2f}x")
    print(f"Total parameters: {info['total_params']:,}")
