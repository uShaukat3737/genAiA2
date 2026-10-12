# Experimental Study: Image Denoising using Denoising Autoencoder

## Overview

This report summarizes the experimental findings on the Berkeley Segmentation Dataset (BSDS500) evaluating convolutional denoising autoencoders under varying noise intensities and bottleneck capacities.

## 1. Study 1: Reconstruction Performance vs Noise Levels

| Noise Type | Intensity | Input PSNR (dB) | Input SSIM | Output PSNR (dB) | Output SSIM | Δ PSNR (dB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| gaussian | 0.05 | 26.19 | 0.7115 | **27.12** | **0.8350** | +0.92 |
| gaussian | 0.1 | 20.35 | 0.4834 | **26.08** | **0.7875** | +5.73 |
| gaussian | 0.2 | 14.83 | 0.2733 | **24.53** | **0.7069** | +9.70 |
| gaussian | 0.3 | 11.97 | 0.1814 | **23.37** | **0.6501** | +11.39 |
| salt_and_pepper | 0.02 | 22.00 | 0.7278 | **26.70** | **0.8306** | +4.70 |
| salt_and_pepper | 0.05 | 18.02 | 0.5084 | **26.41** | **0.8170** | +8.38 |
| salt_and_pepper | 0.1 | 15.00 | 0.3366 | **25.95** | **0.7982** | +10.95 |
| salt_and_pepper | 0.2 | 11.99 | 0.2017 | **25.27** | **0.7665** | +13.28 |

**Observations:** _write these from the table above._

## 2. Study 2: Reconstruction Performance vs Bottleneck Capacity

| Bottleneck Channels | Parameters | Compression Ratio | Output PSNR (dB) | Output SSIM | Δ PSNR (dB) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **8** | 143,827 | 6.00x | **24.73** | **0.7221** | +7.67 |
| **16** | 153,059 | 3.00x | **25.03** | **0.7374** | +7.98 |
| **32** | 171,523 | 1.50x | **25.27** | **0.7485** | +8.21 |
| **64** | 208,451 | 0.75x | **25.54** | **0.7563** | +8.48 |
| **128** | 282,307 | 0.38x | **25.54** | **0.7580** | +8.49 |

**Observations:** _write these from the table above._
