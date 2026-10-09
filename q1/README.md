# Question-1: Image Denoising using Denoising Autoencoder (DAE)

[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](colab_run.ipynb)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-BSDS500-blue.svg)](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/)

This repository implements a **Convolutional Denoising Autoencoder (ConvDAE)** for restoring clean images from corrupted inputs on the standard **Berkeley Segmentation Dataset and Benchmark (BSDS500)**. It is structured with command-line interface (CLI) flags for execution in **Google Colab** and local environments.

---

## Table of Contents
1. [Task Overview](#task-overview)
2. [Dataset Preparation & Noise Models](#dataset-preparation--noise-models)
3. [Autoencoder Architecture Design](#autoencoder-architecture-design)
4. [Training & Loss Formulations](#training--loss-formulations)
5. [Evaluation Metrics & Visualization](#evaluation-metrics--visualization)
6. [Experimental Study & Findings](#experimental-study--findings)
7. [Repository Structure](#repository-structure)
8. [Google Colab Quickstart](#google-colab-quickstart)
9. [CLI Usage Reference](#cli-usage-reference)

---

## Task Overview

Real-world imaging pipelines (medical imaging, surveillance, mobile photography, remote sensing) encounter corruptions due to physical sensor constraints, thermal noise, transmission dropouts, and low illumination. A **Denoising Autoencoder (DAE)** learns to map a noise-perturbed observation $x_{\text{noisy}} \sim q(\tilde{x}|x)$ back to the underlying clean ground truth $x \in \mathcal{X}$, effectively learning a projection onto the low-dimensional manifold of natural images.

---

## Dataset Preparation & Noise Models

### 1. Dataset Split
The **Berkeley Segmentation Dataset (BSDS500)** contains 500 natural images divided into:
- **Training set:** 200 images
- **Validation set:** 100 images
- **Testing set:** 200 images

Images are normalized to the standard floating-point range $[0.0, 1.0]$. For training, random patches of size $128 \times 128$ are cropped with random horizontal and vertical reflections for data augmentation. For deterministic testing and validation, fixed center crops are utilized.

### 2. Noise Injection Models
Two distinct degradation models are implemented:

#### A. Additive Gaussian Noise
Simulates sensor thermal and electronic noise:
$$\tilde{x} = \text{clip}\left( x + \epsilon, 0.0, 1.0 \right), \quad \epsilon \sim \mathcal{N}(0, \sigma^2)$$
where $\sigma \in \{0.05, 0.10, 0.15, 0.20, 0.30\}$ controls the noise standard deviation.

#### B. Impulse Salt-and-Pepper Noise
Simulates transmission errors and dead/saturated sensor pixels:
$$\tilde{x}_i = \begin{cases} 
1.0 & \text{with probability } \frac{p}{2} \quad (\text{Salt}) \\ 
0.0 & \text{with probability } \frac{p}{2} \quad (\text{Pepper}) \\ 
x_i & \text{with probability } 1 - p \quad (\text{Uncorrupted}) 
\end{cases}$$
where $p \in \{0.02, 0.05, 0.10, 0.20\}$ denotes the corruption ratio.

---

## Autoencoder Architecture Design

The model is a fully convolutional neural network designed without skip connections so that the network is forced through an **information bottleneck**:

```
Input (3 x 128 x 128)
       │
   [Conv2d (3→32) + BN + LeakyReLU]
   [Conv2d (32→32, stride 2) + BN + LeakyReLU]    -> (32 x 64 x 64)
       │
   [Conv2d (32→64) + BN + LeakyReLU]
   [Conv2d (64→64, stride 2) + BN + LeakyReLU]    -> (64 x 32 x 32)
       │
   [Conv2d (64→Bottleneck_Ch) + BN + LeakyReLU]   -> (Bottleneck_Ch x 32 x 32)  <-- BOTTLENECK
       │
   [Conv2d (Bottleneck_Ch→64) + BN + LeakyReLU]   -> (64 x 32 x 32)
       │
   [ConvTranspose2d (64→32, stride 2) + BN + LeakyReLU]
   [Conv2d (32→32) + BN + LeakyReLU]              -> (32 x 64 x 64)
       │
   [ConvTranspose2d (32→16, stride 2) + BN + LeakyReLU]
   [Conv2d (16→16) + BN + LeakyReLU]              -> (16 x 128 x 128)
       │
   [Conv2d (16→3) + Sigmoid]                      -> Output (3 x 128 x 128)
```

### Architectural Highlights
- **Encoder:** Downsamples spatial dimensions by a factor of 4 via strided convolutions ($128 \times 128 \to 32 \times 32$), learning hierarchical edge, texture, and semantic representations.
- **Bottleneck Capacity:** The bottleneck channels $C_{\text{btn}} \in \{8, 16, 32, 64, 128\}$ dictate the latent compression capacity. For $C_{\text{btn}} = 32$, the latent representation contains $32 \times 32 \times 32 = 32,768$ values (compression ratio $\approx 1.5\times$ relative to $3 \times 128 \times 128 = 49,152$ inputs).
- **Activation Functions:** 
  - **LeakyReLU ($\alpha = 0.2$):** Prevents zero gradients on negative pre-activations and preserves signal propagation through deeper layers.
  - **Batch Normalization:** Stabilizes training dynamics and accelerates convergence.
  - **Sigmoid Output Activation:** Constrains output pixels strictly to $[0.0, 1.0]$, preventing color clipping and artifacts.

---

## Training & Loss Formulations

### Loss Functions
- **Mean Squared Error (MSE):**
  $$\mathcal{L}_{\text{MSE}}(\hat{x}, x) = \frac{1}{C \cdot H \cdot W} \sum_{c, i, j} (\hat{x}_{c, i, j} - x_{c, i, j})^2$$
- **Structural Similarity (SSIM) Loss:**
  $$\mathcal{L}_{\text{SSIM}}(\hat{x}, x) = 1.0 - \text{SSIM}(\hat{x}, x)$$
- **Compound Loss:** $\mathcal{L} = (1 - \alpha)\mathcal{L}_{\text{MSE}} + \alpha \mathcal{L}_{\text{SSIM}}$ ($\alpha = 0.8$).

### Optimizer & Schedule
- **Optimizer:** Adam with initial learning rate $\eta = 10^{-3}$ and weight decay $\lambda = 10^{-5}$.
- **Learning Rate Scheduler:** Cosine Annealing scheduler decaying to $\eta_{\min} = 5 \times 10^{-5}$.

---

## Evaluation Metrics & Visualization

Three quantitative metrics assess restoration quality on the test set:
1. **Mean Squared Error (MSE):** Measures pixel-wise variance.
2. **Peak Signal-to-Noise Ratio (PSNR):**
   $$\text{PSNR} = 10 \cdot \log_{10}\left( \frac{1.0}{\text{MSE}} \right) \quad (\text{dB})$$
3. **Structural Similarity Index (SSIM):** Computes structural, luminance, and contrast correlation across an $11 \times 11$ Gaussian sliding window.

Visualizations display three synchronized rows:
- **Row 1:** Ground Truth Clean Image $x$
- **Row 2:** Corrupted Input Image $\tilde{x}$ (annotated with input PSNR and SSIM)
- **Row 3:** DAE Reconstructed Image $\hat{x}$ (annotated with output PSNR, SSIM, and gain $\Delta\text{PSNR}$)

---

## Experimental Study & Findings

> **Results pending: these come from the Colab run.** Run the *Experimental Study* section of `colab_run.ipynb`.
> It trains one model per setting and writes everything to `experiments_output/`:
> - `noise_study_results.csv` and `noise_study_curves.png`
> - `bottleneck_study_results.csv` and `bottleneck_study_curves.png`
> - `experimental_study_report.md`, which has both tables in Markdown
>
> Paste the two tables from `experimental_study_report.md` below, then write the observations from those numbers.

### Study 1: Impact of Noise Intensity
Bottleneck fixed at 32 channels. Gaussian σ ∈ {0.05, 0.10, 0.20, 0.30}; salt-and-pepper p ∈ {0.02, 0.05, 0.10, 0.20}.

*(table from `experimental_study_report.md`)*

**Observations:** *(fill in after the run: how ΔPSNR/SSIM change with noise level, Gaussian vs salt-and-pepper)*

### Study 2: Impact of Bottleneck Size
Gaussian noise σ = 0.15; bottleneck channels ∈ {8, 16, 32, 64, 128}.

*(table from `experimental_study_report.md`)*

**Observations:** *(fill in after the run: quality vs compression ratio and parameter count, where the gains level off)*

---

## Repository Structure

```
q1/
├── README.md               # Detailed technical documentation and experiment guide
├── requirements.txt        # Python package dependencies
├── dataset.py              # BSDS500 download, PyTorch Dataset, and noise transforms
├── models.py               # ConvDAE model definition with parameterized bottleneck
├── metrics.py              # MSE, PSNR, SSIM, and combined loss implementations
├── train.py                # Model training CLI script with checkpointing & curves
├── evaluate.py             # Evaluation CLI script with 3-row triplet visualization
├── experiments.py          # Sweep runner for noise level & bottleneck studies
└── colab_run.ipynb         # Interactive Jupyter Notebook ready for Google Colab
```

---

## Google Colab Quickstart

To run on Google Colab:
1. Upload the `q1/` directory to your Google Drive or Colab environment.
2. Open `colab_run.ipynb` in Google Colab.
3. Switch runtime to GPU: **Runtime -> Change runtime type -> Hardware accelerator -> T4 GPU**.
4. Run all cells sequentially.

---

## CLI Usage Reference

### 1. Download Dataset & Inspect Noise
```bash
python dataset.py --download --data_dir ./data/BSDS500 --save_sample ./sample_noise.png
```

### 2. Train DAE Model
```bash
# Train on Gaussian noise (std = 0.15)
python train.py \
  --data_dir ./data/BSDS500 \
  --noise_type gaussian \
  --noise_param 0.15 \
  --bottleneck 32 \
  --loss mse \
  --epochs 35 \
  --batch_size 16 \
  --lr 0.001

# Train on Salt-and-Pepper noise (ratio = 0.10)
python train.py \
  --data_dir ./data/BSDS500 \
  --noise_type salt_and_pepper \
  --noise_param 0.10 \
  --bottleneck 32 \
  --loss mse \
  --epochs 35 \
  --batch_size 16 \
  --lr 0.001
```

### 3. Evaluate & Generate Visualizations
```bash
python evaluate.py \
  --checkpoint ./checkpoints/dae_gaussian_p0.15_btn32_mse/best_model.pth \
  --data_dir ./data/BSDS500 \
  --split test \
  --num_samples 5 \
  --save_dir ./results
```

### 4. Run Experimental Study (Task 6)
```bash
# Complete sweep across noise levels and bottleneck dimensions
python experiments.py \
  --data_dir ./data/BSDS500 \
  --output_dir ./experiments_output \
  --epochs 20 \
  --batch_size 16

# Fast mode (reduced epochs for quick validation)
python experiments.py \
  --data_dir ./data/BSDS500 \
  --output_dir ./experiments_output \
  --quick
```
