# CycleGAN — Horse ↔ Zebra Image Translation

Implementation of CycleGAN for unpaired image-to-image translation on the Horse ↔ Zebra task, with a proposed architectural modification replacing pixel-space L1 cycle consistency loss with a perceptual loss computed in VGG-16 feature space.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Proposed Modification](#proposed-modification)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Running the Inference App](#running-the-inference-app)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Limitations](#limitations)
- [References](#references)

---

## Overview

CycleGAN (Zhu et al., 2017) enables unpaired image-to-image translation by learning two mappings simultaneously — G: Horse → Zebra and F: Zebra → Horse — enforced by a cycle consistency constraint. This project implements the baseline architecture and proposes replacing the standard L1 cycle consistency loss with a perceptual loss based on intermediate VGG-16 feature representations, with the goal of producing sharper, more structurally faithful reconstructions.

---

## Architecture

### Generator
A ResNet-based encoder-decoder (Johnson et al., 2016):

```
c7s1-64 → d128 → d256 → [9 × ResidualBlock] → u128 → u64 → c7s1-3
```

- Reflection padding to reduce edge artifacts
- Instance normalization throughout
- Tanh output activation mapping to [-1, 1]

### Discriminator
70×70 PatchGAN:

```
C64 → C128 → C256 → C512 → 1-channel patch output
```

- Instance normalization (except first layer)
- LeakyReLU activations
- No sigmoid — LSGAN (MSE) loss is used instead of BCE

### Loss Functions

**Baseline:**
- Adversarial loss: LSGAN (MSE-based, more stable than vanilla GAN)
- Cycle consistency loss: L1 in pixel space (λ = 10)
- Identity loss: L1 (λ = 5)

**Modified (proposed):**
- Adversarial loss: same LSGAN
- Cycle consistency loss: MSE in VGG-16 feature space (perceptual loss)
- Identity loss: L1 (λ = 5)

---

## Proposed Modification

The standard L1 cycle consistency loss penalizes pixel-level differences, which can produce blurry reconstructions as the network learns an average over plausible outputs. This project proposes replacing it with a perceptual loss that compares images in the feature space of a pretrained, frozen VGG-16 network.

Features are extracted from configurable layers (`relu1_2`, `relu2_2`, `relu3_2`, `relu4_2`), with ImageNet normalization applied before each forward pass. The motivation is that feature-space distances better capture perceptual and structural similarity than raw pixel differences.

---

## Dataset

**Horse2Zebra** — unpaired images from the ImageNet horse and zebra synsets.

| Split | Domain | Images |
|-------|--------|--------|
| `trainA` | Horses | 1,067 |
| `trainB` | Zebras | 1,334 |
| `testA` | Horses | 120 |
| `testB` | Zebras | 140 |

Dataset location: `data/horse2zebra/`

All images are resized to 256×256. During training: random crop and horizontal flip augmentation, then normalized to [-1, 1]. During evaluation: center crop only, no flipping.

---

## Project Structure

```
DLP Project/
├── app.py                          # Gradio inference interface
├── train_baseline.py               # Baseline training script
├── train_modified.py               # Modified (perceptual loss) training script
├── evaluate.py                     # FID + SSIM evaluation
├── ablation_study.py               # VGG layer ablation
├── visualize_results.py            # Loss curves and result grids
├── config.py                       # Hyperparameters and paths
├── Requirements.txt
│
├── models/
│   ├── generator.py                # ResNet generator
│   └── discriminator.py            # PatchGAN discriminator
│
├── losses/
│   ├── cycle_gan_loss.py           # Baseline loss (LSGAN + L1 cycle)
│   ├── modified_cycle_gan_loss.py  # Modified loss (LSGAN + perceptual cycle)
│   └── perceptual_loss.py          # VGG-16 feature extractor
│
├── utils/
│   ├── utils.py                    # Checkpointing, denormalization
│   ├── replay_buffer.py            # Image replay buffer (50% swap)
│   └── lr_scheduler.py             # Linear LR decay scheduler
│
├── data/
│   ├── horse2zebra_dataset.py      # Dataset loader
│   └── horse2zebra/
│       ├── trainA/                 # 1,067 horse images
│       ├── trainB/                 # 1,334 zebra images
│       ├── testA/                  # 120 horse images
│       └── testB/                  # 140 zebra images
│
├── checkpoints/
│   ├── baseline/                   # Saved checkpoints (epoch 10–200)
│   └── modified_relu3_2/           # Saved checkpoints (epochs 10, 20, 80, 90)
│
├── results/
│   ├── baseline/                   # Sample outputs saved every 10 epochs (epochs 10–200)
│   └── modified_relu3_2/           # Sample outputs saved every 10 epochs (epochs 10–90)
│
└── logs/
    └── baseline_losses.csv         # Generator and discriminator loss per epoch (baseline only)
```

---

## Setup

**Requirements:** Python 3.8+, pip

```bash
# Clone or navigate to the project directory
cd "DLP Project"

# Create and activate a virtual environment
python -m venv cyclegan_env
cyclegan_env\Scripts\activate        # Windows
# source cyclegan_env/bin/activate   # Linux/macOS

# Install dependencies
pip install -r Requirements.txt

# Install Gradio (for the inference app)
pip install gradio
```

**Note:** Training requires a CUDA-capable GPU. CPU-only inference is supported for the app.

---

## Running the Inference App

A Gradio web interface for interactive horse ↔ zebra translation using the trained baseline model (epoch 200 checkpoint).

```bash
python app.py
```

Open **http://127.0.0.1:7861** in your browser.

- Upload any horse or zebra image (JPG or PNG)
- Select the translation direction: **Horse → Zebra** or **Zebra → Horse**
- Click **Translate**

Inference runs on CPU. Expect ~2–5 seconds per image.

---

## Training

### Baseline

```bash
python train_baseline.py \
    --checkpoint_dir checkpoints/baseline \
    --results_dir results/baseline \
    --epochs 200
```

To resume from a checkpoint:

```bash
python train_baseline.py --resume checkpoints/baseline/checkpoint_epoch_150.pth
```

### Modified (Perceptual Loss)

```bash
python train_modified.py \
    --checkpoint_dir checkpoints/modified \
    --results_dir results/modified \
    --vgg_layer relu3_2 \
    --epochs 200
```

### Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Image size | 256×256 |
| Batch size | 4 |
| Learning rate | 0.0002 |
| Optimizer | Adam (β₁=0.5, β₂=0.999) |
| λ cycle | 10 |
| λ identity | 5 |
| Total epochs | 200 |
| LR decay start | Epoch 100 |
| Residual blocks | 9 |

Checkpoints and sample images are saved every 10 epochs. Mixed precision (AMP) and discriminator replay buffer (buffer size 50) are used throughout.

---

## Evaluation

Quantitative comparison using FID (Fréchet Inception Distance) and SSIM (Structural Similarity Index):

```bash
python evaluate.py \
    --baseline_checkpoint checkpoints/baseline/checkpoint_epoch_200.pth \
    --modified_checkpoint checkpoints/modified_relu3_2/checkpoint_epoch_090.pth \
    --allow_cpu
```

Ablation study across VGG layers:

```bash
python ablation_study.py \
    --baseline_checkpoint checkpoints/baseline/checkpoint_epoch_200.pth
```

Visualize loss curves and result grids:

```bash
python visualize_results.py \
    --baseline_loss_log logs/baseline_losses.csv \
    --baseline_results results/baseline
```

---

## Results

### Training Convergence (Baseline, Epochs 71–150)

| Epoch | Generator Loss | Discriminator Loss |
|-------|---------------|-------------------|
| 71 | 3.630 | 0.468 |
| 80 | 3.219 | 0.425 |
| 90 | 3.257 | 0.419 |
| 100 | 3.714 | 0.277 |
| 110 | 2.849 | 0.427 |
| 120 | 2.873 | 0.315 |
| 130 | 3.060 | 0.525 |
| 140 | 3.376 | 0.178 |
| 150 | 2.690 | 0.404 |

Generator loss decreased steadily from ~3.6 to ~2.7 over this range. Discriminator loss remained in the 0.17–0.54 range, indicating stable adversarial training without collapse.

The baseline was trained to the full 200 epochs including the linear LR decay phase (epochs 100–200). Qualitative results saved at every 10 epochs are available in `results/baseline/`.

The modified (relu3_2 perceptual loss) model was trained to epoch 90. Qualitative results for epochs 10–90 are available in `results/modified_relu3_2/`. Due to checkpoint corruption during transfer, `.pth` files are only available for epochs 10, 20, 80, and 90; the epoch 90 checkpoint is used for evaluation.

### Observations

- **Horse → Zebra** translation produces consistently good results at epoch 150: stripes are well-placed and textures are plausible.
- **Zebra → Horse** is visibly weaker — a known asymmetry documented in the original CycleGAN paper. Removing the high-frequency stripe pattern is structurally harder than adding one. Background domain shift (savanna vs. varied environments) also contributes.

---

## Limitations

- The modified (relu3_2 perceptual loss) model was trained to epoch 90; full 200-epoch training was not completed due to resource and time constraints.
- Checkpoint files for the modified model are available only at epochs 10, 20, 80, and 90 — intermediate checkpoints were lost during file transfer due to corruption.
- Loss logs are available for the baseline only; training logs for the modified model were not retained.
- The Zebra → Horse direction underperforms relative to Horse → Zebra, consistent with findings in the original CycleGAN paper.

---

## References

- Zhu, J.-Y., Park, T., Isola, P., & Efros, A. A. (2017). *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks*. ICCV 2017.
- Johnson, J., Alahi, A., & Fei-Fei, L. (2016). *Perceptual Losses for Real-Time Style Transfer and Super-Resolution*. ECCV 2016.
- Simonyan, K., & Zisserman, A. (2014). *Very Deep Convolutional Networks for Large-Scale Image Recognition*. ICLR 2015.
- Mao, X., et al. (2017). *Least Squares Generative Adversarial Networks*. ICCV 2017.
