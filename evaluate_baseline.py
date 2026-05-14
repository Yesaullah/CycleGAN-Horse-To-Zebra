"""
Baseline-only evaluation: computes FID and SSIM for the trained CycleGAN baseline.

Usage (Colab / GPU):
    python evaluate_baseline.py \
        --checkpoint checkpoints/baseline/checkpoint_epoch_150.pth \
        --data_root  data/horse2zebra

Usage (CPU, slower):
    python evaluate_baseline.py \
        --checkpoint checkpoints/baseline/checkpoint_epoch_150.pth \
        --data_root  data/horse2zebra \
        --allow_cpu
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_fid.fid_score import calculate_fid_given_paths
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image

import sys
import types

# generator.py imports config.py — provide the two values it needs inline
# so no config.py file is required in the Colab environment
_config = types.ModuleType("config")
_config.image_size = 256
_config.num_residual_blocks = 9
sys.modules["config"] = _config

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.generator import Generator


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ── Dataset ────────────────────────────────────────────────────────────────────
class ImageFolder(Dataset):
    def __init__(self, folder):
        self.paths = sorted(
            p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTS
        )
        if not self.paths:
            raise RuntimeError(f"No images found in {folder}")
        self.transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.paths[idx].stem


# ── Helpers ────────────────────────────────────────────────────────────────────
def denorm(t):
    return t.mul(0.5).add(0.5).clamp(0, 1)


def load_generators(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    G = Generator().to(device)   # horse -> zebra
    F = Generator().to(device)   # zebra -> horse
    G.load_state_dict(ckpt["generator_ab"])
    F.load_state_dict(ckpt["generator_ba"])
    G.eval()
    F.eval()
    return G, F


def save_batch(imgs, names, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for img, name in zip(denorm(imgs.cpu()), names):
        save_image(img, out_dir / f"{name}.png")


def compute_ssim(originals, reconstructed):
    def to_np(t):
        return denorm(t).detach().cpu().permute(0, 2, 3, 1).numpy().clip(0, 1)
    scores = []
    for orig, rec in zip(to_np(originals), to_np(reconstructed)):
        scores.append(structural_similarity(orig, rec, channel_axis=-1, data_range=1.0))
    return scores


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to baseline checkpoint .pth file")
    parser.add_argument("--data_root", default="data/horse2zebra",
                        help="Root folder containing testA and testB")
    parser.add_argument("--output_dir", default="results/evaluation",
                        help="Where to save generated images")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--allow_cpu", action="store_true")
    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.allow_cpu:
        device = torch.device("cpu")
        print("WARNING: Running on CPU. FID computation will be slow.")
    else:
        raise RuntimeError("No CUDA device found. Pass --allow_cpu to run on CPU.")

    print(f"Device : {device}")
    print(f"Checkpoint : {args.checkpoint}")

    data_root  = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Directories for generated and real images
    fake_zebra_dir  = output_dir / "fake_zebra"
    fake_horse_dir  = output_dir / "fake_horse"
    rec_horse_dir   = output_dir / "rec_horse"
    rec_zebra_dir   = output_dir / "rec_zebra"
    real_zebra_dir  = output_dir / "real_zebra"
    real_horse_dir  = output_dir / "real_horse"

    # Data loaders
    horse_loader = DataLoader(
        ImageFolder(data_root / "testA"), batch_size=args.batch_size,
        shuffle=False, num_workers=2, pin_memory=device.type == "cuda")
    zebra_loader = DataLoader(
        ImageFolder(data_root / "testB"), batch_size=args.batch_size,
        shuffle=False, num_workers=2, pin_memory=device.type == "cuda")

    # Load generators
    print("\nLoading generators...")
    G, F = load_generators(args.checkpoint, device)

    # Save real images (needed by FID)
    print("Saving real images for FID reference...")
    for imgs, names in horse_loader:
        save_batch(imgs.to(device), names, real_horse_dir)
    for imgs, names in zebra_loader:
        save_batch(imgs.to(device), names, real_zebra_dir)

    # Inference + SSIM
    print("\nRunning inference on test set...")
    horse_ssim, zebra_ssim = [], []

    with torch.no_grad():
        for imgs, names in horse_loader:
            real_h  = imgs.to(device)
            fake_z  = G(real_h)
            rec_h   = F(fake_z)
            save_batch(fake_z, names, fake_zebra_dir)
            save_batch(rec_h,  names, rec_horse_dir)
            horse_ssim.extend(compute_ssim(real_h, rec_h))
            print(f"  Processed {len(horse_ssim)}/{len(horse_loader.dataset)} horse images", end="\r")

        print()
        for imgs, names in zebra_loader:
            real_z  = imgs.to(device)
            fake_h  = F(real_z)
            rec_z   = G(fake_h)
            save_batch(fake_h, names, fake_horse_dir)
            save_batch(rec_z,  names, rec_zebra_dir)
            zebra_ssim.extend(compute_ssim(real_z, rec_z))
            print(f"  Processed {len(zebra_ssim)}/{len(zebra_loader.dataset)} zebra images", end="\r")

    print()

    # FID
    print("\nComputing FID scores (this may take a few minutes)...")
    fid_h2z = calculate_fid_given_paths(
        [str(fake_zebra_dir), str(real_zebra_dir)],
        batch_size=16, device=device, dims=2048, num_workers=2)

    fid_z2h = calculate_fid_given_paths(
        [str(fake_horse_dir), str(real_horse_dir)],
        batch_size=16, device=device, dims=2048, num_workers=2)

    # Results
    ssim_h = float(np.mean(horse_ssim))
    ssim_z = float(np.mean(zebra_ssim))
    ssim_mean = float(np.mean(horse_ssim + zebra_ssim))

    print("\n" + "="*55)
    print("  BASELINE EVALUATION RESULTS (Epoch 150)")
    print("="*55)
    print(f"  FID  Horse -> Zebra  : {fid_h2z:.2f}")
    print(f"  FID  Zebra -> Horse  : {fid_z2h:.2f}")
    print(f"  FID  Mean            : {(fid_h2z + fid_z2h) / 2:.2f}")
    print("-"*55)
    print(f"  SSIM Horse cycle     : {ssim_h:.4f}")
    print(f"  SSIM Zebra cycle     : {ssim_z:.4f}")
    print(f"  SSIM Mean            : {ssim_mean:.4f}")
    print("="*55)
    print(f"\nGenerated images saved to: {output_dir}")


if __name__ == "__main__":
    main()
