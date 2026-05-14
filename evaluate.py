import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_fid.fid_score import calculate_fid_given_paths
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image

from config import DATA_ROOT, DEVICE, RESULTS_DIR, image_size, num_workers
from models import Generator
from utils import denormalize


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate baseline and modified CycleGAN checkpoints with FID and SSIM."
    )
    parser.add_argument("--baseline_checkpoint", "--baseline-checkpoint", required=True)
    parser.add_argument("--modified_checkpoint", "--modified-checkpoint", required=True)
    parser.add_argument("--data_root", "--data-root", type=str, default=str(DATA_ROOT))
    parser.add_argument("--output_dir", "--output-dir", type=str, default=str(RESULTS_DIR / "evaluation"))
    parser.add_argument("--batch_size", "--batch-size", type=int, default=1)
    parser.add_argument("--num_workers", "--num-workers", type=int, default=num_workers)
    parser.add_argument("--fid_batch_size", "--fid-batch-size", type=int, default=16)
    parser.add_argument("--fid_dims", "--fid-dims", type=int, default=2048)
    parser.add_argument("--allow_cpu", "--allow-cpu", action="store_true")
    return parser.parse_args()


class SingleDomainImageDataset(Dataset):
    """Deterministic image dataset for metric computation."""

    def __init__(self, image_dir):
        self.image_dir = Path(image_dir)
        self.image_paths = sorted(
            path for path in self.image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.image_paths:
            raise RuntimeError(f"No images found in {self.image_dir}")

        # Evaluation should be deterministic: resize only, no random crop/flip.
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), image_path.stem


def create_loader(image_dir, batch_size, workers):
    dataset = SingleDomainImageDataset(image_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )


def load_generators(checkpoint_path, device):
    """Load G: horse->zebra and F: zebra->horse from a CycleGAN checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    generator_horse_to_zebra = Generator().to(device)
    generator_zebra_to_horse = Generator().to(device)
    generator_horse_to_zebra.load_state_dict(checkpoint["generator_ab"])
    generator_zebra_to_horse.load_state_dict(checkpoint["generator_ba"])
    generator_horse_to_zebra.eval()
    generator_zebra_to_horse.eval()

    return generator_horse_to_zebra, generator_zebra_to_horse


def tensor_to_ssim_image(images):
    """Convert [-1, 1] tensors to NHWC float images in [0, 1]."""
    images = denormalize(images).detach().cpu().permute(0, 2, 3, 1).numpy()
    return np.clip(images, 0.0, 1.0)


def batch_ssim(original_images, reconstructed_images):
    originals = tensor_to_ssim_image(original_images)
    reconstructions = tensor_to_ssim_image(reconstructed_images)

    scores = []
    for original, reconstructed in zip(originals, reconstructions):
        scores.append(
            structural_similarity(
                original,
                reconstructed,
                channel_axis=-1,
                data_range=1.0,
            )
        )
    return scores


def save_batch(images, names, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    images = denormalize(images.detach().cpu())
    for image, name in zip(images, names):
        save_image(image, output_dir / f"{name}.png")


@torch.no_grad()
def run_model_inference(
    model_name,
    checkpoint_path,
    horse_loader,
    zebra_loader,
    output_root,
    device,
):
    """Generate fake images, reconstructed images, and cycle SSIM scores."""
    generator_horse_to_zebra, generator_zebra_to_horse = load_generators(checkpoint_path, device)

    model_output_dir = output_root / model_name
    fake_zebra_dir = model_output_dir / "fake_zebra"
    fake_horse_dir = model_output_dir / "fake_horse"
    reconstructed_horse_dir = model_output_dir / "reconstructed_horse"
    reconstructed_zebra_dir = model_output_dir / "reconstructed_zebra"

    horse_ssim_scores = []
    zebra_ssim_scores = []

    for real_horse, names in horse_loader:
        real_horse = real_horse.to(device)
        fake_zebra = generator_horse_to_zebra(real_horse)
        reconstructed_horse = generator_zebra_to_horse(fake_zebra)

        save_batch(fake_zebra, names, fake_zebra_dir)
        save_batch(reconstructed_horse, names, reconstructed_horse_dir)
        horse_ssim_scores.extend(batch_ssim(real_horse, reconstructed_horse))

    for real_zebra, names in zebra_loader:
        real_zebra = real_zebra.to(device)
        fake_horse = generator_zebra_to_horse(real_zebra)
        reconstructed_zebra = generator_horse_to_zebra(fake_horse)

        save_batch(fake_horse, names, fake_horse_dir)
        save_batch(reconstructed_zebra, names, reconstructed_zebra_dir)
        zebra_ssim_scores.extend(batch_ssim(real_zebra, reconstructed_zebra))

    return {
        "fake_zebra_dir": fake_zebra_dir,
        "fake_horse_dir": fake_horse_dir,
        "ssim_horse_cycle": float(np.mean(horse_ssim_scores)),
        "ssim_zebra_cycle": float(np.mean(zebra_ssim_scores)),
        "ssim_mean": float(np.mean(horse_ssim_scores + zebra_ssim_scores)),
    }


def save_real_images(loader, output_dir, device):
    output_dir.mkdir(parents=True, exist_ok=True)
    for images, names in loader:
        save_batch(images.to(device), names, output_dir)


def compute_fid(fake_dir, real_dir, device, batch_size, dims, workers):
    return calculate_fid_given_paths(
        paths=[str(fake_dir), str(real_dir)],
        batch_size=batch_size,
        device=device,
        dims=dims,
        num_workers=workers,
    )


def print_comparison_table(rows):
    headers = [
        "Model",
        "FID horse->zebra",
        "FID zebra->horse",
        "FID mean",
        "SSIM horse cycle",
        "SSIM zebra cycle",
        "SSIM mean",
    ]
    table_rows = [
        [
            row["model"],
            f"{row['fid_horse_to_zebra']:.4f}",
            f"{row['fid_zebra_to_horse']:.4f}",
            f"{row['fid_mean']:.4f}",
            f"{row['ssim_horse_cycle']:.4f}",
            f"{row['ssim_zebra_cycle']:.4f}",
            f"{row['ssim_mean']:.4f}",
        ]
        for row in rows
    ]
    widths = [
        max(len(header), *(len(row[index]) for row in table_rows))
        for index, header in enumerate(headers)
    ]

    header_line = " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    separator = "-+-".join("-" * width for width in widths)
    print(header_line)
    print(separator)
    for row in table_rows:
        print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def main():
    args = parse_args()
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is required for cluster evaluation. Use --allow-cpu only for debugging.")

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    output_root = Path(args.output_dir)
    data_root = Path(args.data_root)

    horse_loader = create_loader(data_root / "testA", args.batch_size, args.num_workers)
    zebra_loader = create_loader(data_root / "testB", args.batch_size, args.num_workers)

    real_horse_dir = output_root / "real_horse"
    real_zebra_dir = output_root / "real_zebra"
    save_real_images(horse_loader, real_horse_dir, device)
    save_real_images(zebra_loader, real_zebra_dir, device)

    evaluations = [
        ("baseline", args.baseline_checkpoint),
        ("modified", args.modified_checkpoint),
    ]

    rows = []
    for model_name, checkpoint_path in evaluations:
        print(f"Running inference for {model_name}: {checkpoint_path}")
        metrics = run_model_inference(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            horse_loader=horse_loader,
            zebra_loader=zebra_loader,
            output_root=output_root,
            device=device,
        )

        fid_horse_to_zebra = compute_fid(
            fake_dir=metrics["fake_zebra_dir"],
            real_dir=real_zebra_dir,
            device=device,
            batch_size=args.fid_batch_size,
            dims=args.fid_dims,
            workers=args.num_workers,
        )
        fid_zebra_to_horse = compute_fid(
            fake_dir=metrics["fake_horse_dir"],
            real_dir=real_horse_dir,
            device=device,
            batch_size=args.fid_batch_size,
            dims=args.fid_dims,
            workers=args.num_workers,
        )

        rows.append(
            {
                "model": model_name,
                "fid_horse_to_zebra": fid_horse_to_zebra,
                "fid_zebra_to_horse": fid_zebra_to_horse,
                "fid_mean": (fid_horse_to_zebra + fid_zebra_to_horse) / 2.0,
                "ssim_horse_cycle": metrics["ssim_horse_cycle"],
                "ssim_zebra_cycle": metrics["ssim_zebra_cycle"],
                "ssim_mean": metrics["ssim_mean"],
            }
        )

    print("\nQuantitative comparison")
    print_comparison_table(rows)
    print(f"\nGenerated images and real-image metric folders saved to: {output_root}")


if __name__ == "__main__":
    main()

