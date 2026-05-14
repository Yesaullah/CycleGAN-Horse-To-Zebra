import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import structural_similarity

from config import DATA_ROOT, DEVICE, RESULTS_DIR, num_workers
from evaluate import create_loader, load_generators
from losses.perceptual_loss import PerceptualLoss
from utils import denormalize


LOG_PATTERN = re.compile(
    r"Epoch\s+\[(?P<epoch>\d+)/\d+\].*?"
    r"G:\s+(?P<generator>[0-9.]+).*?"
    r"D:\s+(?P<discriminator>[0-9.]+)"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate report figures for CycleGAN experiments.")
    parser.add_argument("--baseline_checkpoint", "--baseline-checkpoint", required=True)
    parser.add_argument("--modified_checkpoint", "--modified-checkpoint", required=True)
    parser.add_argument("--baseline_loss_log", "--baseline-loss-log", type=str, default=None)
    parser.add_argument("--modified_loss_log", "--modified-loss-log", type=str, default=None)
    parser.add_argument("--data_root", "--data-root", type=str, default=str(DATA_ROOT))
    parser.add_argument("--output_dir", "--output-dir", type=str, default=str(RESULTS_DIR / "figures"))
    parser.add_argument("--batch_size", "--batch-size", type=int, default=1)
    parser.add_argument("--num_workers", "--num-workers", type=int, default=num_workers)
    parser.add_argument("--grid_count", "--grid-count", type=int, default=8)
    parser.add_argument("--failure_count", "--failure-count", type=int, default=4)
    parser.add_argument("--vgg_layer", "--vgg-layer", type=str, default="all")
    parser.add_argument("--pretrained_vgg", "--pretrained-vgg", action="store_true", default=True)
    parser.add_argument("--no_pretrained_vgg", "--no-pretrained-vgg", action="store_false", dest="pretrained_vgg")
    parser.add_argument("--allow_cpu", "--allow-cpu", action="store_true")
    return parser.parse_args()


def configure_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def tensor_to_image(tensor):
    image = denormalize(tensor.detach().cpu()).permute(1, 2, 0).numpy()
    return np.clip(image, 0.0, 1.0)


def tensor_batch_to_images(tensors):
    return [tensor_to_image(tensor) for tensor in tensors]


def load_loss_history(path):
    """Load loss curves from CSV files or plain training logs.

    CSV files may use columns such as epoch, generator_loss, discriminator_loss.
    Plain logs are parsed from lines printed by train_baseline.py/train_modified.py.
    """
    if path is None:
        return None

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Loss log not found: {path}")

    if path.suffix.lower() == ".csv":
        return load_loss_history_from_csv(path)
    return load_loss_history_from_text(path)


def load_loss_history_from_csv(path):
    rows = []
    with path.open("r", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            epoch = int(float(row.get("epoch", row.get("Epoch"))))
            generator = row.get("generator_loss", row.get("G", row.get("g_loss")))
            discriminator = row.get("discriminator_loss", row.get("D", row.get("d_loss")))
            rows.append((epoch, float(generator), float(discriminator)))
    return average_losses_by_epoch(rows)


def load_loss_history_from_text(path):
    rows = []
    for line in path.read_text().splitlines():
        match = LOG_PATTERN.search(line)
        if match:
            rows.append(
                (
                    int(match.group("epoch")),
                    float(match.group("generator")),
                    float(match.group("discriminator")),
                )
            )
    if not rows:
        raise ValueError(f"No train_baseline.py/train_modified.py loss lines found in {path}")
    return average_losses_by_epoch(rows)


def average_losses_by_epoch(rows):
    grouped = {}
    for epoch, generator, discriminator in rows:
        grouped.setdefault(epoch, {"generator": [], "discriminator": []})
        grouped[epoch]["generator"].append(generator)
        grouped[epoch]["discriminator"].append(discriminator)

    epochs = sorted(grouped)
    return {
        "epoch": np.array(epochs),
        "generator": np.array([np.mean(grouped[epoch]["generator"]) for epoch in epochs]),
        "discriminator": np.array([np.mean(grouped[epoch]["discriminator"]) for epoch in epochs]),
    }


def plot_loss_curves(baseline_history, modified_history, output_path):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    if baseline_history is not None:
        ax.plot(baseline_history["epoch"], baseline_history["generator"], label="Baseline generator", linewidth=2.0)
        ax.plot(baseline_history["epoch"], baseline_history["discriminator"], label="Baseline discriminator", linewidth=2.0)
    if modified_history is not None:
        ax.plot(modified_history["epoch"], modified_history["generator"], label="Modified generator", linewidth=2.0)
        ax.plot(modified_history["epoch"], modified_history["discriminator"], label="Modified discriminator", linewidth=2.0)

    if baseline_history is None and modified_history is None:
        ax.text(
            0.5,
            0.5,
            "No loss logs provided",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
        )

    ax.set_title("Training Loss Curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def collect_comparison_rows(horse_loader, baseline_g, modified_g, modified_f, device, max_images):
    rows = []
    for real_horse, names in horse_loader:
        real_horse = real_horse.to(device)
        baseline_zebra = baseline_g(real_horse)
        modified_zebra = modified_g(real_horse)
        reconstructed_horse = modified_f(modified_zebra)

        for index, name in enumerate(names):
            rows.append(
                {
                    "name": name,
                    "real_horse": real_horse[index].detach().cpu(),
                    "baseline_zebra": baseline_zebra[index].detach().cpu(),
                    "modified_zebra": modified_zebra[index].detach().cpu(),
                    "reconstructed_horse": reconstructed_horse[index].detach().cpu(),
                }
            )
            if len(rows) >= max_images:
                return rows
    return rows


def plot_image_grid(rows, output_path, title):
    columns = ["Real Horse", "Baseline Zebra", "Modified Zebra", "Reconstructed Horse"]
    keys = ["real_horse", "baseline_zebra", "modified_zebra", "reconstructed_horse"]
    fig, axes = plt.subplots(
        len(rows),
        len(columns),
        figsize=(14, 3.1 * len(rows)),
        squeeze=False,
    )

    for row_index, row in enumerate(rows):
        for column_index, (column, key) in enumerate(zip(columns, keys)):
            axis = axes[row_index, column_index]
            axis.imshow(tensor_to_image(row[key]))
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                axis.set_title(column)
            if column_index == 0:
                axis.set_ylabel(row["name"], rotation=0, labelpad=45, va="center")

    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def image_ssim(original_tensor, reconstructed_tensor):
    original = tensor_to_image(original_tensor)
    reconstructed = tensor_to_image(reconstructed_tensor)
    return structural_similarity(original, reconstructed, channel_axis=-1, data_range=1.0)


@torch.no_grad()
def find_failure_cases(horse_loader, baseline_g, modified_g, modified_f, device, count):
    """Select lowest-SSIM modified cycle reconstructions as reproducible failure cases."""
    candidates = []
    for real_horse, names in horse_loader:
        real_horse = real_horse.to(device)
        baseline_zebra = baseline_g(real_horse)
        modified_zebra = modified_g(real_horse)
        reconstructed_horse = modified_f(modified_zebra)

        for index, name in enumerate(names):
            score = image_ssim(real_horse[index].cpu(), reconstructed_horse[index].cpu())
            candidates.append(
                {
                    "name": f"{name} | SSIM {score:.3f}",
                    "score": score,
                    "real_horse": real_horse[index].detach().cpu(),
                    "baseline_zebra": baseline_zebra[index].detach().cpu(),
                    "modified_zebra": modified_zebra[index].detach().cpu(),
                    "reconstructed_horse": reconstructed_horse[index].detach().cpu(),
                }
            )

    candidates.sort(key=lambda item: item["score"])
    return candidates[:count]


def normalize_feature_map(feature_map):
    feature_map = feature_map.detach().cpu()
    feature_map = feature_map.mean(dim=0).numpy()
    min_value = feature_map.min()
    max_value = feature_map.max()
    if max_value - min_value < 1e-8:
        return np.zeros_like(feature_map)
    return (feature_map - min_value) / (max_value - min_value)


def normalize_difference_map(feature_a, feature_b):
    difference = (feature_a - feature_b).abs().detach().cpu().mean(dim=0).numpy()
    max_value = difference.max()
    if max_value < 1e-8:
        return np.zeros_like(difference)
    return difference / max_value


@torch.no_grad()
def plot_vgg_feature_visualization(
    horse_loader,
    modified_g,
    modified_f,
    device,
    output_path,
    vgg_layer,
    pretrained_vgg,
):
    real_horse, _ = next(iter(horse_loader))
    real_horse = real_horse[:1].to(device)
    modified_zebra = modified_g(real_horse)
    reconstructed_horse = modified_f(modified_zebra)

    perceptual = PerceptualLoss(
        vgg_layer=vgg_layer,
        pretrained=pretrained_vgg,
        device=str(device),
    )
    real_features = perceptual.feature_extractor(perceptual._normalize_for_vgg(real_horse))
    reconstructed_features = perceptual.feature_extractor(
        perceptual._normalize_for_vgg(reconstructed_horse)
    )

    layers = list(real_features.keys())
    fig, axes = plt.subplots(len(layers), 3, figsize=(12, 3.2 * len(layers)), squeeze=False)
    for row_index, layer in enumerate(layers):
        real_map = normalize_feature_map(real_features[layer][0])
        reconstructed_map = normalize_feature_map(reconstructed_features[layer][0])
        difference_map = normalize_difference_map(real_features[layer][0], reconstructed_features[layer][0])

        panels = [
            ("Original feature", real_map, "magma"),
            ("Reconstructed feature", reconstructed_map, "magma"),
            ("Absolute difference", difference_map, "viridis"),
        ]
        for column_index, (title, image, cmap) in enumerate(panels):
            axis = axes[row_index, column_index]
            axis.imshow(image, cmap=cmap)
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                axis.set_title(title)
            if column_index == 0:
                axis.set_ylabel(layer, rotation=0, labelpad=38, va="center")

    fig.suptitle("VGG Feature Maps Used by Perceptual Cycle Loss", y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is required for cluster visualization. Use --allow-cpu only for debugging.")

    configure_style()
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_history = load_loss_history(args.baseline_loss_log)
    modified_history = load_loss_history(args.modified_loss_log)
    plot_loss_curves(
        baseline_history=baseline_history,
        modified_history=modified_history,
        output_path=output_dir / "training_loss_curves.png",
    )

    horse_loader = create_loader(
        Path(args.data_root) / "testA",
        batch_size=args.batch_size,
        workers=args.num_workers,
    )
    baseline_g, _ = load_generators(args.baseline_checkpoint, device)
    modified_g, modified_f = load_generators(args.modified_checkpoint, device)

    comparison_rows = collect_comparison_rows(
        horse_loader=horse_loader,
        baseline_g=baseline_g,
        modified_g=modified_g,
        modified_f=modified_f,
        device=device,
        max_images=args.grid_count,
    )
    plot_image_grid(
        rows=comparison_rows,
        output_path=output_dir / "qualitative_comparison_grid.png",
        title="Qualitative Comparison",
    )

    failure_rows = find_failure_cases(
        horse_loader=horse_loader,
        baseline_g=baseline_g,
        modified_g=modified_g,
        modified_f=modified_f,
        device=device,
        count=args.failure_count,
    )
    plot_image_grid(
        rows=failure_rows,
        output_path=output_dir / "failure_cases.png",
        title="Lowest-SSIM Modified Cycle Reconstructions",
    )

    plot_vgg_feature_visualization(
        horse_loader=horse_loader,
        modified_g=modified_g,
        modified_f=modified_f,
        device=device,
        output_path=output_dir / "vgg_feature_visualization.png",
        vgg_layer=args.vgg_layer,
        pretrained_vgg=args.pretrained_vgg,
    )

    print(f"Saved report figures to: {output_dir}")
    print("Generated:")
    print(f"- {output_dir / 'training_loss_curves.png'}")
    print(f"- {output_dir / 'qualitative_comparison_grid.png'}")
    print(f"- {output_dir / 'failure_cases.png'}")
    print(f"- {output_dir / 'vgg_feature_visualization.png'}")


if __name__ == "__main__":
    main()

