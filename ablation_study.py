import argparse
from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image

from config import CHECKPOINT_DIR, DATA_ROOT, DEVICE, RESULTS_DIR, num_workers
from evaluate import (
    compute_fid,
    create_loader,
    load_generators,
    run_model_inference,
    save_real_images,
)
from utils import denormalize


VGG_LAYERS = ("relu1_2", "relu2_2", "relu3_2", "relu4_2")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a VGG-layer ablation study for the modified CycleGAN."
    )
    parser.add_argument("--baseline_checkpoint", "--baseline-checkpoint", required=True)
    parser.add_argument("--relu1_2_checkpoint", type=str, default=None)
    parser.add_argument("--relu2_2_checkpoint", type=str, default=None)
    parser.add_argument("--relu3_2_checkpoint", type=str, default=None)
    parser.add_argument("--relu4_2_checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint_dir", "--checkpoint-dir", type=str, default=str(CHECKPOINT_DIR / "modified"))
    parser.add_argument("--data_root", "--data-root", type=str, default=str(DATA_ROOT))
    parser.add_argument("--output_dir", "--output-dir", type=str, default=str(RESULTS_DIR / "ablation"))
    parser.add_argument("--batch_size", "--batch-size", type=int, default=1)
    parser.add_argument("--num_workers", "--num-workers", type=int, default=num_workers)
    parser.add_argument("--fid_batch_size", "--fid-batch-size", type=int, default=16)
    parser.add_argument("--fid_dims", "--fid-dims", type=int, default=2048)
    parser.add_argument("--visualization_layer", "--visualization-layer", type=str, default="relu3_2", choices=VGG_LAYERS)
    parser.add_argument("--visualization_count", "--visualization-count", type=int, default=4)
    parser.add_argument("--allow_cpu", "--allow-cpu", action="store_true")
    return parser.parse_args()


def resolve_checkpoints(args):
    checkpoints = {}
    for layer in VGG_LAYERS:
        explicit_path = getattr(args, f"{layer}_checkpoint")
        if explicit_path:
            checkpoints[layer] = Path(explicit_path)
        else:
            candidates = sorted(Path(args.checkpoint_dir).glob(f"modified_{layer}_epoch_*.pth"))
            if not candidates:
                raise FileNotFoundError(
                    f"No checkpoint found for {layer}. Pass --{layer}_checkpoint or place "
                    f"modified_{layer}_epoch_XXX.pth in {args.checkpoint_dir}."
                )
            checkpoints[layer] = candidates[-1]
    return checkpoints


def format_table(rows):
    headers = ["VGG Layer", "FID Score", "SSIM"]
    table_rows = [
        [row["label"], f"{row['fid_score']:.2f}", f"{row['ssim']:.3f}"]
        for row in rows
    ]
    widths = [
        max(len(header), *(len(row[index]) for row in table_rows))
        for index, header in enumerate(headers)
    ]

    header_line = "| " + " | ".join(
        header.ljust(widths[index]) for index, header in enumerate(headers)
    ) + " |"
    separator = "|-" + "-|-".join("-" * width for width in widths) + "-|"
    body = [
        "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) + " |"
        for row in table_rows
    ]
    return "\n".join([header_line, separator, *body])


@torch.no_grad()
def save_side_by_side_grid(
    baseline_checkpoint,
    modified_checkpoint,
    horse_loader,
    output_path,
    device,
    max_images=4,
):
    """Save Input | Baseline Output | Modified Output | Reconstructed grid.

    The grid uses horse->zebra as the visual direction: input horse, baseline
    fake zebra, modified fake zebra, and modified cycle reconstruction.
    """
    baseline_g, _ = load_generators(baseline_checkpoint, device)
    modified_g, modified_f = load_generators(modified_checkpoint, device)

    rows = []
    used_images = 0
    for real_horse, _ in horse_loader:
        real_horse = real_horse.to(device)
        baseline_fake_zebra = baseline_g(real_horse)
        modified_fake_zebra = modified_g(real_horse)
        modified_reconstructed_horse = modified_f(modified_fake_zebra)

        batch_rows = torch.stack(
            [
                real_horse,
                baseline_fake_zebra,
                modified_fake_zebra,
                modified_reconstructed_horse,
            ],
            dim=1,
        )
        rows.append(batch_rows.flatten(0, 1))
        used_images += real_horse.size(0)
        if used_images >= max_images:
            break

    grid_images = torch.cat(rows, dim=0)[: max_images * 4]
    grid = make_grid(denormalize(grid_images.detach().cpu()), nrow=4, padding=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, output_path)


def main():
    args = parse_args()
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is required for cluster evaluation. Use --allow-cpu only for debugging.")

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    data_root = Path(args.data_root)
    output_root = Path(args.output_dir)

    horse_loader = create_loader(data_root / "testA", args.batch_size, args.num_workers)
    zebra_loader = create_loader(data_root / "testB", args.batch_size, args.num_workers)

    real_horse_dir = output_root / "real_horse"
    real_zebra_dir = output_root / "real_zebra"
    save_real_images(horse_loader, real_horse_dir, device)
    save_real_images(zebra_loader, real_zebra_dir, device)

    layer_checkpoints = resolve_checkpoints(args)
    evaluations = [(layer, layer_checkpoints[layer]) for layer in VGG_LAYERS]
    evaluations.append(("Baseline", Path(args.baseline_checkpoint)))

    rows = []
    for label, checkpoint_path in evaluations:
        model_name = label.lower()
        print(f"Evaluating {label}: {checkpoint_path}")
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
                "label": label,
                "fid_score": (fid_horse_to_zebra + fid_zebra_to_horse) / 2.0,
                "ssim": metrics["ssim_mean"],
            }
        )

    print("\nAblation study results")
    print(format_table(rows))

    visualization_checkpoint = layer_checkpoints[args.visualization_layer]
    grid_path = output_root / f"ablation_grid_{args.visualization_layer}.png"
    save_side_by_side_grid(
        baseline_checkpoint=args.baseline_checkpoint,
        modified_checkpoint=visualization_checkpoint,
        horse_loader=horse_loader,
        output_path=grid_path,
        device=device,
        max_images=args.visualization_count,
    )
    print(f"\nSaved report grid: {grid_path}")


if __name__ == "__main__":
    main()
