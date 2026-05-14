import argparse
from pathlib import Path

import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from config import (
    CHECKPOINT_DIR,
    DATA_ROOT,
    DEVICE,
    RESULTS_DIR,
    batch_size as config_batch_size,
    betas,
    decay_epoch,
    lambda_cycle as config_lambda_cycle,
    lr as config_lr,
    n_epochs,
    num_workers,
)
from data import Horse2ZebraDataset
from losses import CycleGANLoss
from models import Generator, PatchGANDiscriminator
from utils import ReplayBuffer, denormalize, get_linear_decay_scheduler


def parse_args():
    parser = argparse.ArgumentParser(description="Train the original CycleGAN baseline.")
    parser.add_argument("--data-root", type=str, default=str(DATA_ROOT))
    parser.add_argument("--epochs", type=int, default=n_epochs)
    parser.add_argument("--batch_size", "--batch-size", type=int, default=config_batch_size)
    parser.add_argument("--lr", type=float, default=config_lr)
    parser.add_argument("--lambda_cycle", "--lambda-cycle", type=float, default=config_lambda_cycle)
    parser.add_argument("--vgg_layer", "--vgg-layer", type=str, default="relu3_2")
    parser.add_argument("--checkpoint_dir", "--checkpoint-dir", type=str, default=str(CHECKPOINT_DIR / "baseline"))
    parser.add_argument("--results_dir", "--results-dir", type=str, default=str(RESULTS_DIR / "baseline"))
    parser.add_argument("--decay-epoch", type=int, default=decay_epoch)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision training")
    return parser.parse_args()


def create_dataloader(data_root, split, batch_size_value, shuffle):
    dataset = Horse2ZebraDataset(root_dir=data_root, split=split)
    return DataLoader(
        dataset,
        batch_size=batch_size_value,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def set_requires_grad(models, requires_grad):
    if not isinstance(models, (list, tuple)):
        models = [models]

    for model in models:
        for parameter in model.parameters():
            parameter.requires_grad = requires_grad


def weights_init_normal(module):
    class_name = module.__class__.__name__
    if "Conv" in class_name:
        torch.nn.init.normal_(module.weight.data, mean=0.0, std=0.02)
        if getattr(module, "bias", None) is not None:
            torch.nn.init.constant_(module.bias.data, 0.0)
    elif "InstanceNorm2d" in class_name and getattr(module, "weight", None) is not None:
        torch.nn.init.normal_(module.weight.data, mean=1.0, std=0.02)
        torch.nn.init.constant_(module.bias.data, 0.0)


def save_training_checkpoint(
    file_path,
    epoch,
    generator_horse_to_zebra,
    generator_zebra_to_horse,
    discriminator_horse,
    discriminator_zebra,
    optimizer_g,
    optimizer_d,
    scheduler_g,
    scheduler_d,
):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "generator_ab": generator_horse_to_zebra.state_dict(),
            "generator_ba": generator_zebra_to_horse.state_dict(),
            "discriminator_a": discriminator_horse.state_dict(),
            "discriminator_b": discriminator_zebra.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
            "scheduler_g": scheduler_g.state_dict(),
            "scheduler_d": scheduler_d.state_dict(),
        },
        file_path,
    )


def load_training_checkpoint(
    file_path,
    device,
    generator_horse_to_zebra,
    generator_zebra_to_horse,
    discriminator_horse,
    discriminator_zebra,
    optimizer_g,
    optimizer_d,
    scheduler_g,
    scheduler_d,
):
    checkpoint = torch.load(file_path, map_location=device)
    generator_horse_to_zebra.load_state_dict(checkpoint["generator_ab"])
    generator_zebra_to_horse.load_state_dict(checkpoint["generator_ba"])
    discriminator_horse.load_state_dict(checkpoint["discriminator_a"])
    discriminator_zebra.load_state_dict(checkpoint["discriminator_b"])
    optimizer_g.load_state_dict(checkpoint["optimizer_g"])
    optimizer_d.load_state_dict(checkpoint["optimizer_d"])

    if "scheduler_g" in checkpoint:
        scheduler_g.load_state_dict(checkpoint["scheduler_g"])
    if "scheduler_d" in checkpoint:
        scheduler_d.load_state_dict(checkpoint["scheduler_d"])

    return checkpoint["epoch"] + 1


@torch.no_grad()
def save_sample_outputs(
    epoch,
    real_horse,
    real_zebra,
    generator_horse_to_zebra,
    generator_zebra_to_horse,
    output_dir,
):
    generator_horse_to_zebra.eval()
    generator_zebra_to_horse.eval()

    fake_zebra = generator_horse_to_zebra(real_horse)
    reconstructed_horse = generator_zebra_to_horse(fake_zebra)
    fake_horse = generator_zebra_to_horse(real_zebra)
    reconstructed_zebra = generator_horse_to_zebra(fake_horse)

    image_grid = make_grid(
        denormalize(
            torch.cat(
                [
                    real_horse,
                    fake_zebra,
                    reconstructed_horse,
                    real_zebra,
                    fake_horse,
                    reconstructed_zebra,
                ],
                dim=0,
            )
        ),
        nrow=3,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_image(image_grid, output_dir / f"epoch_{epoch:03d}.png")

    generator_horse_to_zebra.train()
    generator_zebra_to_horse.train()


def main():
    args = parse_args()
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is required for cluster training. Use --allow-cpu only for debugging.")

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    
    results_dir = Path(args.results_dir)
    checkpoint_dir = Path(args.checkpoint_dir)

    train_loader = create_dataloader(
        data_root=args.data_root,
        split="train",
        batch_size_value=args.batch_size,
        shuffle=True,
    )
    sample_loader = create_dataloader(
        data_root=args.data_root,
        split="test",
        batch_size_value=args.batch_size,
        shuffle=True,
    )
    sample_batch = next(iter(sample_loader))
    sample_horse = sample_batch["A"].to(device)
    sample_zebra = sample_batch["B"].to(device)

    generator_horse_to_zebra = Generator().to(device)
    generator_zebra_to_horse = Generator().to(device)
    discriminator_horse = PatchGANDiscriminator().to(device)
    discriminator_zebra = PatchGANDiscriminator().to(device)

    generator_horse_to_zebra.apply(weights_init_normal)
    generator_zebra_to_horse.apply(weights_init_normal)
    discriminator_horse.apply(weights_init_normal)
    discriminator_zebra.apply(weights_init_normal)

    optimizer_g = optim.Adam(
        list(generator_horse_to_zebra.parameters())
        + list(generator_zebra_to_horse.parameters()),
        lr=args.lr,
        betas=betas,
    )
    optimizer_d = optim.Adam(
        list(discriminator_horse.parameters()) + list(discriminator_zebra.parameters()),
        lr=args.lr,
        betas=betas,
    )

    scheduler_g = get_linear_decay_scheduler(
        optimizer_g,
        start_decay_epoch=args.decay_epoch,
        total_epochs=args.epochs,
    )
    scheduler_d = get_linear_decay_scheduler(
        optimizer_d,
        start_decay_epoch=args.decay_epoch,
        total_epochs=args.epochs,
    )

    loss_fn = CycleGANLoss(lambda_cycle_weight=args.lambda_cycle, device=str(device))
    fake_horse_buffer = ReplayBuffer(max_size=50)
    fake_zebra_buffer = ReplayBuffer(max_size=50)

    # Initialize AMP scalers
    use_amp = not args.no_amp and device.type == "cuda"
    scaler_g = torch.amp.GradScaler("cuda", enabled=use_amp)
    scaler_d = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 0
    if args.resume:
        start_epoch = load_training_checkpoint(
            file_path=args.resume,
            device=device,
            generator_horse_to_zebra=generator_horse_to_zebra,
            generator_zebra_to_horse=generator_zebra_to_horse,
            discriminator_horse=discriminator_horse,
            discriminator_zebra=discriminator_zebra,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            scheduler_g=scheduler_g,
            scheduler_d=scheduler_d,
        )
        print(f"Resumed training from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        for iteration, batch in enumerate(train_loader, start=1):
            real_horse = batch["A"].to(device)
            real_zebra = batch["B"].to(device)

            # 2. Update generators while discriminators are frozen.
            set_requires_grad([discriminator_horse, discriminator_zebra], False)
            set_requires_grad([generator_horse_to_zebra, generator_zebra_to_horse], True)
            optimizer_g.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                # 1. Generate fake images for both translation directions.
                fake_zebra = generator_horse_to_zebra(real_horse)
                fake_horse = generator_zebra_to_horse(real_zebra)
                
                reconstructed_horse = generator_zebra_to_horse(fake_zebra)
                reconstructed_zebra = generator_horse_to_zebra(fake_horse)
                identity_horse = generator_zebra_to_horse(real_horse)
                identity_zebra = generator_horse_to_zebra(real_zebra)

                fake_horse_prediction = discriminator_horse(fake_horse)
                fake_zebra_prediction = discriminator_zebra(fake_zebra)
                generator_losses = loss_fn.total_generator_loss(
                    fake_a_prediction=fake_horse_prediction,
                    fake_b_prediction=fake_zebra_prediction,
                    real_a=real_horse,
                    real_b=real_zebra,
                    reconstructed_a=reconstructed_horse,
                    reconstructed_b=reconstructed_zebra,
                    identity_a=identity_horse,
                    identity_b=identity_zebra,
                )

            scaler_g.scale(generator_losses["total"]).backward()
            scaler_g.step(optimizer_g)
            scaler_g.update()

            # 3. Update discriminators using replay buffers while generators are frozen.
            set_requires_grad([generator_horse_to_zebra, generator_zebra_to_horse], False)
            set_requires_grad([discriminator_horse, discriminator_zebra], True)
            optimizer_d.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                buffered_fake_horse = fake_horse_buffer.push_and_pop(fake_horse).to(device)
                buffered_fake_zebra = fake_zebra_buffer.push_and_pop(fake_zebra).to(device)

                discriminator_horse_loss = loss_fn.discriminator_loss(
                    real_prediction=discriminator_horse(real_horse),
                    fake_prediction=discriminator_horse(buffered_fake_horse),
                )
                discriminator_zebra_loss = loss_fn.discriminator_loss(
                    real_prediction=discriminator_zebra(real_zebra),
                    fake_prediction=discriminator_zebra(buffered_fake_zebra),
                )
                discriminator_loss = 0.5 * (discriminator_horse_loss + discriminator_zebra_loss)

            scaler_d.scale(discriminator_loss).backward()
            scaler_d.step(optimizer_d)
            scaler_d.update()

            set_requires_grad([generator_horse_to_zebra, generator_zebra_to_horse], True)

            if iteration % args.log_interval == 0:
                print(
                    f"Epoch [{epoch + 1}/{args.epochs}] "
                    f"Iter [{iteration}/{len(train_loader)}] "
                    f"G: {generator_losses['total'].item():.4f} "
                    f"Adv: {generator_losses['adversarial'].item():.4f} "
                    f"Cycle: {generator_losses['cycle'].item():.4f} "
                    f"Id: {generator_losses['identity'].item():.4f} "
                    f"D: {discriminator_loss.item():.4f}"
                )

        scheduler_g.step()
        scheduler_d.step()

        save_sample_outputs(
            epoch=epoch + 1,
            real_horse=sample_horse,
            real_zebra=sample_zebra,
            generator_horse_to_zebra=generator_horse_to_zebra,
            generator_zebra_to_horse=generator_zebra_to_horse,
            output_dir=results_dir,
        )

        if (epoch + 1) % args.checkpoint_interval == 0:
            checkpoint_path = checkpoint_dir / f"baseline_epoch_{epoch + 1:03d}.pth"
            save_training_checkpoint(
                file_path=checkpoint_path,
                epoch=epoch,
                generator_horse_to_zebra=generator_horse_to_zebra,
                generator_zebra_to_horse=generator_zebra_to_horse,
                discriminator_horse=discriminator_horse,
                discriminator_zebra=discriminator_zebra,
                optimizer_g=optimizer_g,
                optimizer_d=optimizer_d,
                scheduler_g=scheduler_g,
                scheduler_d=scheduler_d,
            )
            print(f"Saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
