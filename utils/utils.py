from pathlib import Path

import torch


def save_checkpoint(
    file_path,
    generator_ab,
    generator_ba,
    discriminator_a,
    discriminator_b,
    optimizer_g,
    optimizer_d,
    epoch,
):
    """Save CycleGAN model and optimizer states."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "generator_ab": generator_ab.state_dict(),
        "generator_ba": generator_ba.state_dict(),
        "discriminator_a": discriminator_a.state_dict(),
        "discriminator_b": discriminator_b.state_dict(),
        "optimizer_g": optimizer_g.state_dict(),
        "optimizer_d": optimizer_d.state_dict(),
    }
    torch.save(checkpoint, file_path)


def load_checkpoint(
    file_path,
    generator_ab,
    generator_ba,
    discriminator_a,
    discriminator_b,
    optimizer_g=None,
    optimizer_d=None,
    device="cuda",
):
    """Load CycleGAN model states and optionally optimizer states."""
    checkpoint = torch.load(file_path, map_location=device)

    generator_ab.load_state_dict(checkpoint["generator_ab"])
    generator_ba.load_state_dict(checkpoint["generator_ba"])
    discriminator_a.load_state_dict(checkpoint["discriminator_a"])
    discriminator_b.load_state_dict(checkpoint["discriminator_b"])

    if optimizer_g is not None and "optimizer_g" in checkpoint:
        optimizer_g.load_state_dict(checkpoint["optimizer_g"])
    if optimizer_d is not None and "optimizer_d" in checkpoint:
        optimizer_d.load_state_dict(checkpoint["optimizer_d"])

    return checkpoint.get("epoch", 0)


def denormalize(images):
    """Convert tensors normalized to [-1, 1] back to [0, 1]."""
    return images.mul(0.5).add(0.5).clamp(0.0, 1.0)

