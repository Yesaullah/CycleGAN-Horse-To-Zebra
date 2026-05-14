import torch
import torch.nn as nn

from config import image_size


class PatchGANDiscriminator(nn.Module):
    """70x70 PatchGAN discriminator used by CycleGAN.

    The discriminator produces a spatial map of real/fake predictions. Each
    output value judges a local image patch rather than the whole image.
    """

    def __init__(self, in_channels=3):
        super().__init__()

        self.model = nn.Sequential(
            # C64: first feature extractor. No InstanceNorm in the first layer.
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # C128: downsample and increase feature depth.
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128, affine=False, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            # C256: keep expanding the receptive field over local patches.
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256, affine=False, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            # C512: stride 1 preserves a denser patch grid near the output.
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(512, affine=False, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            # Final prediction layer: 1-channel patch map, no sigmoid for LSGAN loss.
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x):
        return self.model(x)


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def print_model_summary():
    model = PatchGANDiscriminator()
    sample = torch.randn(1, 3, image_size, image_size)

    with torch.no_grad():
        output = model(sample)

    print(model)
    print(f"Input shape:  {tuple(sample.shape)}")
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"Trainable parameters: {count_parameters(model):,}")


if __name__ == "__main__":
    print_model_summary()

