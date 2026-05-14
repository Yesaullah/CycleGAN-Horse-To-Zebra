import torch
import torch.nn as nn

from config import image_size, num_residual_blocks


class ResidualBlock(nn.Module):
    """Residual block used in the Johnson et al. CycleGAN generator."""

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            # Reflection padding preserves spatial size and reduces edge artifacts.
            nn.ReflectionPad2d(1),
            # First 3x3 convolution learns residual features.
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(channels, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True),
            # Second 3x3 convolution maps the residual features back to the input depth.
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(channels, affine=False, track_running_stats=False),
        )

    def forward(self, x):
        # Skip connection keeps content structure while the block learns changes.
        return x + self.block(x)


class Generator(nn.Module):
    """CycleGAN generator: c7s1-64, d128, d256, 9 residual blocks, u128, u64, c7s1-3."""

    def __init__(self, in_channels=3, out_channels=3, residual_blocks=num_residual_blocks):
        super().__init__()

        layers = [
            # c7s1-64: large receptive field at full resolution.
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(64, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True),
            # d128: first downsampling layer halves resolution and doubles channels.
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True),
            # d256: second downsampling layer builds compact high-level features.
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True),
        ]

        # 9 residual blocks transform features at 64x64 for 256x256 inputs.
        layers.extend(ResidualBlock(256) for _ in range(residual_blocks))

        layers.extend(
            [
                # u128: fractionally-strided convolution upsamples back toward image space.
                nn.ConvTranspose2d(
                    256,
                    128,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=False,
                ),
                nn.InstanceNorm2d(128, affine=False, track_running_stats=False),
                nn.ReLU(inplace=True),
                # u64: second upsampling layer restores the original spatial resolution.
                nn.ConvTranspose2d(
                    128,
                    64,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=False,
                ),
                nn.InstanceNorm2d(64, affine=False, track_running_stats=False),
                nn.ReLU(inplace=True),
                # c7s1-3: final RGB projection with values in [-1, 1].
                nn.ReflectionPad2d(3),
                nn.Conv2d(64, out_channels, kernel_size=7, stride=1, padding=0),
                nn.Tanh(),
            ]
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def print_model_summary():
    model = Generator()
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

