__all__ = ["Generator", "PatchGANDiscriminator", "ResidualBlock"]


def __getattr__(name):
    if name in {"Generator", "ResidualBlock"}:
        from .generator import Generator, ResidualBlock

        return {"Generator": Generator, "ResidualBlock": ResidualBlock}[name]
    if name == "PatchGANDiscriminator":
        from .discriminator import PatchGANDiscriminator

        return PatchGANDiscriminator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
