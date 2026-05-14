from torch.optim.lr_scheduler import LambdaLR

from config import decay_epoch, n_epochs


def lambda_rule(epoch, start_decay_epoch=decay_epoch, total_epochs=n_epochs):
    """Return LR multiplier for CycleGAN's linear decay schedule.

    CycleGAN keeps the learning rate fixed early so both generators and
    discriminators can learn a useful mapping. The later linear decay lets the
    adversarial training settle instead of continuing to make large updates.
    """
    if epoch < start_decay_epoch:
        return 1.0

    decay_epochs = total_epochs - start_decay_epoch
    if decay_epochs <= 0:
        return 0.0

    return max(0.0, 1.0 - (epoch - start_decay_epoch + 1) / decay_epochs)


def get_linear_decay_scheduler(optimizer, start_decay_epoch=decay_epoch, total_epochs=n_epochs):
    """Create a PyTorch LambdaLR scheduler for 100 constant + 100 decay epochs."""
    return LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: lambda_rule(
            epoch,
            start_decay_epoch=start_decay_epoch,
            total_epochs=total_epochs,
        ),
    )

