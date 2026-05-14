import torch
import torch.nn as nn

from config import lambda_cycle, lambda_identity


class CycleGANLoss:
    """Loss collection for the original CycleGAN objective.

    CycleGAN uses LSGAN adversarial loss instead of the original binary
    cross-entropy GAN loss. Least-squares targets give smoother gradients when
    samples are on the correct side of the discriminator boundary, which helps
    reduce vanishing gradients and makes adversarial training more stable.
    """

    def __init__(
        self,
        lambda_cycle_weight=lambda_cycle,
        lambda_identity_weight=lambda_identity,
        device="cuda",
    ):
        self.lambda_cycle = lambda_cycle_weight
        self.lambda_identity = lambda_identity_weight
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.adversarial_criterion = nn.MSELoss()
        self.reconstruction_criterion = nn.L1Loss()

        self._real_target = None
        self._fake_target = None

    def _target_tensor(self, prediction, is_real):
        """Return a cached target tensor matching the PatchGAN output map."""
        cached_target = self._real_target if is_real else self._fake_target
        target_value = 1.0 if is_real else 0.0

        if (
            cached_target is None
            or cached_target.shape != prediction.shape
            or cached_target.device != prediction.device
            or cached_target.dtype != prediction.dtype
        ):
            cached_target = torch.full_like(
                prediction,
                fill_value=target_value,
                device=prediction.device,
                requires_grad=False,
            )
            if is_real:
                self._real_target = cached_target
            else:
                self._fake_target = cached_target

        return cached_target

    def generator_adversarial_loss(self, fake_prediction):
        """LSGAN generator loss: E[(D(G(x)) - 1)^2]."""
        real_target = self._target_tensor(fake_prediction, is_real=True)
        return self.adversarial_criterion(fake_prediction, real_target)

    def discriminator_loss(self, real_prediction, fake_prediction):
        """LSGAN discriminator loss: E[(D(y)-1)^2] + E[D(G(x))^2]."""
        real_target = self._target_tensor(real_prediction, is_real=True)
        fake_target = self._target_tensor(fake_prediction, is_real=False)

        real_loss = self.adversarial_criterion(real_prediction, real_target)
        fake_loss = self.adversarial_criterion(fake_prediction, fake_target)
        return real_loss + fake_loss

    def cycle_consistency_loss(self, real_a, real_b, reconstructed_a, reconstructed_b):
        """Cycle loss: lambda_cycle * (||F(G(x))-x||_1 + ||G(F(y))-y||_1)."""
        forward_cycle_loss = self.reconstruction_criterion(reconstructed_a, real_a)
        backward_cycle_loss = self.reconstruction_criterion(reconstructed_b, real_b)
        total_cycle_loss = forward_cycle_loss + backward_cycle_loss

        return self.lambda_cycle * total_cycle_loss

    def identity_loss(self, real_a, real_b, identity_a, identity_b):
        """Identity loss: lambda_identity * (||F(x)-x||_1 + ||G(y)-y||_1)."""
        identity_a_loss = self.reconstruction_criterion(identity_a, real_a)
        identity_b_loss = self.reconstruction_criterion(identity_b, real_b)
        total_identity_loss = identity_a_loss + identity_b_loss

        return self.lambda_identity * total_identity_loss

    def total_generator_loss(
        self,
        fake_a_prediction,
        fake_b_prediction,
        real_a,
        real_b,
        reconstructed_a,
        reconstructed_b,
        identity_a,
        identity_b,
    ):
        """Combine adversarial, cycle consistency, and identity losses."""
        adversarial_a_loss = self.generator_adversarial_loss(fake_a_prediction)
        adversarial_b_loss = self.generator_adversarial_loss(fake_b_prediction)
        adversarial_loss = adversarial_a_loss + adversarial_b_loss

        cycle_loss = self.cycle_consistency_loss(
            real_a=real_a,
            real_b=real_b,
            reconstructed_a=reconstructed_a,
            reconstructed_b=reconstructed_b,
        )
        identity_loss = self.identity_loss(
            real_a=real_a,
            real_b=real_b,
            identity_a=identity_a,
            identity_b=identity_b,
        )

        total_loss = adversarial_loss + cycle_loss + identity_loss

        return {
            "total": total_loss,
            "adversarial": adversarial_loss,
            "cycle": cycle_loss,
            "identity": identity_loss,
            "adversarial_a": adversarial_a_loss,
            "adversarial_b": adversarial_b_loss,
        }
