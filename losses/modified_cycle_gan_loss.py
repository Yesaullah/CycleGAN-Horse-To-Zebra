from config import lambda_cycle, lambda_identity
from losses.cycle_gan_loss import CycleGANLoss
from losses.perceptual_loss import PerceptualLoss


class ModifiedCycleGANLoss(CycleGANLoss):
    """CycleGAN loss where L1 cycle consistency is replaced by perceptual loss."""

    def __init__(
        self,
        lambda_cycle_weight=lambda_cycle,
        lambda_identity_weight=lambda_identity,
        vgg_layer="relu3_2",
        perceptual_layer_weights=None,
        pretrained_vgg=True,
        device="cuda",
    ):
        super().__init__(
            lambda_cycle_weight=lambda_cycle_weight,
            lambda_identity_weight=lambda_identity_weight,
            device=device,
        )
        self.perceptual_loss = PerceptualLoss(
            vgg_layer=vgg_layer,
            layer_weights=perceptual_layer_weights,
            pretrained=pretrained_vgg,
            device=device,
        )

    def cycle_consistency_loss(self, real_a, real_b, reconstructed_a, reconstructed_b):
        """Perceptual cycle loss replacing pixel L1 reconstruction.

        Forward cycle: compare F(G(x)) with x in VGG feature space.
        Backward cycle: compare G(F(y)) with y in VGG feature space.
        """
        forward_cycle_loss = self.perceptual_loss(reconstructed_a, real_a)
        backward_cycle_loss = self.perceptual_loss(reconstructed_b, real_b)
        total_cycle_loss = forward_cycle_loss + backward_cycle_loss

        return self.lambda_cycle * total_cycle_loss

