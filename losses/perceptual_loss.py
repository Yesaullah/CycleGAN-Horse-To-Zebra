import torch
import torch.nn as nn
from torchvision.models import VGG16_Weights, vgg16


VGG_LAYER_INDICES = {
    "relu1_2": 3,
    "relu2_2": 8,
    "relu3_2": 13,
    "relu4_2": 20,
}


class VGG16FeatureExtractor(nn.Module):
    """Frozen VGG-16 feature extractor for perceptual comparisons."""

    def __init__(self, layers=("relu1_2", "relu2_2", "relu3_2", "relu4_2"), pretrained=True):
        super().__init__()
        self.layers = self._normalize_layers(layers)
        max_layer_index = max(VGG_LAYER_INDICES[layer] for layer in self.layers)

        weights = VGG16_Weights.DEFAULT if pretrained else None
        vgg = vgg16(weights=weights).features[: max_layer_index + 1]

        # VGG is used as a fixed perceptual metric, not as a trainable network.
        for parameter in vgg.parameters():
            parameter.requires_grad = False

        self.features = vgg.eval()

    @staticmethod
    def _normalize_layers(layers):
        if isinstance(layers, str):
            if layers == "all":
                return tuple(VGG_LAYER_INDICES.keys())
            layers = (layers,)

        invalid_layers = set(layers) - set(VGG_LAYER_INDICES)
        if invalid_layers:
            valid = ", ".join(VGG_LAYER_INDICES)
            raise ValueError(f"Invalid VGG layer(s): {invalid_layers}. Valid layers: {valid}")

        return tuple(layers)

    def forward(self, x):
        extracted_features = {}
        layer_indices = {
            VGG_LAYER_INDICES[layer]: layer
            for layer in self.layers
        }

        for index, layer in enumerate(self.features):
            x = layer(x)
            if index in layer_indices:
                extracted_features[layer_indices[index]] = x

        return extracted_features


class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss for replacing pixel L1 cycle consistency.

    Pixel L1 forces exact per-pixel reconstruction, which can over-penalize
    small shifts and encourage blurry averages. VGG feature maps compare images
    in a learned representation: early layers capture edges and textures, while
    deeper layers capture larger structures and object-level layout.
    """

    def __init__(
        self,
        vgg_layer="relu3_2",
        layer_weights=None,
        pretrained=True,
        device="cuda",
    ):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.feature_extractor = VGG16FeatureExtractor(
            layers=vgg_layer,
            pretrained=pretrained,
        ).to(self.device)
        self.feature_extractor.eval()
        self.criterion = nn.MSELoss()

        self.layers = self.feature_extractor.layers
        self.layer_weights = self._build_layer_weights(layer_weights)

        # ImageNet normalization expected by pretrained VGG-16.
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1),
        )

    def _build_layer_weights(self, layer_weights):
        if layer_weights is None:
            return {layer: 1.0 for layer in self.layers}

        if isinstance(layer_weights, dict):
            return {layer: float(layer_weights.get(layer, 1.0)) for layer in self.layers}

        if len(layer_weights) != len(self.layers):
            raise ValueError("layer_weights must match the number of selected VGG layers")

        return {
            layer: float(weight)
            for layer, weight in zip(self.layers, layer_weights)
        }

    def _normalize_for_vgg(self, images):
        # CycleGAN images are normalized to [-1, 1]; VGG expects ImageNet-normalized [0, 1].
        images = images.to(self.device)
        images = images.mul(0.5).add(0.5).clamp(0.0, 1.0)
        return (images - self.mean) / self.std

    def forward(self, reconstructed_images, original_images):
        reconstructed_images = self._normalize_for_vgg(reconstructed_images)
        original_images = self._normalize_for_vgg(original_images)

        reconstructed_features = self.feature_extractor(reconstructed_images)

        # Original features are targets for the loss, so they do not need gradients.
        with torch.no_grad():
            original_features = self.feature_extractor(original_images)

        loss = reconstructed_images.new_tensor(0.0)
        for layer in self.layers:
            layer_loss = self.criterion(
                reconstructed_features[layer],
                original_features[layer],
            )
            loss = loss + self.layer_weights[layer] * layer_loss

        return loss

