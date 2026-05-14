import random

import torch


class ReplayBuffer:
    """Image replay buffer for CycleGAN discriminator training.

    Feeding discriminators only the newest generated images can make the two
    adversarial games chase each other and oscillate. Reusing a small history of
    generated images smooths the discriminator updates and improves stability.
    """

    def __init__(self, max_size=50):
        if max_size <= 0:
            raise ValueError("ReplayBuffer max_size must be greater than 0")

        self.max_size = max_size
        self.images = []

    def __len__(self):
        return len(self.images)

    def push_and_pop(self, generated_images):
        """Return current or historical generated images.

        If the buffer is not full, each image is added and returned. Once full,
        each queried image has a 50% chance of being swapped with a stored image;
        otherwise the current image is returned directly.
        """
        returned_images = []

        for image in generated_images:
            image = image.detach().clone().unsqueeze(0)

            if len(self.images) < self.max_size:
                self.images.append(image)
                returned_images.append(image)
                continue

            if random.random() > 0.5:
                random_index = random.randrange(self.max_size)
                old_image = self.images[random_index].clone()
                self.images[random_index] = image
                returned_images.append(old_image)
            else:
                returned_images.append(image)

        return torch.cat(returned_images, dim=0)

