import sys
from pathlib import Path

import gradio as gr
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.generator import Generator

CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "baseline" / "checkpoint_epoch_200.pth"
DEVICE = torch.device("cpu")

transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def load_models():
    gen_h2z = Generator().to(DEVICE)
    gen_z2h = Generator().to(DEVICE)

    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    gen_h2z.load_state_dict(checkpoint["generator_ab"])
    gen_z2h.load_state_dict(checkpoint["generator_ba"])

    gen_h2z.eval()
    gen_z2h.eval()
    return gen_h2z, gen_z2h


print("Loading model weights...")
gen_h2z, gen_z2h = load_models()
print("Models ready.")


def translate(image: Image.Image, direction: str) -> Image.Image:
    if image is None:
        raise gr.Error("Please upload an image first.")

    tensor = transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)

    generator = gen_h2z if direction == "Horse → Zebra" else gen_z2h

    with torch.no_grad():
        output = generator(tensor)

    output = output.squeeze(0).mul(0.5).add(0.5).clamp(0, 1)
    output = output.permute(1, 2, 0).numpy()
    output = (output * 255).astype("uint8")
    return Image.fromarray(output)


with gr.Blocks(title="CycleGAN Horse ↔ Zebra") as demo:
    gr.Markdown(
        """
        # CycleGAN — Horse ↔ Zebra Translation
        **Baseline model · Epoch 200 checkpoint**

        Upload a horse or zebra image, choose the translation direction, and click **Translate**.
        """
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Input Image")
            direction = gr.Radio(
                choices=["Horse → Zebra", "Zebra → Horse"],
                value="Horse → Zebra",
                label="Translation Direction",
            )
            btn = gr.Button("Translate", variant="primary")

        with gr.Column():
            output_image = gr.Image(type="pil", label="Translated Output")

    btn.click(fn=translate, inputs=[input_image, direction], outputs=output_image)


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
