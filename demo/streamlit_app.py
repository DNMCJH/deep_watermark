"""Streamlit demo: embed and extract watermarks interactively."""

import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from models.encoder import WatermarkEncoder
from models.decoder import WatermarkDecoder
from utils.image_utils import tensor_to_image
from data.watermark_generator import generate_watermark


@st.cache_resource
def load_model(checkpoint_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]

    encoder = WatermarkEncoder(
        watermark_length=config["watermark_length"],
        image_size=config["image_size"],
        residual_scale=config.get("residual_scale", 0.1),
    ).to(device)
    decoder = WatermarkDecoder(watermark_length=config["watermark_length"]).to(device)

    encoder.load_state_dict(ckpt["encoder_state_dict"])
    decoder.load_state_dict(ckpt["decoder_state_dict"])
    encoder.eval()
    decoder.eval()
    return encoder, decoder, config, device


def main():
    st.title("Deep Watermark Demo")
    st.sidebar.header("Settings")

    ckpt_path = st.sidebar.text_input("Checkpoint path", "experiments/latest/checkpoint.pt")

    try:
        encoder, decoder, config, device = load_model(ckpt_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return

    image_size = config["image_size"]
    wm_length = config["watermark_length"]

    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded is None:
        return

    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Original Image", use_container_width=True)

    # Watermark input
    wm_input = st.text_input(
        f"Watermark ({wm_length} bits, e.g. 10110...)",
        value="".join(["1", "0"] * (wm_length // 2)),
    )
    wm_bits = [int(b) for b in wm_input if b in ("0", "1")]
    if len(wm_bits) != wm_length:
        st.warning(f"Need exactly {wm_length} bits, got {len(wm_bits)}")
        return

    if st.button("Embed Watermark"):
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        img_tensor = transform(img).unsqueeze(0).to(device)
        wm_tensor = torch.tensor(wm_bits, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            encoded = encoder(img_tensor, wm_tensor)
            recovered = decoder(encoded)

        col1, col2 = st.columns(2)
        with col1:
            st.image(tensor_to_image(img_tensor), caption="Original")
        with col2:
            st.image(tensor_to_image(encoded), caption="Watermarked")

        pred_bits = (recovered.squeeze() > 0.5).int().cpu().tolist()
        accuracy = sum(a == b for a, b in zip(pred_bits, wm_bits)) / wm_length
        st.metric("Bit Accuracy", f"{accuracy:.2%}")
        st.text(f"Recovered: {''.join(map(str, pred_bits))}")


if __name__ == "__main__":
    main()
