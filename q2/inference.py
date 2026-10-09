"""Load trained generators, detect sketch vs photo, and translate a single image."""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dataset import build_transform
from models import Generator

# ponytail: colour/brightness heuristic; sketches are near-grey on a mostly white page.
# Upgrade path if it misfires on grey photos: train a tiny CNN on the two dataset domains.
SKETCH_MAX_SATURATION = 0.12
SKETCH_MIN_BRIGHT_FRACTION = 0.35
BRIGHT_LEVEL = 0.75
DIRECTIONS = ("photo2sketch", "sketch2photo")


def pick_device(name="auto"):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def is_sketch(img):
    rgb = np.asarray(img.convert("RGB").resize((128, 128)), dtype=np.float32) / 255.0
    hi, lo = rgb.max(axis=2), rgb.min(axis=2)
    saturation = np.where(hi > 0, (hi - lo) / np.maximum(hi, 1e-6), 0).mean()
    bright = (rgb.mean(axis=2) > BRIGHT_LEVEL).mean()
    return bool(saturation < SKETCH_MAX_SATURATION and bright > SKETCH_MIN_BRIGHT_FRACTION)


def load_generators(checkpoint, device="cpu"):
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    n_blocks = state["args"]["n_blocks"]
    gens = {}
    for direction, key in (("photo2sketch", "G_AB"), ("sketch2photo", "G_BA")):
        g = Generator(n_blocks=n_blocks).to(device)
        g.load_state_dict(state[key])
        gens[direction] = g.eval()
    return gens, state["args"]["image_size"]


@torch.no_grad()
def translate(gen, img, image_size, device="cpu"):
    x = build_transform(image_size, train=False)(img.convert("RGB")).unsqueeze(0).to(device)
    y = gen(x)[0].cpu().clamp(-1, 1).add(1).div(2)
    return Image.fromarray((y.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))


def resolve_direction(img, mode):
    if mode == "auto":
        return "sketch2photo" if is_sketch(img) else "photo2sketch"
    return mode


def main(argv=None):
    parser = argparse.ArgumentParser(description="Translate one image with a trained CycleGAN")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="output.png")
    parser.add_argument("--direction", default="auto", choices=("auto",) + DIRECTIONS)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    device = pick_device(args.device)
    gens, size = load_generators(args.checkpoint, device)
    img = Image.open(args.input)
    direction = resolve_direction(img, args.direction)
    translate(gens[direction], img, size, device).save(args.output)
    print(f"{direction}: {Path(args.input).name} -> {args.output}")


if __name__ == "__main__":
    main()
