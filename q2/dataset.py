"""Unpaired photo/sketch dataset for the Kaggle 'Person Face Sketches' dataset."""
import argparse
import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SUBSAMPLE_SEED = 0
RESIZE_MARGIN = 1.12  # paper: resize 256 -> 286, random-crop back to 256


def list_images(root):
    return sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in IMG_EXTS)


def split_domains(root):
    """Photos vs sketches by folder name: any parent dir containing 'sketch' is domain B."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"data dir not found: {root}")
    photos, sketches = [], []
    for p in list_images(root):
        is_sketch_dir = any("sketch" in part.lower() for part in p.relative_to(root).parts[:-1])
        (sketches if is_sketch_dir else photos).append(p)
    if not photos or not sketches:
        raise ValueError(f"need both photos and sketches under {root} "
                         f"(found {len(photos)} photos, {len(sketches)} sketches); "
                         "use --photo_dir/--sketch_dir to point at them explicitly")
    return photos, sketches


def build_transform(image_size, train):
    if train:
        resize = [transforms.Resize(int(image_size * RESIZE_MARGIN)),
                  transforms.RandomCrop(image_size), transforms.RandomHorizontalFlip()]
    else:
        resize = [transforms.Resize(image_size), transforms.CenterCrop(image_size)]
    return transforms.Compose(resize + [transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])


def subsample(paths, n, seed=SUBSAMPLE_SEED):
    """Seeded random subset, so --max_samples is reproducible and doesn't just take the first folder."""
    if not n or n >= len(paths):
        return paths
    return sorted(random.Random(seed).sample(paths, n))


class UnpairedDataset(Dataset):
    """Returns (photo, sketch); the sketch is sampled at random so pairs are never aligned."""

    def __init__(self, photos, sketches, image_size=256, train=True, max_samples=None):
        self.photos = subsample(photos, max_samples)
        self.sketches = subsample(sketches, max_samples)
        self.transform = build_transform(image_size, train)
        self.train = train

    def __len__(self):
        return max(len(self.photos), len(self.sketches))

    def _load(self, path):
        return self.transform(Image.open(path).convert("RGB"))

    def __getitem__(self, i):
        photo = self.photos[i % len(self.photos)]
        j = random.randrange(len(self.sketches)) if self.train else i % len(self.sketches)
        return self._load(photo), self._load(self.sketches[j])


def resolve_domains(data_dir, photo_dir=None, sketch_dir=None):
    if photo_dir and sketch_dir:
        photos, sketches = list_images(photo_dir), list_images(sketch_dir)
        if not photos or not sketches:
            raise ValueError(f"no images in {photo_dir} or {sketch_dir}")
        return photos, sketches
    return split_domains(data_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect the photo/sketch split of a dataset folder")
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--photo_dir")
    parser.add_argument("--sketch_dir")
    args = parser.parse_args()
    a, b = resolve_domains(args.data_dir, args.photo_dir, args.sketch_dir)
    print(f"photos: {len(a)}  e.g. {a[0]}\nsketches: {len(b)}  e.g. {b[0]}")
