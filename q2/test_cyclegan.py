"""Smoke tests for the CycleGAN pipeline. Run: python -m pytest -q test_cyclegan.py"""
import io
import json

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw


def _photo(size=48, seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8))


def _sketch(size=48):
    img = Image.new("RGB", (size, size), "white")
    ImageDraw.Draw(img).ellipse((8, 8, size - 8, size - 8), outline="black", width=2)
    return img


def _make_dataset(root, n=4):
    for sub in ("photos", "sketches"):
        (root / sub).mkdir(parents=True)
        for i in range(n):
            (_photo(seed=i) if sub == "photos" else _sketch()).save(root / sub / f"{i}.png")


def test_generator_preserves_shape_and_range():
    from models import Generator
    g = Generator(n_blocks=2)
    out = g(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 3, 64, 64)
    assert out.min() >= -1 and out.max() <= 1


def test_discriminator_outputs_patch_map():
    from models import Discriminator
    out = Discriminator()(torch.randn(1, 3, 64, 64))
    assert out.dim() == 4 and out.shape[1] == 1 and out.shape[-1] > 1


def test_split_domains_by_folder_name(tmp_path):
    from dataset import split_domains
    _make_dataset(tmp_path)
    photos, sketches = split_domains(tmp_path)
    assert len(photos) == 4 and len(sketches) == 4
    assert all("sketch" in str(p) for p in sketches)


def test_split_domains_rejects_missing_domain(tmp_path):
    from dataset import split_domains
    (tmp_path / "photos").mkdir()
    _photo().save(tmp_path / "photos" / "a.png")
    with pytest.raises(ValueError):
        split_domains(tmp_path)


def test_split_domains_rejects_missing_dir(tmp_path):
    from dataset import split_domains
    with pytest.raises(FileNotFoundError):
        split_domains(tmp_path / "nope")


def test_is_sketch_heuristic():
    from inference import is_sketch
    assert is_sketch(_sketch(128))
    assert not is_sketch(_photo(128))


def test_train_saves_every_epoch_and_resumes(tmp_path):
    import train
    data, ckpt = tmp_path / "data", tmp_path / "ckpt"
    _make_dataset(data)
    common = ["--data_dir", str(data), "--checkpoint_dir", str(ckpt), "--image_size", "32",
              "--n_blocks", "1", "--batch_size", "2", "--num_workers", "0", "--device", "cpu", "--snapshot_every", "1"]
    train.main(common + ["--epochs", "1"])
    assert (ckpt / "latest.pth").exists() and (ckpt / "epoch_001.pth").exists()
    assert torch.load(ckpt / "latest.pth", weights_only=False)["epoch"] == 1

    train.main(common + ["--epochs", "2"])  # resumes from latest.pth, trains only epoch 2
    state = torch.load(ckpt / "latest.pth", weights_only=False)
    assert state["epoch"] == 2
    assert len((ckpt / "history.csv").read_text().strip().splitlines()) == 3  # header + 2 epochs


@pytest.fixture
def client():
    from app import create_app
    from models import Generator
    gens = {"photo2sketch": Generator(n_blocks=1).eval(), "sketch2photo": Generator(n_blocks=1).eval()}
    return create_app(gens, image_size=32, device="cpu").test_client()


def test_convert_requires_file(client):
    assert client.post("/convert").status_code == 400


def test_convert_rejects_non_image(client):
    resp = client.post("/convert", data={"image": (io.BytesIO(b"not an image"), "x.png")})
    assert resp.status_code == 400


def test_convert_rejects_bad_mode(client):
    buf = io.BytesIO()
    _photo().save(buf, "PNG")
    buf.seek(0)
    resp = client.post("/convert", data={"image": (buf, "x.png"), "mode": "evil"})
    assert resp.status_code == 400


def test_convert_auto_detects_sketch(client):
    buf = io.BytesIO()
    _sketch(128).save(buf, "PNG")
    buf.seek(0)
    resp = client.post("/convert", data={"image": (buf, "s.png"), "mode": "auto"})
    body = json.loads(resp.data)
    assert resp.status_code == 200
    assert body["detected"] == "sketch" and body["direction"] == "sketch2photo"
    assert body["image"].startswith("data:image/png;base64,")
    assert resp.headers["X-Content-Type-Options"] == "nosniff"


def test_lr_schedule_decays_to_near_zero():
    import train
    args = train.parse_args(["--epochs", "10"])
    f = train.make_lr_lambda(args)
    assert f(0) == 1.0 and f(args.decay_epoch - 1) == 1.0
    assert f(9) <= 1 / (10 - args.decay_epoch + 1) + 1e-9
    assert all(f(e + 1) <= f(e) for e in range(9))


def test_max_samples_draws_across_whole_list():
    from dataset import UnpairedDataset
    paths = [f"{i:04d}" for i in range(1000)]
    ds = UnpairedDataset(paths, paths, image_size=32, max_samples=50)
    assert len(ds.photos) == 50 and max(ds.photos) > "0100"  # not just the first 50 sorted
    assert UnpairedDataset(paths, paths, image_size=32, max_samples=50).photos == ds.photos  # deterministic


def test_resume_with_different_architecture_fails_clearly(tmp_path):
    import train
    data, ckpt = tmp_path / "data", tmp_path / "ckpt"
    _make_dataset(data)
    common = ["--data_dir", str(data), "--checkpoint_dir", str(ckpt), "--image_size", "32",
              "--batch_size", "2", "--num_workers", "0", "--device", "cpu", "--epochs", "2"]
    train.main(common + ["--n_blocks", "1", "--epochs", "1"])
    with pytest.raises(ValueError, match="n_blocks"):
        train.main(common + ["--n_blocks", "2"])
