# Question-2: CycleGAN for Person Face Sketches

CycleGAN (Zhu et al., 2017) for unpaired translation between face **photos (A)** and **sketches (B)**, trained on
[Person Face Sketches](https://www.kaggle.com/datasets/almightyj/person-face-sketches). There's a Flask UI that
auto-detects whether an input is a sketch or a photo and converts it.

| File | Purpose |
|---|---|
| `models.py` | ResNet generator (9 blocks) + 70×70 PatchGAN discriminator, N(0, 0.02) init |
| `dataset.py` | Unpaired loader; folders whose name contains `sketch` are domain B, all other images are A |
| `train.py` | End-to-end training, checkpoint every epoch, auto-resume |
| `inference.py` | Load generators, sketch/photo detector, single-image CLI |
| `app.py` + `templates/` + `static/` | Flask UI: upload, camera capture, live mode |
| `run_local.sh` | Launches the UI on your machine with the checkpoint you trained on Colab |
| `colab_run.ipynb` | Colab notebook covering the whole pipeline |
| `test_cyclegan.py` | Smoke tests (`python -m pytest -q`) |

## Method (as in the paper)
- **Losses:** LSGAN adversarial loss (MSE), cycle loss L1 × λ<sub>cyc</sub>=10, identity loss L1 × 0.5·λ<sub>cyc</sub>.
- **Optimiser:** Adam, lr 2e-4, β=(0.5, 0.999). LR stays constant for the first half of training, then decays linearly to 0.
- **Discriminator:** trained on a 50-image history pool of generated fakes, with its loss halved.
- **Augmentation:** resize to 1.12× the training size, random crop, horizontal flip.

## Google Colab

The easiest way is to open **`colab_run.ipynb`** and run it top to bottom. It goes through setup, a hyperparameter sweep, full training, monitoring, a test of both directions, and exporting the model. The web UI runs on your own machine (see [Run the UI locally](#run-the-ui-locally)). The same steps as plain cells:

```python
# 1. Code + Drive (checkpoints go to Drive so they survive disconnects)
from google.colab import drive; drive.mount('/content/drive')
%cd /content/drive/MyDrive/as2/q2      # upload this q2/ folder there first
!pip -q install -r requirements.txt
```

```python
# 2. Dataset (kagglehub is preinstalled on Colab; it asks for your Kaggle credentials if needed)
import kagglehub
DATA = kagglehub.dataset_download("almightyj/person-face-sketches")
print(DATA)
!python dataset.py --data_dir "$DATA"   # prints photo/sketch counts and an example path from each
```
If the auto-split picks the wrong folders, pass them yourself: `--photo_dir <path> --sketch_dir <path>`.
The dataset may also have `train/`/`test/` subfolders. If so, point `--data_dir` at the train one.

```python
# 3. Train. Run the same cell again after a disconnect: it resumes from checkpoints/latest.pth
!python train.py --data_dir "$DATA" --checkpoint_dir /content/drive/MyDrive/cyclegan_ckpt \
    --epochs 30 --max_samples 3000 --image_size 256 --batch_size 1 --amp --num_workers 2
```
On a T4, expect roughly 12–18 minutes per epoch at these settings. If the dataset has a `train/` folder, use that as `--data_dir`; the notebook does this automatically.
Faster or lower-memory option: `--image_size 128 --n_blocks 6 --batch_size 4 --max_samples 3000`.

```python
# 4. Monitor training
import pandas as pd; pd.read_csv('/content/drive/MyDrive/cyclegan_ckpt/history.csv').plot(x='epoch', y=['loss_G','loss_D','loss_cycle'])
from IPython.display import Image; Image('/content/drive/MyDrive/cyclegan_ckpt/samples/epoch_030.png')
```
Rows in each sample grid: photo → its generated sketch, then sketch → its generated photo.

```python
# 5. Export the generators (about 91 MB) for the local UI and download them
import torch
CKPT = '/content/drive/MyDrive/cyclegan_ckpt'
s = torch.load(f'{CKPT}/latest.pth', map_location='cpu', weights_only=False)
torch.save({k: s[k] for k in ('epoch', 'args', 'G_AB', 'G_BA')}, f'{CKPT}/cyclegan_final.pth')
from google.colab import files; files.download(f'{CKPT}/cyclegan_final.pth')
```

## Run the UI locally
Training happens on Colab; the web UI runs on your computer.

1. Get `cyclegan_final.pth` from the export step: the notebook downloads it, and a copy is also in your Drive checkpoint folder.
2. Put it in `q2/checkpoints/`. That folder is git-ignored, so the weights never end up in the repo.
3. Run the launcher:
   ```bash
   cd q2
   ./run_local.sh                               # uses checkpoints/cyclegan_final.pth
   ./run_local.sh checkpoints/epoch_030.pth     # or any other checkpoint
   PORT=8000 ./run_local.sh                     # different port
   ```
   The first run creates `q2/.venv` and installs `requirements.txt` (torch is large, so this takes a few minutes). Later runs start straight away.
4. Open http://127.0.0.1:5000. You can upload an image, or use the camera (browsers allow it on localhost). On a Mac it uses the Apple GPU (MPS) automatically.

The image size and generator depth are read from the checkpoint, so no extra flags are needed. `latest.pth` and any `epoch_XXX.pth` work too; `cyclegan_final.pth` is just the smallest.


## What gets saved in the checkpoint dir
- `latest.pth`: everything needed to resume (all 4 networks, both optimisers, LR schedulers, AMP scaler, epoch). It's written every epoch, and written atomically so a disconnect mid-save can't corrupt it.
- `epoch_XXX.pth`: generator weights only, about 90 MB each, written every 5 epochs by default. Change the interval with `--snapshot_every N`, or turn them off with `0`. Overwritten Drive files still count toward your quota until you empty the Trash.
- `history.csv`: average losses, LR and time for each epoch.
- `samples/epoch_XXX.png`: a fixed preview grid for each epoch.

## CLI reference
```bash
python train.py --help
python train.py --resume none ...                 # start fresh; ignores latest.pth
python train.py --resume path/to/latest.pth ...   # resume from a specific file
python train.py --epochs 60 ...                   # continue a finished run to 60 epochs; LR follows the new schedule
python inference.py --checkpoint ckpt/latest.pth --input face.jpg --output sketch.png   # --direction auto|photo2sketch|sketch2photo
./run_local.sh checkpoints/cyclegan_final.pth                                         # local UI (venv + app.py)
python -m pytest -q test_cyclegan.py
```

## Sketch vs photo detection
The detector is a heuristic in `inference.is_sketch`. An image counts as a sketch when its average colour saturation
is below 0.12 and more than 35% of its pixels are bright. If it gets an image wrong, the UI has a manual direction
selector that overrides it.

## Results (fill in after the Colab run)
> Nothing below has been measured yet. Copy it from the notebook outputs.

**Hyperparameter tuning:** the table from notebook section 4 (final-epoch `loss_cycle` for each lr/batch size), plus which setting you picked and why.

**Training curves:** the loss plot and the last rows of `history.csv` from section 6. Note how many epochs you trained and roughly how long each took.

**Qualitative results:** a sample grid from the final epoch and the two section 7 outputs (photo → sketch, sketch → photo), with a short comment on what works and what fails.

**Auto-detection:** the accuracy printed by the detector cell in section 7.

**UI:** a screenshot of the web UI converting an uploaded image and a camera frame.
