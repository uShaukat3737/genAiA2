"""Train CycleGAN (photo <-> sketch) end-to-end with per-epoch checkpoints and auto-resume.

A = photo, B = sketch. G_AB: photo -> sketch, G_BA: sketch -> photo.
"""
import argparse
import csv
import itertools
import logging
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dataset import UnpairedDataset, resolve_domains
from inference import pick_device
from models import Discriminator, Generator, init_weights

log = logging.getLogger("cyclegan")
NUM_SAMPLE_IMAGES = 4
HISTORY_FIELDS = ["epoch", "loss_G", "loss_D", "loss_gan", "loss_cycle", "loss_idt", "lr", "seconds"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train CycleGAN on Person Face Sketches",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_dir", default="./data", help="root of the dataset; dirs named *sketch* are domain B")
    p.add_argument("--photo_dir", help="explicit photo folder (overrides auto split)")
    p.add_argument("--sketch_dir", help="explicit sketch folder (overrides auto split)")
    p.add_argument("--checkpoint_dir", default="./checkpoints", help="put this on Google Drive in Colab")
    p.add_argument("--epochs", type=int, default=30, help="total epochs (resume continues up to this)")
    p.add_argument("--decay_epoch", type=int, default=None, help="epoch where linear LR decay starts (default epochs/2)")
    p.add_argument("--batch_size", type=int, default=1, help="paper uses 1")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--n_blocks", type=int, default=9, help="ResNet blocks in generator (paper: 9 @256, 6 @128)")
    p.add_argument("--lambda_cycle", type=float, default=10.0)
    p.add_argument("--lambda_identity", type=float, default=0.5, help="scaled by lambda_cycle, 0 disables")
    p.add_argument("--pool_size", type=int, default=50, help="history buffer of fakes for D")
    p.add_argument("--max_samples", type=int, default=None, help="cap images per domain (memory/time)")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--log_every", type=int, default=100, help="iterations between log lines")
    p.add_argument("--snapshot_every", type=int, default=5, help="keep epoch_XXX.pth generator weights every N epochs, 0=off")
    p.add_argument("--resume", default="auto", help="'auto' = checkpoint_dir/latest.pth if present, a path, or 'none'")
    p.add_argument("--amp", action="store_true", help="mixed precision (CUDA only)")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    if args.decay_epoch is None:
        args.decay_epoch = args.epochs // 2
    return args


class ImagePool:
    """Shrivastava et al. buffer: D sees a mix of current and past fakes."""

    def __init__(self, size):
        self.size, self.images = size, []

    def query(self, batch):
        if self.size == 0:
            return batch
        out = []
        for img in batch.detach():
            img = img.unsqueeze(0)
            if len(self.images) < self.size:
                self.images.append(img)
                out.append(img)
            elif random.random() < 0.5:
                i = random.randrange(self.size)
                out.append(self.images[i].clone())
                self.images[i] = img
            else:
                out.append(img)
        return torch.cat(out)


def make_lr_lambda(args):
    """Constant LR for decay_epoch epochs, then linear to ~0 at the last epoch (paper / official code)."""
    def linear_decay(epoch):  # epoch is 0-based
        return 1.0 - max(0, epoch + 1 - args.decay_epoch) / float(args.epochs - args.decay_epoch + 1)
    return linear_decay


def build(args, device):
    nets = {
        "G_AB": Generator(n_blocks=args.n_blocks), "G_BA": Generator(n_blocks=args.n_blocks),
        "D_A": Discriminator(), "D_B": Discriminator(),
    }
    nets = {k: init_weights(v).to(device) for k, v in nets.items()}
    betas = (args.beta1, 0.999)
    opt_G = torch.optim.Adam(itertools.chain(nets["G_AB"].parameters(), nets["G_BA"].parameters()), args.lr, betas=betas)
    opt_D = torch.optim.Adam(itertools.chain(nets["D_A"].parameters(), nets["D_B"].parameters()), args.lr, betas=betas)

    scheds = [torch.optim.lr_scheduler.LambdaLR(o, make_lr_lambda(args)) for o in (opt_G, opt_D)]
    return nets, opt_G, opt_D, scheds


def atomic_save(obj, path):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)  # a Colab disconnect mid-write never corrupts the last good checkpoint


def save_checkpoint(path, epoch, args, nets, opt_G, opt_D, scheds, scaler):
    atomic_save({
        "epoch": epoch, "args": vars(args),
        **{k: v.state_dict() for k, v in nets.items()},
        "opt_G": opt_G.state_dict(), "opt_D": opt_D.state_dict(),
        "sched_G": scheds[0].state_dict(), "sched_D": scheds[1].state_dict(),
        "scaler": scaler.state_dict(),
    }, path)


def load_checkpoint(path, nets, opt_G, opt_D, scheds, scaler, device, args):
    state = torch.load(path, map_location=device, weights_only=False)
    check_compatible(state["args"], args)
    for k, net in nets.items():
        net.load_state_dict(state[k])
    opt_G.load_state_dict(state["opt_G"])
    opt_D.load_state_dict(state["opt_D"])
    scheds[0].load_state_dict(state["sched_G"])
    scheds[1].load_state_dict(state["sched_D"])
    if state["scaler"] and scaler.is_enabled():  # a disabled scaler saves {}
        scaler.load_state_dict(state["scaler"])
    # lr follows the *current* --epochs/--decay_epoch, so extending a run doesn't make lr jump
    for opt, sched in ((opt_G, scheds[0]), (opt_D, scheds[1])):
        for group, base in zip(opt.param_groups, sched.base_lrs):
            group["lr"] = base * sched.lr_lambdas[0](sched.last_epoch)
    return state["epoch"]


def check_compatible(saved, args):
    if saved["n_blocks"] != args.n_blocks:
        raise ValueError(f"checkpoint has n_blocks={saved['n_blocks']} but --n_blocks={args.n_blocks}; "
                         "pass the same value or use a new --checkpoint_dir")
    if saved["lr"] != args.lr:
        log.warning("--lr %g ignored on resume; checkpoint was trained with %g", args.lr, saved["lr"])


def resume_path(args, ckpt_dir):
    if args.resume == "none":
        return None
    if args.resume == "auto":
        latest = ckpt_dir / "latest.pth"
        return latest if latest.exists() else None
    path = Path(args.resume)
    if not path.exists():
        raise FileNotFoundError(f"--resume checkpoint not found: {path}")
    return path


def gan_loss(pred, real):
    return nn.functional.mse_loss(pred, torch.ones_like(pred) if real else torch.zeros_like(pred))


def train_step(real_A, real_B, nets, opt_G, opt_D, pools, scaler, args, amp_ctx):
    G_AB, G_BA, D_A, D_B = nets["G_AB"], nets["G_BA"], nets["D_A"], nets["D_B"]

    for d in (D_A, D_B):
        d.requires_grad_(False)
    opt_G.zero_grad(set_to_none=True)
    with amp_ctx():
        fake_B, fake_A = G_AB(real_A), G_BA(real_B)
        loss_gan = gan_loss(D_B(fake_B), True) + gan_loss(D_A(fake_A), True)
        loss_cycle = (nn.functional.l1_loss(G_BA(fake_B), real_A) + nn.functional.l1_loss(G_AB(fake_A), real_B)) * args.lambda_cycle
        loss_idt = torch.zeros((), device=real_A.device)
        if args.lambda_identity > 0:
            w = args.lambda_cycle * args.lambda_identity
            loss_idt = (nn.functional.l1_loss(G_AB(real_B), real_B) + nn.functional.l1_loss(G_BA(real_A), real_A)) * w
        loss_G = loss_gan + loss_cycle + loss_idt
    scaler.scale(loss_G).backward()
    scaler.step(opt_G)

    for d in (D_A, D_B):
        d.requires_grad_(True)
    opt_D.zero_grad(set_to_none=True)
    with amp_ctx():
        pool_A, pool_B = pools[0].query(fake_A), pools[1].query(fake_B)
        loss_D = 0.5 * (gan_loss(D_A(real_A), True) + gan_loss(D_A(pool_A), False)) \
            + 0.5 * (gan_loss(D_B(real_B), True) + gan_loss(D_B(pool_B), False))
    scaler.scale(loss_D).backward()
    scaler.step(opt_D)
    scaler.update()
    return {"loss_G": loss_G.item(), "loss_D": loss_D.item(), "loss_gan": loss_gan.item(),
            "loss_cycle": loss_cycle.item(), "loss_idt": loss_idt.item()}


@torch.no_grad()
def save_samples(nets, fixed_A, fixed_B, path):
    nets["G_AB"].eval(), nets["G_BA"].eval()
    rows = torch.cat([fixed_A, nets["G_AB"](fixed_A), fixed_B, nets["G_BA"](fixed_B)])
    nets["G_AB"].train(), nets["G_BA"].train()
    save_image(rows * 0.5 + 0.5, path, nrow=len(fixed_A))  # rows: photo, ->sketch, sketch, ->photo


def append_history(path, row):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    ckpt_dir = Path(args.checkpoint_dir)
    (ckpt_dir / "samples").mkdir(parents=True, exist_ok=True)

    photos, sketches = resolve_domains(args.data_dir, args.photo_dir, args.sketch_dir)
    data = UnpairedDataset(photos, sketches, args.image_size, train=True, max_samples=args.max_samples)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                        pin_memory=device == "cuda", drop_last=len(data) >= args.batch_size)
    preview = UnpairedDataset(photos, sketches, args.image_size, train=False)
    n_fixed = min(NUM_SAMPLE_IMAGES, len(preview))
    fixed_A = torch.stack([preview[i][0] for i in range(n_fixed)]).to(device)
    fixed_B = torch.stack([preview[i][1] for i in range(n_fixed)]).to(device)
    log.info("photos=%d sketches=%d iters/epoch=%d device=%s", len(data.photos), len(data.sketches), len(loader), device)

    nets, opt_G, opt_D, scheds = build(args, device)
    use_amp = args.amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def amp_ctx():
        return torch.autocast("cuda", enabled=use_amp)

    start_epoch = 0
    path = resume_path(args, ckpt_dir)
    if path:
        start_epoch = load_checkpoint(path, nets, opt_G, opt_D, scheds, scaler, device, args)
        log.info("resumed from %s at epoch %d", path, start_epoch)
    if start_epoch >= args.epochs:
        log.info("already trained %d/%d epochs; raise --epochs to continue", start_epoch, args.epochs)
        return

    pools = (ImagePool(args.pool_size), ImagePool(args.pool_size))
    for epoch in range(start_epoch + 1, args.epochs + 1):
        t0, totals = time.time(), {k: 0.0 for k in HISTORY_FIELDS[1:6]}
        for it, (real_A, real_B) in enumerate(loader, 1):
            losses = train_step(real_A.to(device), real_B.to(device), nets, opt_G, opt_D, pools, scaler, args, amp_ctx)
            for k, v in losses.items():
                totals[k] += v
            if it % args.log_every == 0:
                log.info("epoch %d it %d/%d %s", epoch, it, len(loader), " ".join(f"{k}={v:.3f}" for k, v in losses.items()))
        lr = opt_G.param_groups[0]["lr"]
        for s in scheds:
            s.step()

        row = {"epoch": epoch, **{k: round(v / max(len(loader), 1), 4) for k, v in totals.items()},
               "lr": lr, "seconds": round(time.time() - t0, 1)}
        append_history(ckpt_dir / "history.csv", row)
        save_samples(nets, fixed_A, fixed_B, ckpt_dir / "samples" / f"epoch_{epoch:03d}.png")
        save_checkpoint(ckpt_dir / "latest.pth", epoch, args, nets, opt_G, opt_D, scheds, scaler)
        if args.snapshot_every and epoch % args.snapshot_every == 0:
            atomic_save({"epoch": epoch, "args": vars(args), "G_AB": nets["G_AB"].state_dict(),
                         "G_BA": nets["G_BA"].state_dict()}, ckpt_dir / f"epoch_{epoch:03d}.pth")
        log.info("epoch %d done %s", epoch, row)


if __name__ == "__main__":
    main()
