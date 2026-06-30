"""
Train ClimateUNet on real IMD anomalies (PyTorch / CUDA).

Loss: latitude-area-weighted, land-masked Huber loss in scaled-anomaly space.
Because every variable is divided by its own anomaly std, the three variables
contribute on a comparable scale by construction.

Saves best checkpoint to models/checkpoints/climate_unet.pt and a training
curve to outputs/training_curve.png.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402
from models.architecture import build_model  # noqa: E402
from models import dataset as D  # noqa: E402

EPOCHS = 60
BATCH = 32
LR = 2.5e-3
WEIGHT_DECAY = 2e-3
PATIENCE = 12
INPUT_NOISE = 0.12   # gaussian augmentation on scaled-anomaly inputs (train only)
NUM_WORKERS = 0      # single-process loading: reliable on a 16 GB machine
TRAIN_STRIDE = 3     # consecutive 7-day windows overlap ~86%; stride keeps full
                     # year coverage while cutting epoch cost ~3x


def make_weight(landmask, lat):
    """(H,W) loss weight = land x cos(latitude), normalised to mean 1 over land."""
    latw = np.cos(np.deg2rad(lat))[:, None]          # (H,1)
    w = np.where(landmask, latw, 0.0).astype("float32")
    w *= landmask.size / max(w.sum(), 1.0)
    return torch.from_numpy(w)


def weighted_huber(pred, target, w, nv=3, horizon=C.HORIZON, beta=1.0):
    """pred/target: (N, horizon*nv, H, W). w: (H,W)."""
    N = pred.shape[0]
    pred = pred.view(N, horizon, nv, *pred.shape[2:])
    target = target.view(N, horizon, nv, *target.shape[2:])
    err = F.smooth_l1_loss(pred, target, reduction="none", beta=beta)  # (N,hz,nv,H,W)
    wm = w.view(1, 1, 1, *w.shape)
    return (err * wm).mean()


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device: {dev}", flush=True)

    obs, clim, stats, landmask, grid = D.load_cache()
    cube, dates, carr, std = D.build_anomaly_cube(obs, clim, stats)
    splits = D.split_indices(dates)
    print(f"[train] windows  train={len(splits['train'])} "
          f"val={len(splits['val'])} test={len(splits['test'])}", flush=True)

    w = make_weight(landmask, grid["lat"]).to(dev)

    # keep the anomaly cube in RAM (fits) for fast random access; subsample the
    # highly-overlapping training windows for ~3x faster epochs.
    train_idx = splits["train"][::TRAIN_STRIDE]
    print(f"[train] using {len(train_idx)} train windows (stride {TRAIN_STRIDE})", flush=True)
    pin = False
    tr = DataLoader(D.WindowDataset(cube, train_idx), batch_size=BATCH,
                    shuffle=True, num_workers=NUM_WORKERS, drop_last=True, pin_memory=pin)
    va = DataLoader(D.WindowDataset(cube, splits["val"]), batch_size=BATCH,
                    shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin)

    model = build_model().to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] params: {n_params:,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=(dev.type == "cuda"))

    best_val, best_epoch, bad = float("inf"), -1, 0
    hist = {"train": [], "val": []}
    ckpt = os.path.join(C.CKPT_DIR, "climate_unet.pt")

    for ep in range(EPOCHS):
        model.train()
        t0, tl, nb = time.time(), 0.0, 0
        for X, Y in tr:
            X, Y = X.to(dev, non_blocking=True), Y.to(dev, non_blocking=True)
            if INPUT_NOISE > 0:
                # augment only the history channels, never the POA prior
                noise = torch.zeros_like(X)
                noise[:, :C.INPUT_DAYS * 3] = INPUT_NOISE * torch.randn_like(X[:, :C.INPUT_DAYS * 3])
                X = X + noise
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, enabled=(dev.type == "cuda")):
                loss = weighted_huber(model(X), Y, w)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            tl += loss.item(); nb += 1
        sched.step()
        tr_loss = tl / max(nb, 1)

        model.eval()
        vl, vnb = 0.0, 0
        with torch.no_grad():
            for X, Y in va:
                X, Y = X.to(dev), Y.to(dev)
                with torch.autocast(device_type=dev.type, enabled=(dev.type == "cuda")):
                    vl += weighted_huber(model(X), Y, w).item(); vnb += 1
        val_loss = vl / max(vnb, 1)
        hist["train"].append(tr_loss); hist["val"].append(val_loss)
        print(f"[train] epoch {ep+1:02d}/{EPOCHS}  train {tr_loss:.4f}  "
              f"val {val_loss:.4f}  lr {sched.get_last_lr()[0]:.2e}  "
              f"{time.time()-t0:.1f}s", flush=True)

        if val_loss < best_val - 1e-5:
            best_val, best_epoch, bad = val_loss, ep, 0
            torch.save({"state_dict": model.state_dict(),
                        "config": {"input_days": C.INPUT_DAYS, "horizon": C.HORIZON},
                        "val_loss": val_loss, "epoch": ep}, ckpt)
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"[train] early stop at epoch {ep+1}", flush=True)
                break

    print(f"[train] best val {best_val:.4f} @ epoch {best_epoch+1}. saved {ckpt}", flush=True)
    with open(os.path.join(C.OUTPUTS_DIR, "train_history.json"), "w") as f:
        json.dump(hist, f)
    _plot(hist)


def _plot(hist):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    plt.plot(hist["train"], label="train")
    plt.plot(hist["val"], label="val")
    plt.xlabel("epoch"); plt.ylabel("weighted Huber loss"); plt.legend()
    plt.title("ClimateUNet training")
    plt.tight_layout()
    plt.savefig(os.path.join(C.OUTPUTS_DIR, "training_curve.png"), dpi=120)


if __name__ == "__main__":
    main()
