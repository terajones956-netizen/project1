"""
training_video_effnet.py - fine-tune a PRETRAINED EfficientNet-B0 as the video deepfake detector.

Why this instead of VideoFrameCNN: a from-scratch CNN on 64x64 crops stayed at chance.
This script uses ImageNet weights, 160px face crops (--size), and trains on individual
frames (16 per video), then scores a video as the mean of its frame probabilities.

It reuses the video selection / zip / metrics code from training_video.py (same
person-ID split, same matched real/fake pairs), so put both files in the same folder.
Videos already copied to <data_dir>/video_hf are reused, not downloaded again.

Colab (GPU runtime), after uploading training_video.py, your model file and this file:
    !python training_video_effnet.py --n_per_class 16 --epochs 15 --no_augment --data_dir /content/data --ckpt /content/checkpoints   # overfit check
    !python training_video_effnet.py --n_per_class 400 --data_dir /content/data --ckpt /content/checkpoints

Outputs (in --ckpt): video_effnet.pt, scores_video_effnet.json, metrics.json["video_effnet"]
NOTE: this is a different architecture from VideoFrameCNN, so your inference code must
build the same EfficientNet and apply the same face crop (see read_frames/crop_faces).
"""

import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from training_video import DEFAULT_MANIPS, extract_members, metrics, open_zip, select_videos

T_FRAMES = 16
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


# ---------- data: frames + face crops ----------
def read_frames(path, n=T_FRAMES):
    """n evenly spaced BGR frames from the whole video."""
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in np.linspace(0, max(total - 1, 0), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    return frames


def crop_faces(frames, size, margin=1.4):
    """Detect the face on each frame, use the median box for the whole clip, crop and
    resize to size x size RGB. Falls back to a centered crop if no face is found.
    Returns (array (T,size,size,3) uint8, face_found: bool)."""
    boxes = []
    for f in frames:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        d = CASCADE.detectMultiScale(g, 1.1, 5, minSize=(60, 60))
        if len(d):
            boxes.append(max(d, key=lambda b: b[2] * b[3]))
    h, w = frames[0].shape[:2]
    found = len(boxes) > 0
    if found:
        x, y, bw, bh = np.median(np.array(boxes, dtype=np.float32), axis=0)
        cx, cy, s = x + bw / 2, y + bh / 2, max(bw, bh) * margin
    else:
        cx, cy, s = w / 2, h / 2, min(h, w) * 0.6
    x0, y0 = int(max(cx - s / 2, 0)), int(max(cy - s / 2, 0))
    x1, y1 = int(min(cx + s / 2, w)), int(min(cy + s / 2, h))
    out = []
    for f in frames:
        c = cv2.resize(f[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_AREA)
        out.append(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
    while len(out) < T_FRAMES:
        out.append(out[-1])
    return np.stack(out[:T_FRAMES]), found


# ---------- model ----------
def make_model():
    m = torchvision.models.efficientnet_b0(
        weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(1280, 2))
    return m


def prep(xb, train, augment):
    """xb: uint8 tensor (B,S,S,3) on device -> normalized float (B,3,S,S)."""
    x = xb.permute(0, 3, 1, 2).float() / 255.0
    if train and augment:
        flip = torch.rand(x.shape[0], device=x.device) < 0.5
        x = torch.where(flip.view(-1, 1, 1, 1), x.flip(-1), x)
        b = (torch.rand(x.shape[0], 1, 1, 1, device=x.device) - 0.5) * 0.2      # brightness
        c = 1 + (torch.rand(x.shape[0], 1, 1, 1, device=x.device) - 0.5) * 0.2  # contrast
        x = ((x - 0.5) * c + 0.5 + b).clamp(0, 1)
    return (x - MEAN.to(x.device)) / STD.to(x.device)


@torch.no_grad()
def predict(model, X, dev, vb=4):
    """Video-level P(fake) = mean of per-frame softmax. X: (N,T,S,S,3) uint8 numpy."""
    model.eval()
    scores = []
    for i in range(0, len(X), vb):
        xb = torch.from_numpy(X[i:i + vb]).to(dev)
        B, T = xb.shape[:2]
        with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
            logits = model(prep(xb.flatten(0, 1), False, False))
        p = F.softmax(logits.float(), dim=-1).view(B, T, 2).mean(1)[:, 1]
        scores += p.tolist()
    return scores


def auc_ci(y, p, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) == 2:
            vals.append(roc_auc_score(y[i], p[i]))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"),) * 2


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints")
    ap.add_argument("--dataset", default="bitmind/FaceForensicsC23")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--zip", default=None)
    ap.add_argument("--manipulations", default=DEFAULT_MANIPS)
    ap.add_argument("--n_per_class", type=int, default=300)
    ap.add_argument("--size", type=int, default=160, help="face crop resolution")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64, help="frames per batch")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_augment", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev)
    ckpt = Path(args.ckpt)
    ckpt.mkdir(parents=True, exist_ok=True)
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    cache = Path(args.data_dir) / f"video_faces_{args.n_per_class}_{args.seed}_{args.size}.npz"

    if cache.exists():
        print(f"Loading cached crops from {cache}")
        z = np.load(cache)
        X, Y, S = z["X"], z["Y"], z["S"]
    else:
        zf = open_zip(args)
        files = [i.filename for i in zf.infolist() if not i.is_dir()]
        sel = select_videos(files, set(args.manipulations.split(",")), args.n_per_class, args.seed)
        items = [(f, lab, s) for s in range(3) for f, lab in sel[s]]
        print(f"Selected {len(items)} videos: " + ", ".join(
            f"{n}={len(sel[s])}" for s, n in enumerate(["train", "val", "test"])))
        paths = extract_members(zf, [f for f, _, _ in items], args)
        zf.close()

        X = np.zeros((len(items), T_FRAMES, args.size, args.size, 3), np.uint8)
        ok = np.zeros(len(items), bool)
        found_n = 0
        print("Cropping faces ...")
        for k, p in enumerate(paths):
            try:
                frames = read_frames(p)
                if not frames:
                    raise RuntimeError("no frames decoded")
                X[k], found = crop_faces(frames, args.size)
                ok[k] = True
                found_n += found
            except Exception as e:
                print(f"  skipped {p}: {e}")
            if (k + 1) % 50 == 0:
                print(f"  {k+1}/{len(items)}")
        print(f"Face detected in {found_n}/{ok.sum()} videos (rest use a center-crop fallback)")
        Y = np.array([lab for _, lab, _ in items], np.int64)[ok]
        S = np.array([s for _, _, s in items])[ok]
        X = X[ok]
        np.savez(cache, X=X, Y=Y, S=S)

    tr_i, va_i, te_i = (np.where(S == s)[0] for s in range(3))
    print(f"train={len(tr_i)} val={len(va_i)} test={len(te_i)} videos")
    Xtr = X[tr_i].reshape(-1, args.size, args.size, 3)          # every frame is a sample
    Ytr = np.repeat(Y[tr_i], T_FRAMES)

    model = make_model().to(dev)
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler(dev.type, enabled=use_amp)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    steps_per_ep = math.ceil(len(Xtr) / args.batch)
    total, warm = args.epochs * steps_per_ep, steps_per_ep
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))

    best, best_state, bad = -1.0, None, 0
    for ep in range(args.epochs):
        model.train()
        perm = np.random.permutation(len(Xtr))
        tot_loss, correct, n = 0.0, 0, 0
        for i in range(0, len(perm), args.batch):
            idx = np.sort(perm[i:i + args.batch])
            xb = torch.from_numpy(Xtr[idx]).to(dev)
            yb = torch.from_numpy(Ytr[idx]).to(dev)
            with torch.autocast(device_type=dev.type, enabled=use_amp):
                logits = model(prep(xb, True, not args.no_augment))
                loss = F.cross_entropy(logits.float(), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot_loss += loss.item() * len(idx)
            correct += (logits.argmax(1) == yb).sum().item()
            n += len(idx)
        vs = predict(model, X[va_i], dev)
        m = metrics(Y[va_i], vs)
        print(f"epoch {ep+1}/{args.epochs}  loss={tot_loss/n:.4f}  train_frame_acc={correct/n:.3f}  "
              f"val_acc={m['acc']:.3f}  val_auc={m['auc']:.3f}")
        crit = m["auc"] if not np.isnan(m["auc"]) else m["acc"]
        if crit > best:
            best, best_state, bad = crit, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= args.patience:
                print("early stopping")
                break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), ckpt / "video_effnet.pt")
    v_s, v_y = predict(model, X[va_i], dev), Y[va_i].tolist()
    t_s, t_y = predict(model, X[te_i], dev), Y[te_i].tolist()
    tm = metrics(t_y, t_s)
    lo, hi = auc_ci(t_y, t_s)
    print(f"\nTEST RESULTS ({len(te_i)} videos): {tm}")
    print(f"Test AUC 95% bootstrap CI: [{lo:.3f}, {hi:.3f}]  (wide CI = result is not reliable)")
    if best < 0.65:
        print("WARNING: validation AUC stayed low - treat this model as not learned; don't use it in fusion.")

    with open(ckpt / "scores_video_effnet.json", "w") as f:
        json.dump({"val": {"scores": v_s, "labels": v_y}, "test": {"scores": t_s, "labels": t_y}}, f)
    mpath = ckpt / "metrics.json"
    allm = json.load(open(mpath)) if mpath.exists() else {}
    allm["video_effnet"] = tm
    json.dump(allm, open(mpath, "w"), indent=2)
    print(f"Saved {ckpt / 'video_effnet.pt'}, scores_video_effnet.json, metrics.json")


if __name__ == "__main__":
    main()