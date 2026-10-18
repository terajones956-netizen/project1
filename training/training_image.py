"""
training_image.py - train the FreqAwareCNN image detector on CIFAKE.

CIFAKE: real images (CIFAR-10) vs Stable Diffusion v1.4 images, 32x32 pixels.
Downloaded automatically from Hugging Face (community copy of the original
Kaggle dataset): the zip is fetched and extracted to ./data/cifake (needs ~1 GB disk). If you use it, cite:
  - Krizhevsky & Hinton (2009), "Learning multiple layers of features from tiny images"
  - Bird & Lotfi (2023), "CIFAKE: Image Classification and Explainable
    Identification of AI-Generated Synthetic Images"

Run from the project root (e.g. C:\\Users\\mayan\\Multimodel), venv active:

    python training\\training_image.py                       # 10,000 images per class
    python training\\training_image.py --max_per_class 3000   # faster test run
    python training\\training_image.py --max_per_class 0      # all images (~100k, needs ~4 GB RAM)

Outputs (in --ckpt, default ./checkpoints):
    image.pt                 trained FreqAwareCNN weights
    scores_image.json        val/test AI-probabilities (used by fusion)
    metrics.json             test metrics under "image"

Features are computed ONCE up front with your own ImageFeatureExtractor (same
code as inference), cached in RAM as float16, then reused every epoch.
Label convention: 0 = real, 1 = AI-generated.
"""

import argparse
import copy
import json
import logging
import os
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Make the model file importable whether it is called model.py (in ./model)
# or multimodal_detector.py (anywhere near this script).
HERE = Path(__file__).resolve().parent
for p in (HERE.parent / "model", HERE.parent, HERE):
    sys.path.insert(0, str(p))
try:
    from model import FreqAwareCNN, ImageFeatureExtractor
except ImportError:
    from model import FreqAwareCNN, ImageFeatureExtractor
logging.getLogger("httpx").setLevel(logging.WARNING)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # harmless Windows warning


def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    pred = (p >= 0.5).astype(int)
    out = {"acc": float(accuracy_score(y, pred)), "f1": float(f1_score(y, pred))}
    try:
        out["auc"] = float(roc_auc_score(y, p))
    except ValueError:
        out["auc"] = float("nan")
    return out


def prepare_raw(args):
    """Download the dataset's zip from Hugging Face and extract it once."""
    raw = Path(args.data_dir) / "cifake"
    marker = raw / ".extracted"
    if marker.exists():
        return raw
    from huggingface_hub import hf_hub_download, list_repo_files
    zips = [f for f in list_repo_files(args.dataset, repo_type="dataset")
            if f.lower().endswith(".zip")]
    if not zips:
        raise RuntimeError(f"No .zip files found in dataset repo {args.dataset}")
    raw.mkdir(parents=True, exist_ok=True)
    for f in zips:
        print(f"Downloading {f} ...")
        z = hf_hub_download(args.dataset, f, repo_type="dataset")
        print("Extracting (about 100k small images, can take a few minutes) ...")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(raw)
    marker.write_text("ok")
    return raw


def build_features(args):
    raw = prepare_raw(args)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    items, unlabeled = [], []
    for p in raw.rglob("*"):
        if p.suffix.lower() not in exts:
            continue
        folders = [x.lower() for x in p.relative_to(raw).parts[:-1]]
        is_fake = any("fake" in x for x in folders)
        is_real = any("real" in x for x in folders)
        if is_fake != is_real:                    # exactly one matched
            items.append((str(p), int(is_fake)))  # 1 = AI-generated, 0 = real
        else:
            unlabeled.append(str(p))
    if not items:
        raise RuntimeError("Could not infer labels from folder names. Example paths found:\n  "
                           + "\n  ".join(unlabeled[:5]))
    n_fake = sum(l for _, l in items)
    print(f"Found {len(items)} labelled images ({n_fake} fake / {len(items) - n_fake} real); "
          f"{len(unlabeled)} skipped")

    rng = np.random.default_rng(args.seed)
    chosen = []
    for lab in (0, 1):
        cls = [it for it in items if it[1] == lab]
        if args.max_per_class and len(cls) > args.max_per_class:
            cls = [cls[i] for i in rng.choice(len(cls), args.max_per_class, replace=False)]
        chosen += cls
    n = len(chosen)
    print(f"Extracting features for {n} images ...")

    fx = ImageFeatureExtractor()
    SP = np.zeros((n, 3, 64, 64), np.float16)
    FR = np.zeros((n, 1, 64, 64), np.float16)
    Y = np.zeros(n, np.int64)
    for k, (path, lab) in enumerate(chosen):
        img = cv2.imread(path)                    # BGR, same as inference
        if img is None:
            img = np.zeros((32, 32, 3), np.uint8)
        sp, fr = fx.extract(img)
        SP[k] = sp.transpose(2, 0, 1)
        FR[k, 0] = fr
        Y[k] = lab
        if (k + 1) % 5000 == 0:
            print(f"  {k+1}/{n}")
    return SP, FR, Y


@torch.no_grad()
def predict(model, loader):
    model.eval()
    scores, ys = [], []
    for sp, fr, y in loader:
        scores += model(sp.float(), fr.float())[:, 1].tolist()
        ys += y.tolist()
    return scores, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints")
    ap.add_argument("--dataset", default="yanbax/CIFAKE_autotrain_compatible")
    ap.add_argument("--data_dir", default="data", help="where the dataset zip is extracted")
    ap.add_argument("--max_per_class", type=int, default=10000, help="0 = all")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(os.cpu_count() or 4)
    ckpt = Path(args.ckpt)
    ckpt.mkdir(parents=True, exist_ok=True)

    SP, FR, Y = build_features(args)

    # Stratified 70 / 15 / 15 split
    idx = np.arange(len(Y))
    tr_i, tmp_i = train_test_split(idx, test_size=0.30, stratify=Y, random_state=args.seed)
    va_i, te_i = train_test_split(tmp_i, test_size=0.50, stratify=Y[tmp_i], random_state=args.seed)
    print(f"train={len(tr_i)} val={len(va_i)} test={len(te_i)}")

    def loader(ids, shuffle):
        ds = TensorDataset(torch.from_numpy(SP[ids]), torch.from_numpy(FR[ids]),
                           torch.from_numpy(Y[ids]))
        return DataLoader(ds, batch_size=args.batch, shuffle=shuffle)

    tr, va, te = loader(tr_i, True), loader(va_i, False), loader(te_i, False)

    model = FreqAwareCNN()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best, best_state, bad = -1.0, None, 0

    for ep in range(args.epochs):
        model.train()
        total = 0.0
        for sp, fr, y in tr:
            probs = model(sp.float(), fr.float())
            loss = F.nll_loss(torch.log(probs + 1e-9), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        v_s, v_y = predict(model, va)
        m = metrics(v_y, v_s)
        print(f"epoch {ep+1}/{args.epochs}  loss={total/len(tr_i):.4f}  "
              f"val_acc={m['acc']:.4f}  val_auc={m['auc']:.4f}")
        crit = m["auc"] if not np.isnan(m["auc"]) else m["acc"]
        if crit > best:
            best, best_state, bad = crit, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= args.patience:
                print("early stopping")
                break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), ckpt / "image.pt")

    v_s, v_y = predict(model, va)
    t_s, t_y = predict(model, te)
    tm = metrics(t_y, t_s)
    print(f"\nTEST RESULTS: {tm}")

    with open(ckpt / "scores_image.json", "w") as f:
        json.dump({"val": {"scores": v_s, "labels": v_y},
                   "test": {"scores": t_s, "labels": t_y}}, f)
    mpath = ckpt / "metrics.json"
    allm = json.load(open(mpath)) if mpath.exists() else {}
    allm["image"] = tm
    json.dump(allm, open(mpath, "w"), indent=2)
    print(f"Saved {ckpt / 'image.pt'}, scores_image.json, metrics.json")


if __name__ == "__main__":
    main()