"""
training_image_combined.py - train the FreqAwareCNN image detector on CIFAKE + 140k Real and Fake Faces.

CIFAKE : real CIFAR-10 images vs Stable Diffusion images, 32x32 (downloaded from Hugging Face).
Faces  : 140k Real and Fake Faces (FFHQ real vs StyleGAN fake), 256x256 JPG, from Kaggle:
             kaggle datasets download -d xhlulu/140k-real-and-fake-faces -p data/faces --unzip
         (needs a Kaggle API token; check the dataset licence and cite both datasets.)
Cite: Krizhevsky & Hinton (2009); Bird & Lotfi (2023) for CIFAKE.

Label: 0 = real, 1 = AI-generated. Labels come from the folder named exactly real / fake
(closest ancestor), so the faces path "real_vs_fake/real-vs-fake/train/real" is read correctly.

The two sources look very different (32x32 upscaled vs 256x256 downscaled), so the script prints
metrics PER SOURCE as well as overall. An overall score alone can hide a weak source.

Run from the project root (venv active; model/model.py must be importable):
    python training/training_image_combined.py --faces_dir data/faces
    python training/training_image_combined.py --faces_dir data/faces --cifake_per_class 3000 --faces_per_class 3000   # quick test
Uses the GPU automatically if CUDA is available (Colab).

Outputs (--ckpt): image.pt, scores_image.json, metrics.json["image"]. Existing image.pt and
scores_image.json are kept as image_prev.pt / scores_image_prev.json. Re-run training_fusion.py afterwards.
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
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
for p in (HERE.parent / "model", HERE.parent, HERE):
    sys.path.insert(0, str(p))
from model import FreqAwareCNN, ImageFeatureExtractor  # noqa: E402

logging.getLogger("httpx").setLevel(logging.WARNING)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    pred = (p >= 0.5).astype(int)
    out = {"acc": float(accuracy_score(y, pred)),
           "precision": float(precision_score(y, pred, zero_division=0)),
           "recall": float(recall_score(y, pred, zero_division=0)),
           "f1": float(f1_score(y, pred, zero_division=0))}
    try:
        out["auc"] = float(roc_auc_score(y, p))
    except ValueError:
        out["auc"] = float("nan")
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out.update({"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)})
    return out


def label_from_path(p, root):
    """1 = fake, 0 = real, None = cannot tell. Closest ancestor folder named exactly real/fake wins."""
    parts = [x.lower() for x in p.relative_to(root).parts[:-1]]
    for x in reversed(parts):
        if x in ("real", "fake"):
            return int(x == "fake")
    for x in reversed(parts):                      # fallback: names like "FAKE_images"
        f, r = "fake" in x, "real" in x
        if f != r:
            return int(f)
    return None


def prepare_cifake(args):
    raw = Path(args.data_dir) / "cifake"
    marker = raw / ".extracted"
    if marker.exists():
        return raw
    from huggingface_hub import hf_hub_download, list_repo_files
    zips = [f for f in list_repo_files(args.dataset, repo_type="dataset") if f.lower().endswith(".zip")]
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


def list_images(root, source):
    items, skipped = [], 0
    for p in Path(root).rglob("*"):
        if p.suffix.lower() not in EXTS:
            continue
        lab = label_from_path(p, Path(root))
        if lab is None:
            skipped += 1
        else:
            items.append((str(p), lab, source))
    n_fake = sum(l for _, l, _ in items)
    print(f"{source}: {len(items)} labelled images ({n_fake} fake / {len(items) - n_fake} real); "
          f"{skipped} skipped")
    if not items:
        raise RuntimeError(f"No labelled images found for {source} under {root}")
    return items


def sample(items, per_class, rng):
    out = []
    for lab in (0, 1):
        cls = [it for it in items if it[1] == lab]
        if per_class and len(cls) > per_class:
            cls = [cls[i] for i in rng.choice(len(cls), per_class, replace=False)]
        out += cls
    return out


def build_features(args):
    rng = np.random.default_rng(args.seed)
    cifake = sample(list_images(prepare_cifake(args), "cifake"), args.cifake_per_class, rng)
    if not Path(args.faces_dir).exists():
        raise SystemExit(f"--faces_dir {args.faces_dir} not found. Download the Kaggle dataset first "
                         f"(see the docstring), or use training_image.py for CIFAKE only.")
    faces = sample(list_images(args.faces_dir, "faces"), args.faces_per_class, rng)
    chosen = cifake + faces
    n = len(chosen)
    print(f"Extracting features for {n} images ...")

    fx = ImageFeatureExtractor()
    SP = np.zeros((n, 3, 64, 64), np.float16)
    FR = np.zeros((n, 1, 64, 64), np.float16)
    Y = np.zeros(n, np.int64)
    SRC = np.empty(n, dtype=object)
    for k, (path, lab, src) in enumerate(chosen):
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((32, 32, 3), np.uint8)
        sp, fr = fx.extract(img)
        SP[k] = sp.transpose(2, 0, 1)
        FR[k, 0] = fr
        Y[k], SRC[k] = lab, src
        if (k + 1) % 5000 == 0:
            print(f"  {k + 1}/{n}")
    return SP, FR, Y, SRC


@torch.no_grad()
def predict(model, loader, dev):
    model.eval()
    scores, ys = [], []
    for sp, fr, y in loader:
        scores += model(sp.float().to(dev), fr.float().to(dev))[:, 1].cpu().tolist()
        ys += y.tolist()
    return scores, ys


def keep_previous(path):
    path = Path(path)
    if path.exists():
        prev = path.with_name(path.stem + "_prev" + path.suffix)
        if prev.exists():
            prev.unlink()
        path.rename(prev)
        print(f"Kept previous file as {prev}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints")
    ap.add_argument("--dataset", default="yanbax/CIFAKE_autotrain_compatible")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--faces_dir", default="data/faces")
    ap.add_argument("--cifake_per_class", type=int, default=10000, help="0 = all")
    ap.add_argument("--faces_per_class", type=int, default=10000, help="0 = all")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(os.cpu_count() or 4)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev)
    ckpt = Path(args.ckpt)
    ckpt.mkdir(parents=True, exist_ok=True)

    SP, FR, Y, SRC = build_features(args)

    idx = np.arange(len(Y))
    strat = np.array([f"{s}_{l}" for s, l in zip(SRC, Y)])
    tr_i, tmp_i = train_test_split(idx, test_size=0.30, stratify=strat, random_state=args.seed)
    va_i, te_i = train_test_split(tmp_i, test_size=0.50, stratify=strat[tmp_i], random_state=args.seed)
    print(f"train={len(tr_i)} val={len(va_i)} test={len(te_i)}")

    def loader(ids, shuffle):
        ds = TensorDataset(torch.from_numpy(SP[ids]), torch.from_numpy(FR[ids]), torch.from_numpy(Y[ids]))
        return DataLoader(ds, batch_size=args.batch, shuffle=shuffle)

    tr, va, te = loader(tr_i, True), loader(va_i, False), loader(te_i, False)

    model = FreqAwareCNN().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best, best_state, bad = -1.0, None, 0

    for ep in range(args.epochs):
        model.train()
        total = 0.0
        for sp, fr, y in tr:
            sp, fr, y = sp.float().to(dev), fr.float().to(dev), y.to(dev)
            probs = model(sp, fr)
            loss = F.nll_loss(torch.log(probs + 1e-9), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        v_s, v_y = predict(model, va, dev)
        m = metrics(v_y, v_s)
        print(f"epoch {ep + 1}/{args.epochs}  loss={total / len(tr_i):.4f}  val_acc={m['acc']:.4f}  "
              f"val_prec={m['precision']:.4f}  val_rec={m['recall']:.4f}  val_auc={m['auc']:.4f}")
        crit = m["auc"] if not np.isnan(m["auc"]) else m["acc"]
        if crit > best:
            best, best_state, bad = crit, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= args.patience:
                print("early stopping")
                break

    model.load_state_dict(best_state)
    v_s, v_y = predict(model, va, dev)
    t_s, t_y = predict(model, te, dev)
    tm = metrics(t_y, t_s)
    t_y_arr, t_s_arr, src_te = np.array(t_y), np.array(t_s), SRC[te_i]
    tm["per_source"] = {s: metrics(t_y_arr[src_te == s], t_s_arr[src_te == s]) for s in np.unique(src_te)}
    print("\nTEST (all):", {k: round(v, 4) if isinstance(v, float) else v
                            for k, v in tm.items() if k != "per_source"})
    for s, mm in tm["per_source"].items():
        print(f"TEST ({s}): acc {mm['acc']:.4f}  prec {mm['precision']:.4f}  rec {mm['recall']:.4f}  "
              f"F1 {mm['f1']:.4f}  AUC {mm['auc']:.4f}")

    keep_previous(ckpt / "image.pt")
    keep_previous(ckpt / "scores_image.json")
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, ckpt / "image.pt")
    with open(ckpt / "scores_image.json", "w") as f:
        json.dump({"val": {"scores": v_s, "labels": v_y, "sources": SRC[va_i].tolist()},
                   "test": {"scores": t_s, "labels": t_y, "sources": src_te.tolist()}}, f)
    mpath = ckpt / "metrics.json"
    allm = json.load(open(mpath)) if mpath.exists() else {}
    allm["image"] = tm
    json.dump(allm, open(mpath, "w"), indent=2)
    print(f"\nSaved {ckpt / 'image.pt'}, scores_image.json, metrics.json['image']")


if __name__ == "__main__":
    main()
