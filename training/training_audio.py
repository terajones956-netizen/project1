"""
training_audio.py - train the AudioSpectrogramCNN voice-deepfake detector.

Data source (pick one):
  1. DEFAULT: Hugging Face dataset "garystafford/deepfake-audio-detection"
     (1,866 FLAC clips: 933 real, 933 synthetic, in fake/ and real/ folders).
     Small and CPU-friendly. Downloaded to ./data/audio_hf.
  2. Your own folder (e.g. ASVspoof / WaveFake after you arrange it):
         python training\\training_audio.py --folder D:\\data\\audio
     Folder names decide the label:
         real / human / bonafide / genuine / authentic   -> 0 (real)
         fake / spoof / ai / synthetic / generated        -> 1 (AI)
     (any parent folder of a file may carry the name, e.g. audio\\ai\\wavefake\\x.wav)

Run from the project root (e.g. C:\\Users\\mayan\\Multimodel), venv active:

    python training\\training_audio.py
    python training\\training_audio.py --max_per_class 500     # quick test

Outputs (in --ckpt, default ./checkpoints):
    audio.pt, scores_audio.json, metrics.json["audio"]

Features use YOUR AudioFeatureExtractor on the first --max_sec seconds of each
clip (default 5 s). Inference must truncate the same way (see reply notes).
"""

import argparse
import copy
import json
import logging
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
for p in (HERE.parent / "model", HERE.parent, HERE):
    sys.path.insert(0, str(p))
try:
    from model import AudioFeatureExtractor, AudioSpectrogramCNN
except ImportError:
    from model import AudioFeatureExtractor, AudioSpectrogramCNN
logging.getLogger("httpx").setLevel(logging.WARNING)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
FAKE = {"fake", "spoof", "spoofed", "ai", "synthetic", "generated", "deepfake", "tts"}
REAL = {"real", "human", "bonafide", "bona_fide", "genuine", "authentic"}
SR = 16000


def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    pred = (p >= 0.5).astype(int)
    out = {"acc": float(accuracy_score(y, pred)), "f1": float(f1_score(y, pred))}
    try:
        out["auc"] = float(roc_auc_score(y, p))
    except ValueError:
        out["auc"] = float("nan")
    return out


def prepare_hf(args):
    from huggingface_hub import list_repo_files, snapshot_download
    files = list_repo_files(args.dataset, repo_type="dataset")
    audio = [f for f in files if Path(f).suffix.lower() in AUDIO_EXTS]
    if not audio:
        raise RuntimeError("No audio files found in the dataset repo. Files seen:\n  "
                           + "\n  ".join(files[:15]))
    print(f"Dataset repo has {len(audio)} audio files; downloading (skips files already there) ...")
    local = Path(args.data_dir) / "audio_hf"
    snapshot_download(args.dataset, repo_type="dataset", local_dir=str(local),
                      allow_patterns=[f"*{e}" for e in AUDIO_EXTS])
    return local


def collect(root):
    items, unlabeled = [], []
    for p in Path(root).rglob("*"):
        if p.suffix.lower() not in AUDIO_EXTS:
            continue
        parts = {x.lower() for x in p.relative_to(root).parts[:-1]}
        is_fake, is_real = bool(parts & FAKE), bool(parts & REAL)
        if is_fake != is_real:
            items.append((str(p), int(is_fake)))   # 1 = AI, 0 = real
        else:
            unlabeled.append(str(p))
    if not items:
        raise RuntimeError("Could not infer labels from folder names. Example paths:\n  "
                           + "\n  ".join(unlabeled[:5]))
    n_fake = sum(l for _, l in items)
    print(f"Found {len(items)} labelled clips ({n_fake} fake / {len(items) - n_fake} real); "
          f"{len(unlabeled)} skipped")
    return items


def build_features(items, args):
    rng = np.random.default_rng(args.seed)
    chosen = []
    for lab in (0, 1):
        cls = [it for it in items if it[1] == lab]
        if args.max_per_class and len(cls) > args.max_per_class:
            cls = [cls[i] for i in rng.choice(len(cls), args.max_per_class, replace=False)]
        chosen += cls

    fx = AudioFeatureExtractor(sr=SR)
    specs, ys, bad = [], [], 0
    print(f"Extracting features for {len(chosen)} clips ...")
    for k, (path, lab) in enumerate(chosen):
        try:
            y, _ = librosa.load(path, sr=SR, mono=True, duration=args.max_sec)
            if len(y) < 1600:
                y = np.pad(y, (0, 1600 - len(y)))
            specs.append(fx.extract(y, SR))
            ys.append(lab)
        except Exception as e:                      # unreadable file -> skip, don't poison data
            bad += 1
            if bad <= 3:
                print(f"  skipped {path}: {e}")
        if (k + 1) % 500 == 0:
            print(f"  {k+1}/{len(chosen)}")
    if bad:
        print(f"Skipped {bad} unreadable clips")
    X = np.stack(specs)[:, None, :, :].astype(np.float16)   # (N,1,64,64)
    return X, np.array(ys, dtype=np.int64)


@torch.no_grad()
def predict(model, loader):
    model.eval()
    scores, ys = [], []
    for x, y in loader:
        scores += model(x.float())[:, 1].tolist()
        ys += y.tolist()
    return scores, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints")
    ap.add_argument("--dataset", default="garystafford/deepfake-audio-detection")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--folder", default=None, help="use a local folder instead of downloading")
    ap.add_argument("--max_per_class", type=int, default=0, help="0 = all")
    ap.add_argument("--max_sec", type=float, default=5.0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(os.cpu_count() or 4)
    ckpt = Path(args.ckpt)
    ckpt.mkdir(parents=True, exist_ok=True)

    root = Path(args.folder) if args.folder else prepare_hf(args)
    X, Y = build_features(collect(root), args)

    idx = np.arange(len(Y))
    tr_i, tmp_i = train_test_split(idx, test_size=0.30, stratify=Y, random_state=args.seed)
    va_i, te_i = train_test_split(tmp_i, test_size=0.50, stratify=Y[tmp_i], random_state=args.seed)
    print(f"train={len(tr_i)} val={len(va_i)} test={len(te_i)}")

    def loader(ids, shuffle):
        return DataLoader(TensorDataset(torch.from_numpy(X[ids]), torch.from_numpy(Y[ids])),
                          batch_size=args.batch, shuffle=shuffle)

    tr, va, te = loader(tr_i, True), loader(va_i, False), loader(te_i, False)

    model = AudioSpectrogramCNN()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best, best_state, bad = -1.0, None, 0
    for ep in range(args.epochs):
        model.train()
        total = 0.0
        for x, y in tr:
            probs = model(x.float())
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
    torch.save(model.state_dict(), ckpt / "audio.pt")
    v_s, v_y = predict(model, va)
    t_s, t_y = predict(model, te)
    tm = metrics(t_y, t_s)
    print(f"\nTEST RESULTS: {tm}")

    with open(ckpt / "scores_audio.json", "w") as f:
        json.dump({"val": {"scores": v_s, "labels": v_y},
                   "test": {"scores": t_s, "labels": t_y}}, f)
    mpath = ckpt / "metrics.json"
    allm = json.load(open(mpath)) if mpath.exists() else {}
    allm["audio"] = tm
    json.dump(allm, open(mpath, "w"), indent=2)
    print(f"Saved {ckpt / 'audio.pt'}, scores_audio.json, metrics.json")


if __name__ == "__main__":
    main()
