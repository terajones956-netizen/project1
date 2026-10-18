"""
training_video.py - train the VideoFrameCNN deepfake-video detector.

Data: FaceForensics++ (C23) via the Hugging Face mirror "bitmind/FaceForensicsC23".
The mirror is ONE 17.9 GB zip (FaceForensics++_C23.zip) holding real/ and
fake/<Manipulation>/ folders. This script opens that zip remotely and copies out
only a SUBSET (--n_per_class videos per class, default 300, roughly 1 GB), so you
never download the whole thing. If remote access fails, download the zip yourself
and pass --zip path\\to\\FaceForensics++_C23.zip. FaceForensics++ has its own
research terms of use - read and follow them before using or publishing results.

Leakage control: every fake is made from two real source videos (e.g. 000_003.mp4
uses people 000 and 003). We split by PERSON ID first (70/15/15), keep a fake only
if BOTH of its ids are in the same split, and take reals by id. So no source video
appears in more than one split.

Run from the project root (e.g. C:\\Users\\mayan\\Multimodel), venv active:

    python training\\training_video.py --n_per_class 60     # quick test
    python training\\training_video.py                       # 300 per class

Outputs (in --ckpt, default ./checkpoints):
    video.pt, scores_video.json, metrics.json["video"]

IMPORTANT: this needs the face-cropping VideoFeatureExtractor patch in your model
file (see the reply). Training and inference must crop faces the same way.
"""

import argparse
import copy
import json
import logging
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
for p in (HERE.parent / "model", HERE.parent, HERE):
    sys.path.insert(0, str(p))
try:
    from model import ImageFeatureExtractor, VideoFeatureExtractor, VideoFrameCNN
except ImportError:
    from model import ImageFeatureExtractor, VideoFeatureExtractor, VideoFrameCNN
logging.getLogger("httpx").setLevel(logging.WARNING)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

T_FRAMES = 16
ID_RE = re.compile(r"^(\d+)(?:_(\d+))?$")
DEFAULT_MANIPS = "deepfakes,face2face,faceshifter,faceswap,neuraltextures"


def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    pred = (p >= 0.5).astype(int)
    out = {"acc": float(accuracy_score(y, pred)), "f1": float(f1_score(y, pred))}
    try:
        out["auc"] = float(roc_auc_score(y, p))
    except ValueError:
        out["auc"] = float("nan")
    return out


def select_videos(files, manips, n_per_class, seed):
    """Split by person id, then pick up to n_per_class MATCHED real/fake pairs per split
    (each fake comes with the real video it was made from, so scene/identity cannot
    give the label away).
    Returns {0: [(file, label)...], 1: [...], 2: [...]} for train / val / test."""
    rng = np.random.default_rng(seed)
    reals, fakes = {}, []
    for f in files:
        pp = Path(f)
        if pp.suffix.lower() != ".mp4":
            continue
        folders = [x.lower() for x in pp.parts[:-1]]
        m = ID_RE.match(pp.stem)
        if not m:
            continue
        a = int(m.group(1))
        b = int(m.group(2)) if m.group(2) else None
        if "real" in folders and b is None:
            reals[a] = f
        elif b is not None and any(x in manips for x in folders):
            fakes.append((a, b, f))
    ids = sorted(reals)
    if len(ids) < 10 or not fakes:
        sample = [f for f in files if f.lower().endswith(".mp4")][:8]
        raise RuntimeError(f"Unexpected layout: {len(ids)} real ids, {len(fakes)} fakes parsed. "
                           f"Example mp4 paths:\n  " + "\n  ".join(sample))

    perm = [int(i) for i in rng.permutation(ids)]
    n_tr, n_va = int(0.70 * len(perm)), int(0.15 * len(perm))
    split_of = {i: (0 if k < n_tr else 1 if k < n_tr + n_va else 2) for k, i in enumerate(perm)}

    q_tr, q_va = int(0.70 * n_per_class), int(0.15 * n_per_class)
    quota = [q_tr, q_va, n_per_class - q_tr - q_va]

    out = {0: [], 1: [], 2: []}
    for s in range(3):
        # fakes whose two source ids both fall in this split, grouped by the target id
        by_target = {}
        for a, b, f in fakes:
            if a in split_of and b in split_of and split_of[a] == s and split_of[b] == s:
                by_target.setdefault(a, []).append(f)
        eligible = [i for i in ids if split_of[i] == s and i in by_target]
        rng.shuffle(eligible)
        # matched pairs: each chosen real video + one fake made from it (random manipulation)
        for i in eligible[:quota[s]]:
            out[s].append((reals[i], 0))
            out[s].append((by_target[i][int(rng.integers(len(by_target[i])))], 1))
    return out


def open_zip(args):
    """Open the dataset zip: a local file (--zip) or remotely via HTTP range reads."""
    import zipfile
    if args.zip:
        return zipfile.ZipFile(args.zip)
    from huggingface_hub import HfFileSystem, list_repo_files
    names = [f for f in list_repo_files(args.dataset, repo_type="dataset")
             if f.lower().endswith(".zip")]
    if not names:
        raise RuntimeError("No .zip file found in the dataset repo; try --zip with a local copy.")
    print(f"Opening {names[0]} remotely (reads only the parts we need) ...")
    fobj = HfFileSystem().open(f"datasets/{args.dataset}/{names[0]}", "rb")
    return zipfile.ZipFile(fobj)


def extract_members(zf, members, args):
    """Copy the chosen videos out of the zip to ./data/video_hf (skips ones already there)."""
    local = Path(args.data_dir) / "video_hf"
    local.mkdir(parents=True, exist_ok=True)
    paths = []
    for k, name in enumerate(members):
        target = local / name
        info = zf.getinfo(name)
        if not (target.exists() and target.stat().st_size == info.file_size):
            for attempt in range(3):
                try:
                    zf.extract(name, path=local)
                    break
                except Exception as e:
                    if attempt == 2:
                        raise
                    print(f"  retrying {name}: {e}")
        paths.append(str(target))
        if (k + 1) % 25 == 0 or k + 1 == len(members):
            print(f"  copied {k+1}/{len(members)}")
    return paths


def clip_features(path, vfx, ifx):
    """Same preprocessing as MultimodalDetector._detect_video."""
    frames = vfx.extract(path)                      # list of (64,64,3) float32
    while len(frames) < T_FRAMES:
        frames.append(frames[-1])
    frames = frames[:T_FRAMES]
    sp = np.stack(frames).transpose(0, 3, 1, 2)     # (T,3,64,64)
    fr = []
    for f in frames:
        gray = cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        fr.append(ifx.extract(gray)[1])
    return sp.astype(np.float16), np.stack(fr)[:, None].astype(np.float16)


def frame_logits(model, sp, fr):
    """Per-frame logits (B*T, 2): same layers as FreqAwareCNN.forward, minus the softmax."""
    fm = model.frame_model
    s = fm.spatial(sp.flatten(0, 1).float()).flatten(1)
    f = fm.freq(fr.flatten(0, 1).float()).flatten(1)
    return fm.head(torch.cat([s, f], dim=1))


def fwd(model, sp, fr):
    """Video-level probabilities (B, 2): mean of per-frame softmax, as at inference."""
    B, T = sp.shape[:2]
    p = F.softmax(frame_logits(model, sp, fr), dim=-1)
    return p.view(B, T, 2).mean(1)


@torch.no_grad()
def predict(model, loader):
    model.eval()
    scores, ys = [], []
    for sp, fr, y in loader:
        scores += fwd(model, sp, fr)[:, 1].tolist()
        ys += y.tolist()
    return scores, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints")
    ap.add_argument("--dataset", default="bitmind/FaceForensicsC23")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--zip", default=None, help="path to a local copy of the dataset zip")
    ap.add_argument("--manipulations", default=DEFAULT_MANIPS,
                    help="comma-separated fake folders to use (lowercase)")
    ap.add_argument("--n_per_class", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=16, help="videos per batch (x16 frames)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_augment", action="store_true", help="disable random horizontal flips")
    ap.add_argument("--no_face_crop_check", action="store_true",
                    help="skip the check that your model file has the face-crop patch")
    args = ap.parse_args()

    if not getattr(VideoFeatureExtractor, "FACE_CROP", False) and not args.no_face_crop_check:
        sys.exit("Your VideoFeatureExtractor has no face cropping (FACE_CROP flag missing).\n"
                 "Apply the patch from my reply to your model file first, so training and\n"
                 "inference crop faces identically. (Or pass --no_face_crop_check to skip.)")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(os.cpu_count() or 4)
    ckpt = Path(args.ckpt)
    ckpt.mkdir(parents=True, exist_ok=True)
    cache = Path(args.data_dir) / f"video_features_{args.n_per_class}_{args.seed}_paired.npz"

    if cache.exists():
        print(f"Loading cached features from {cache}")
        z = np.load(cache)
        SP, FR, Y, S = z["SP"], z["FR"], z["Y"], z["S"]
    else:
        zf = open_zip(args)
        files = [i.filename for i in zf.infolist() if not i.is_dir()]
        print(f"Zip contains {len(files)} files")
        sel = select_videos(files, set(args.manipulations.split(",")), args.n_per_class, args.seed)
        items = [(f, lab, s) for s in range(3) for f, lab in sel[s]]
        print(f"Selected {len(items)} videos: " + ", ".join(
            f"{n}={len(sel[s])}" for s, n in enumerate(["train", "val", "test"])))
        print("Copying selected videos out of the zip (FaceForensics++ terms of use apply) ...")
        local_paths = extract_members(zf, [f for f, _, _ in items], args)
        zf.close()

        vfx, ifx = VideoFeatureExtractor(max_frames=T_FRAMES), ImageFeatureExtractor()
        SPs, FRs, Ys, Ss = [], [], [], []
        print("Extracting frames (face crop) ...")
        for k, (lp, (_, lab, s)) in enumerate(zip(local_paths, items)):
            try:
                sp, fr = clip_features(lp, vfx, ifx)
            except Exception as e:
                print(f"  skipped {lp}: {e}")
                continue
            SPs.append(sp); FRs.append(fr); Ys.append(lab); Ss.append(s)
            if (k + 1) % 50 == 0:
                print(f"  {k+1}/{len(items)}")
        SP, FR = np.stack(SPs), np.stack(FRs)
        Y, S = np.array(Ys, dtype=np.int64), np.array(Ss)
        np.savez(cache, SP=SP, FR=FR, Y=Y, S=S)

    def loader(s, shuffle):
        ids = np.where(S == s)[0]
        ds = TensorDataset(torch.from_numpy(SP[ids]), torch.from_numpy(FR[ids]),
                           torch.from_numpy(Y[ids]))
        return DataLoader(ds, batch_size=args.batch, shuffle=shuffle), len(ids)

    (tr, n_tr), (va, n_va), (te, n_te) = loader(0, True), loader(1, False), loader(2, False)
    print(f"train={n_tr} val={n_va} test={n_te} videos")

    model = VideoFrameCNN()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best, best_state, bad = -1.0, None, 0
    for ep in range(args.epochs):
        model.train()
        total, correct, n_frames = 0.0, 0, 0
        for sp, fr, y in tr:
            if not args.no_augment:                              # random horizontal flip per video
                flip = torch.rand(sp.shape[0]) < 0.5
                sp[flip] = sp[flip].flip(-1)
                fr[flip] = fr[flip].flip(-1)
            logits = frame_logits(model, sp, fr)                 # (B*T, 2)
            target = y.repeat_interleave(sp.shape[1])            # video label for each frame
            loss = F.cross_entropy(logits, target)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(target)
            correct += (logits.argmax(1) == target).sum().item()
            n_frames += len(target)
        v_s, v_y = predict(model, va)
        m = metrics(v_y, v_s)
        print(f"epoch {ep+1}/{args.epochs}  loss={total/n_frames:.4f}  "
              f"train_frame_acc={correct/n_frames:.3f}  "
              f"val_acc={m['acc']:.3f}  val_auc={m['auc']:.3f}")
        crit = m["auc"] if not np.isnan(m["auc"]) else m["acc"]
        if crit > best:
            best, best_state, bad = crit, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= args.patience:
                print("early stopping")
                break

    if best < 0.55:
        print("\nWARNING: validation AUC never rose above chance - the model did not learn. "
              "Do NOT use these results (and don't run fusion with them). Try more videos "
              "(--n_per_class) or a lower --lr.")
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), ckpt / "video.pt")
    v_s, v_y = predict(model, va)
    t_s, t_y = predict(model, te)
    tm = metrics(t_y, t_s)
    print(f"\nTEST RESULTS (video level, {n_te} videos): {tm}")

    with open(ckpt / "scores_video.json", "w") as f:
        json.dump({"val": {"scores": v_s, "labels": v_y},
                   "test": {"scores": t_s, "labels": t_y}}, f)
    mpath = ckpt / "metrics.json"
    allm = json.load(open(mpath)) if mpath.exists() else {}
    allm["video"] = tm
    json.dump(allm, open(mpath, "w"), indent=2)
    print(f"Saved {ckpt / 'video.pt'}, scores_video.json, metrics.json")


if __name__ == "__main__":
    main()