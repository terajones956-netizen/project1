"""
training_fusion.py - train the FusionLayer (Layer 4) that combines per-modality AI-scores.

Inputs (all in --ckpt, written by the per-modality training scripts):
    scores_text.json, scores_image.json, scores_audio.json, scores_video_effnet.json
each shaped {"val": {"scores": [...], "labels": [...]}, "test": {...}}  (label 1 = AI/fake).
Change the video file with --video_file (e.g. scores_video.json for the old CNN).

Why synthetic pairing: the four detectors were trained on four different datasets, so
there are no real clips that have a text AND an image AND audio AND video score. We build
fusion training samples by drawing one score per modality from that modality's pool, always
with the SAME label (all-real or all-fake), and randomly hiding modalities (mask = 0) so the
layer also learns the "only some inputs present" case that happens at inference.

No leakage: fusion is trained only from the VAL score pools and evaluated only on samples
drawn from the TEST score pools. Test scores are never used to pick anything.

Modalities whose own validation AUC is below --min_auc are switched off (always masked):
a detector that is at chance only adds noise to the fusion.

Run from the project root, venv active (or on Colab after uploading the score files):
    python training/training_fusion.py
    python training/training_fusion.py --video_file scores_video.json --min_auc 0.7

Outputs (in --ckpt): fusion.pt, fusion_info.json, metrics.json["fusion"]

Honest caveat: because scores are paired synthetically, the errors of the modalities are
independent by construction. Real clips (e.g. a deepfake with cloned audio) will have
correlated errors, so treat the fusion numbers as optimistic.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score

HERE = Path(__file__).resolve().parent
for p in (HERE.parent / "model", HERE.parent, HERE):
    sys.path.insert(0, str(p))
try:
    from model import FusionLayer, MultimodalDetector
except ImportError:
    from model import FusionLayer, MultimodalDetector

ORDER = ["text", "image", "audio", "video"]      # must match FusionLayer.forward
THRESHOLD = getattr(MultimodalDetector, "THRESHOLD", 0.55)


def load_scores(path):
    d = json.load(open(path))
    out = {}
    for split in ("val", "test"):
        s = np.asarray(d[split]["scores"], dtype=np.float64)
        y = np.asarray(d[split]["labels"], dtype=int)
        out[split] = {0: s[y == 0], 1: s[y == 1]}
    return out


def pool_auc(pool):
    y = np.r_[np.zeros(len(pool[0])), np.ones(len(pool[1]))]
    s = np.r_[pool[0], pool[1]]
    return float(roc_auc_score(y, s)) if len(set(y)) == 2 else float("nan")


def gen(pools, usable, n, rng, p_include=0.6, subset=None):
    """Build n fusion samples. Returns X (n,8) = [4 scores | 4 mask], y (n,).
    subset: fixed list of modality names to show (else each usable one is shown w.p. p_include)."""
    y = rng.integers(0, 2, n)
    scores = np.full((n, 4), 0.5, dtype=np.float32)
    mask = np.zeros((n, 4), dtype=np.float32)
    idx = [ORDER.index(m) for m in usable]
    if subset is not None:
        for m in subset:
            mask[:, ORDER.index(m)] = 1.0
    else:
        for j in idx:
            mask[:, j] = rng.random(n) < p_include
        empty = mask.sum(1) == 0
        if empty.any():
            mask[np.where(empty)[0], rng.choice(idx, empty.sum())] = 1.0
    for j in idx:
        pool = pools[ORDER[j]]
        for lab in (0, 1):
            sel = (mask[:, j] == 1) & (y == lab)
            if sel.any():
                scores[sel, j] = rng.choice(pool[lab], sel.sum())
    return np.concatenate([scores, mask], axis=1), y


def fusion_probs(fusion, X):
    with torch.no_grad():
        return F.softmax(fusion.net(torch.from_numpy(X)), dim=-1)[:, 1].numpy()


def mean_baseline(X):
    s, m = X[:, :4], X[:, 4:]
    return (s * m).sum(1) / np.maximum(m.sum(1), 1)


def report(name, y, p):
    return {"auc": float(roc_auc_score(y, p)),
            "acc": float(accuracy_score(y, (p >= THRESHOLD).astype(int)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints")
    ap.add_argument("--video_file", default="scores_video_effnet.json")
    ap.add_argument("--min_auc", type=float, default=0.65,
                    help="modalities with val AUC below this are switched off")
    ap.add_argument("--min_per_class", type=int, default=10)
    ap.add_argument("--n_train", type=int, default=20000)
    ap.add_argument("--n_test", type=int, default=5000)
    ap.add_argument("--p_include", type=float, default=0.6,
                    help="chance each modality is present in a training sample")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--noise", type=float, default=0.03,
                    help="gaussian jitter on training scores (stops memorising the small score pools)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    ckpt = Path(args.ckpt)
    files = {"text": "scores_text.json", "image": "scores_image.json",
             "audio": "scores_audio.json", "video": args.video_file}

    pools = {"val": {}, "test": {}}
    usable, excluded = [], {}
    print(f"{'modality':<8} {'val n (real/fake)':<18} {'test n':<10} {'val AUC':<8} {'test AUC':<8} status")
    for m in ORDER:
        path = ckpt / files[m]
        if not path.exists():
            excluded[m] = f"missing {files[m]}"
            print(f"{m:<8} {'-':<18} {'-':<10} {'-':<8} {'-':<8} OFF (file missing: {files[m]})")
            continue
        sc = load_scores(path)
        v, t = sc["val"], sc["test"]
        va, ta = pool_auc(v), pool_auc(t)
        nv, nt = (len(v[0]), len(v[1])), (len(t[0]), len(t[1]))
        if min(nv + nt) < args.min_per_class:
            status = f"OFF (fewer than {args.min_per_class} per class)"
            excluded[m] = "too few samples"
        elif not va >= args.min_auc:
            status = f"OFF (val AUC < {args.min_auc})"
            excluded[m] = f"val AUC {va:.3f} below {args.min_auc}"
        else:
            status = "ON"
            usable.append(m)
            pools["val"][m], pools["test"][m] = v, t
        print(f"{m:<8} {str(nv):<18} {str(nt):<10} {va:<8.3f} {ta:<8.3f} {status}")

    if not usable:
        sys.exit("\nNo usable modality scores found - nothing to fuse. Train the detectors first.")
    if len(usable) == 1:
        print(f"\nNOTE: only '{usable[0]}' is usable, so fusion can't add anything beyond that detector.")

    # ---- train on VAL-pool samples ----
    Xtr, ytr = gen(pools["val"], usable, args.n_train, rng, args.p_include)
    Xtr_t, ytr_t = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    fusion = FusionLayer()
    opt = torch.optim.AdamW(fusion.parameters(), lr=args.lr, weight_decay=1e-3)
    print(f"\nTraining fusion on {len(ytr)} synthetic samples from the val pools ...")
    for ep in range(args.epochs):
        perm = torch.randperm(len(ytr_t))
        tot, correct = 0.0, 0
        for i in range(0, len(perm), 256):
            idx = perm[i:i + 256]
            xb, yb = Xtr_t[idx].clone(), ytr_t[idx]
            xb[:, :4] = (xb[:, :4] + torch.randn_like(xb[:, :4]) * args.noise * xb[:, 4:]).clamp(0, 1)
            logits = fusion.net(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
            correct += (logits.argmax(1) == yb).sum().item()
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  epoch {ep+1}/{args.epochs}  loss={tot/len(ytr_t):.4f}  train_acc={correct/len(ytr_t):.3f}")
    fusion.eval()

    # ---- evaluate on TEST-pool samples (never used for training) ----
    print(f"\nEvaluation on samples drawn from the TEST pools (threshold {THRESHOLD}):")
    print(f"{'inputs shown':<28} {'fusion AUC':<11} {'fusion acc':<11} {'mean-score AUC':<15} {'mean-score acc'}")
    rows = {}
    subsets = [("random mix", None)] + [(f"{m} only", [m]) for m in usable]
    if len(usable) > 1:
        subsets.append(("all usable", list(usable)))
    for name, sub in subsets:
        X, y = gen(pools["test"], usable, args.n_test, rng, args.p_include, subset=sub)
        f = report(name, y, fusion_probs(fusion, X))
        b = report(name, y, mean_baseline(X))
        rows[name] = {"fusion": f, "mean_baseline": b}
        print(f"{name:<28} {f['auc']:<11.3f} {f['acc']:<11.3f} {b['auc']:<15.3f} {b['acc']:.3f}")
    print("\nIf 'fusion' is not clearly better than 'mean-score' on the mixed rows, the plain average "
          "is just as good - report that honestly.")
    print("Reminder: modalities are paired synthetically (independent errors), so these numbers are optimistic.")

    torch.save(fusion.state_dict(), ckpt / "fusion.pt")
    info = {"usable": usable, "excluded": excluded, "threshold": THRESHOLD,
            "video_file": args.video_file, "eval": rows}
    json.dump(info, open(ckpt / "fusion_info.json", "w"), indent=2)
    mpath = ckpt / "metrics.json"
    allm = json.load(open(mpath)) if mpath.exists() else {}
    allm["fusion"] = rows.get("all usable", rows["random mix"])["fusion"]
    json.dump(allm, open(mpath, "w"), indent=2)
    print(f"\nSaved {ckpt / 'fusion.pt'}, fusion_info.json, metrics.json")
    if excluded:
        print("At inference, do NOT pass these modalities to the detector (fusion never saw them): "
              + ", ".join(excluded))


if __name__ == "__main__":
    main()