"""
evaluate_provenance.py - evaluate "Fusion + provenance/trust evidence".

Compares, on the SAME labelled test samples:
    (a) fusion only                      -> fusion_p
    (b) fusion + provenance/trust rules  -> final_score(fusion_p, prov)

Provenance rules are tuned ONLY on the validation split (small grid search);
the test split is used once for the final report. No test data picks anything.

INPUT  (default: checkpoints/provenance_eval.json)
    {
      "val":  {"y": [0,1,...], "fusion_p": [0.12,0.97,...], "prov": [{...}, {...}, ...]},
      "test": {"y": [...],     "fusion_p": [...],           "prov": [{...}, ...]}
    }
    y          : 1 = AI/fake, 0 = real
    fusion_p   : probability of fake from your FusionLayer for that sample
    prov       : one dict per sample (every key optional, use null/omit if unknown):
        "ai_watermark"      : 1 if an AI watermark/tag was detected, else 0
        "c2pa"              : "camera" (valid credentials, camera captured)
                              "ai"     (valid credentials, declares AI generation)
                              "invalid"(credentials present but broken/forged)
                              "none" / null (no credentials)
        "metadata_stripped" : 1 if metadata was removed, else 0
        "source_trust"      : 0..1 reputation of the source (1 = very trusted)

OUTPUT (in --ckpt)
    metrics.json["fusion_provenance"]   final metrics (test, fusion + provenance)
    provenance_info.json                tuned parameters, coverage, both rows, CIs

Usage (from the project root):
    python training/evaluate_provenance.py
    python training/evaluate_provenance.py --template      # write an example input file
    python training/evaluate_provenance.py --demo          # pipeline self-test (SIMULATED data)

HONEST CAVEATS
  * Provenance signals must come from REAL files/metadata. If you generate them
    from the label (e.g. "fakes always get a watermark"), the result is circular
    and approaches 100% by construction. --demo does exactly that, so it only
    tests that the script runs; it never writes metrics.json.
  * Most real content has NO provenance. Look at the coverage line and at the
    covered / uncovered split before claiming an overall gain.
  * Missing provenance is not evidence of fakeness; keep stripped-metadata and
    trust weights small. The tuner keeps the neutral setting unless validation
    improves by at least --min_gain.
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

DEFAULT_THRESHOLD = 0.55          # same as training_fusion.py (MultimodalDetector.THRESHOLD)

NEUTRAL = {"camera_scale": 1.0, "invalid_bump": 0.0, "stripped_bump": 0.0,
           "trust_weight": 0.0, "ai_floor": 0.95, "wm_floor": 0.99}

GRID = {
    "camera_scale": [1.0, 0.7, 0.5, 0.3],      # valid camera credentials lower the score
    "invalid_bump": [0.0, 0.1, 0.2],           # broken/forged credentials raise it
    "stripped_bump": [0.0, 0.03, 0.05, 0.1],   # stripped metadata raises it a little
    "trust_weight": [0.0, 0.1, 0.2, 0.4],      # source reputation shifts it
}


# --------------------------------------------------------------------------
# Provenance -> final score
# --------------------------------------------------------------------------
def final_score(fusion_p, prov, params):
    """Return the adjusted probability of fake in [0, 1].
    Copy this function into model/provenance.py (or the detector class) so that
    inference and evaluation use identical logic."""
    p = float(fusion_p)
    prov = prov or {}

    c2pa = prov.get("c2pa")
    if c2pa == "camera":
        p *= params["camera_scale"]
    elif c2pa == "invalid":
        p += params["invalid_bump"]

    trust = prov.get("source_trust")
    if trust is not None:
        p -= params["trust_weight"] * (float(trust) - 0.5)

    if prov.get("metadata_stripped") == 1:
        p += params["stripped_bump"]

    p = min(max(p, 0.0), 1.0)

    # Strong positive AI evidence wins over everything above.
    if c2pa == "ai":
        p = max(p, params["ai_floor"])
    if prov.get("ai_watermark") == 1:
        p = max(p, params["wm_floor"])
    return p


def has_signal(prov):
    prov = prov or {}
    return (prov.get("ai_watermark") == 1
            or prov.get("c2pa") in ("camera", "ai", "invalid")
            or prov.get("source_trust") is not None
            or prov.get("metadata_stripped") == 1)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def evaluate(y, p, thr):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    pred = (p >= thr).astype(int)
    out = {
        "acc": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),   # positive = fake (1)
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auc": float(roc_auc_score(y, p)) if len(set(y.tolist())) == 2 else float("nan"),
    }
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out.update({"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)})
    return out


def bootstrap_ci(y, p, thr, n_boot=1000, seed=0):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    rng = np.random.default_rng(seed)
    aucs, f1s = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i].tolist())) < 2:
            continue
        aucs.append(roc_auc_score(y[i], p[i]))
        f1s.append(f1_score(y[i], (p[i] >= thr).astype(int), zero_division=0))
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return {"auc_ci95": q(aucs), "f1_ci95": q(f1s)}


def apply(split, params):
    return np.array([final_score(fp, pv, params)
                     for fp, pv in zip(split["fusion_p"], split["prov"])])


# --------------------------------------------------------------------------
# Tuning (validation only)
# --------------------------------------------------------------------------
def tune(val, thr, min_gain):
    y = np.asarray(val["y"], dtype=int)
    base = evaluate(y, val["fusion_p"], thr)
    best, best_params, best_m = None, dict(NEUTRAL), base
    keys = list(GRID)
    for combo in itertools.product(*[GRID[k] for k in keys]):
        params = {**NEUTRAL, **dict(zip(keys, combo))}
        m = evaluate(y, apply(val, params), thr)
        score = (m["f1"], m["auc"])
        if best is None or score > best:
            best, best_params, best_m = score, params, m
    gain = best_m["f1"] - base["f1"]
    if gain < min_gain:
        print(f"Tuning: best val F1 gain {gain:+.4f} < min_gain {min_gain}; keeping neutral rules "
              f"(provenance adds only the hard rules: AI watermark / AI credentials).")
        best_params, best_m = dict(NEUTRAL), evaluate(y, apply(val, NEUTRAL), thr)
    return best_params, base, best_m


# --------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------
def load_split(d, name):
    s = d[name]
    n = len(s["y"])
    if not (len(s["fusion_p"]) == n and len(s["prov"]) == n):
        sys.exit(f"'{name}': y, fusion_p and prov must have the same length.")
    if len(set(s["y"])) < 2:
        sys.exit(f"'{name}' needs both classes (0 and 1).")
    return s


def write_template(path):
    ex = {"y": [0, 1, 1, 0], "fusion_p": [0.20, 0.90, 0.40, 0.65],
          "prov": [{"c2pa": "camera", "source_trust": 0.9},
                   {"ai_watermark": 1},
                   {"metadata_stripped": 1, "source_trust": 0.2},
                   {}]}
    json.dump({"val": ex, "test": ex}, open(path, "w"), indent=2)
    print(f"Wrote example input to {path} (replace with your real labelled samples).")


def make_demo(n=600, seed=0):
    """SIMULATED data: provenance is generated from the label, so results are
    circular. Used only to check that the script runs end to end."""
    rng = np.random.default_rng(seed)

    def split(n):
        y = rng.integers(0, 2, n)
        fp = np.clip(np.where(y == 1, rng.beta(5, 2, n), rng.beta(2, 5, n)), 0, 1)
        prov = []
        for lab in y:
            pv = {}
            if lab == 1 and rng.random() < 0.25:
                pv["ai_watermark"] = 1
            if lab == 0 and rng.random() < 0.25:
                pv["c2pa"] = "camera"
            if rng.random() < 0.3:
                pv["metadata_stripped"] = int(rng.random() < (0.7 if lab == 1 else 0.3))
            prov.append(pv)
        return {"y": y.tolist(), "fusion_p": fp.tolist(), "prov": prov}
    return {"val": split(n), "test": split(n)}


def fmt(m):
    return (f"acc {m['acc']:.4f}  prec {m['precision']:.4f}  rec {m['recall']:.4f}  "
            f"F1 {m['f1']:.4f}  AUC {m['auc']:.4f}  (TP {m['tp']} FP {m['fp']} TN {m['tn']} FN {m['fn']})")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints")
    ap.add_argument("--input", default=None, help="default: <ckpt>/provenance_eval.json")
    ap.add_argument("--threshold", type=float, default=None,
                    help=f"decision threshold (default: fusion_info.json value, else {DEFAULT_THRESHOLD})")
    ap.add_argument("--min_gain", type=float, default=0.005,
                    help="min validation F1 gain required to keep tuned (non-neutral) rules")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--template", action="store_true", help="write an example input file and exit")
    ap.add_argument("--demo", action="store_true",
                    help="run on SIMULATED data to test the script (does not save metrics.json)")
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    ckpt.mkdir(parents=True, exist_ok=True)
    in_path = Path(args.input) if args.input else ckpt / "provenance_eval.json"

    if args.template:
        write_template(ckpt / "provenance_eval.example.json")
        return

    if args.demo:
        data = make_demo()
        print("*** DEMO MODE: simulated, circular provenance. Numbers are NOT results. ***\n")
    else:
        if not in_path.exists():
            sys.exit(f"{in_path} not found. Run with --template to see the expected format, "
                     f"or --demo to test the script.")
        data = json.load(open(in_path))

    val, test = load_split(data, "val"), load_split(data, "test")

    thr = args.threshold
    if thr is None:
        info = ckpt / "fusion_info.json"
        thr = json.load(open(info)).get("threshold", DEFAULT_THRESHOLD) if info.exists() else DEFAULT_THRESHOLD
    print(f"Threshold: {thr}")
    print(f"Samples: val={len(val['y'])}  test={len(test['y'])}")

    cov_test = float(np.mean([has_signal(p) for p in test["prov"]]))
    cov_val = float(np.mean([has_signal(p) for p in val["prov"]]))
    print(f"Provenance coverage (any usable signal): val {cov_val:.1%}  test {cov_test:.1%}")

    # ---- tune on VAL only ----
    params, val_base, val_prov = tune(val, thr, args.min_gain)
    print("\nChosen rule parameters:", {k: params[k] for k in GRID} |
          {"ai_floor": params["ai_floor"], "wm_floor": params["wm_floor"]})
    print("VAL  fusion only       :", fmt(val_base))
    print("VAL  fusion+provenance :", fmt(val_prov))

    # ---- evaluate on TEST once ----
    yt = np.asarray(test["y"], dtype=int)
    p_fus = np.asarray(test["fusion_p"], dtype=float)
    p_fin = apply(test, params)
    m_fus, m_fin = evaluate(yt, p_fus, thr), evaluate(yt, p_fin, thr)
    ci_fus = bootstrap_ci(yt, p_fus, thr, args.n_boot)
    ci_fin = bootstrap_ci(yt, p_fin, thr, args.n_boot)

    print("\nTEST fusion only       :", fmt(m_fus))
    print("TEST fusion+provenance :", fmt(m_fin))
    print(f"  95% CI  fusion only        AUC {ci_fus['auc_ci95']}  F1 {ci_fus['f1_ci95']}")
    print(f"  95% CI  fusion+provenance  AUC {ci_fin['auc_ci95']}  F1 {ci_fin['f1_ci95']}")

    # ---- covered vs uncovered ----
    cov = np.array([has_signal(p) for p in test["prov"]])
    subsets = {}
    for name, sel in (("covered", cov), ("uncovered", ~cov)):
        if sel.sum() >= 10 and len(set(yt[sel].tolist())) == 2:
            subsets[name] = {"n": int(sel.sum()),
                             "fusion_only": evaluate(yt[sel], p_fus[sel], thr),
                             "fusion_provenance": evaluate(yt[sel], p_fin[sel], thr)}
            print(f"\nTEST {name} (n={int(sel.sum())})")
            print("  fusion only       :", fmt(subsets[name]["fusion_only"]))
            print("  fusion+provenance :", fmt(subsets[name]["fusion_provenance"]))
        else:
            print(f"\nTEST {name}: too few samples or one class only; skipped.")

    print("\nIf 'fusion+provenance' is not clearly better than 'fusion only' (check the CIs and the "
          "covered/uncovered split), report that honestly.")

    if args.demo:
        print("\nDemo mode: nothing saved.")
        return

    info = {"threshold": thr, "params": params, "coverage": {"val": cov_val, "test": cov_test},
            "val": {"fusion_only": val_base, "fusion_provenance": val_prov},
            "test": {"fusion_only": m_fus, "fusion_provenance": m_fin,
                     "ci_fusion_only": ci_fus, "ci_fusion_provenance": ci_fin,
                     "subsets": subsets}}
    json.dump(info, open(ckpt / "provenance_info.json", "w"), indent=2)
    mpath = ckpt / "metrics.json"
    allm = json.load(open(mpath)) if mpath.exists() else {}
    allm["fusion_provenance"] = m_fin
    json.dump(allm, open(mpath, "w"), indent=2)
    print(f"\nSaved {ckpt / 'provenance_info.json'} and metrics.json['fusion_provenance']")


if __name__ == "__main__":
    main()