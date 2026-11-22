
"""
training_audio.py - Train AudioSpectrogramCNN voice-deepfake detector.

Features:
1. Hugging Face dataset or custom local audio folder.
2. Audio feature extraction using AudioFeatureExtractor.
3. Train / validation / test split.
4. Epoch-wise training and validation accuracy.
5. Epoch-wise training and validation loss.
6. Early stopping based on validation AUC.
7. Save trained model and evaluation metrics.
8. Automatically generate accuracy and loss curves (300 DPI).

Run from project root:

    python training/training_audio.py

Quick test:

    python training/training_audio.py --max_per_class 500

Custom dataset:

    python training/training_audio.py --folder D:\\data\\audio

Outputs:
    checkpoints/audio.pt
    checkpoints/scores_audio.json
    checkpoints/metrics.json
    checkpoints/audio_training_history.json
    checkpoints/audio_accuracy_curve.png
    checkpoints/audio_loss_curve.png
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
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix
)

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


# --------------------------------------------------
# PATH CONFIGURATION
# --------------------------------------------------

HERE = Path(__file__).resolve().parent

for p in (HERE.parent / "model", HERE.parent, HERE):
    sys.path.insert(0, str(p))

try:
    from model import AudioFeatureExtractor, AudioSpectrogramCNN
except ImportError:
    from model import AudioFeatureExtractor, AudioSpectrogramCNN


logging.getLogger("httpx").setLevel(logging.WARNING)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


# --------------------------------------------------
# CONSTANTS
# --------------------------------------------------

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}

FAKE = {
    "fake", "spoof", "spoofed", "ai",
    "synthetic", "generated", "deepfake", "tts"
}

REAL = {
    "real", "human", "bonafide",
    "bona_fide", "genuine", "authentic"
}

SR = 16000


# --------------------------------------------------
# METRICS
# --------------------------------------------------

def metrics(y, p):

    y = np.asarray(y)
    p = np.asarray(p)

    pred = (p >= 0.5).astype(int)

    out = {
        "acc": float(accuracy_score(y, pred)),

        "precision": float(
            precision_score(y, pred, zero_division=0)
        ),

        "recall": float(
            recall_score(y, pred, zero_division=0)
        ),

        "f1": float(
            f1_score(y, pred, zero_division=0)
        )
    }

    try:
        out["auc"] = float(roc_auc_score(y, p))
    except ValueError:
        out["auc"] = float("nan")

    tn, fp, fn, tp = confusion_matrix(
        y, pred, labels=[0, 1]
    ).ravel()

    out.update({
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn)
    })

    return out


# --------------------------------------------------
# DOWNLOAD HUGGING FACE DATASET
# --------------------------------------------------

def prepare_hf(args):

    from huggingface_hub import (
        list_repo_files,
        snapshot_download
    )

    files = list_repo_files(
        args.dataset,
        repo_type="dataset"
    )

    audio = [
        f for f in files
        if Path(f).suffix.lower() in AUDIO_EXTS
    ]

    if not audio:
        raise RuntimeError(
            "No audio files found in dataset repository."
        )

    print(
        f"Dataset contains {len(audio)} audio files."
    )

    local = Path(args.data_dir) / "audio_hf"

    local.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        args.dataset,
        repo_type="dataset",
        local_dir=str(local),
        allow_patterns=[
            f"*{e}" for e in AUDIO_EXTS
        ]
    )

    return local


# --------------------------------------------------
# COLLECT AUDIO FILES AND LABELS
# --------------------------------------------------

def collect(root):

    items = []
    unlabeled = []

    for p in Path(root).rglob("*"):

        if p.suffix.lower() not in AUDIO_EXTS:
            continue

        parts = {
            x.lower()
            for x in p.relative_to(root).parts[:-1]
        }

        is_fake = bool(parts & FAKE)
        is_real = bool(parts & REAL)

        if is_fake != is_real:

            items.append(
                (str(p), int(is_fake))
            )

        else:
            unlabeled.append(str(p))

    if not items:

        raise RuntimeError(
            "Could not infer labels from folder names.\n"
            "Please organize audio files into real/ and fake/ folders."
        )

    n_fake = sum(label for _, label in items)

    print(
        f"Found {len(items)} labelled clips "
        f"({n_fake} fake / {len(items)-n_fake} real)."
    )

    print(
        f"Skipped unlabeled clips: {len(unlabeled)}"
    )

    return items


# --------------------------------------------------
# FEATURE EXTRACTION
# --------------------------------------------------

def build_features(items, args):

    rng = np.random.default_rng(args.seed)

    chosen = []

    for lab in (0, 1):

        cls = [
            it for it in items
            if it[1] == lab
        ]

        if (
            args.max_per_class
            and len(cls) > args.max_per_class
        ):

            indices = rng.choice(
                len(cls),
                args.max_per_class,
                replace=False
            )

            cls = [cls[i] for i in indices]

        chosen += cls

    fx = AudioFeatureExtractor(sr=SR)

    specs = []
    ys = []

    bad = 0

    print(
        f"\nExtracting features for {len(chosen)} clips..."
    )

    for k, (path, lab) in enumerate(chosen):

        try:

            y, _ = librosa.load(
                path,
                sr=SR,
                mono=True,
                duration=args.max_sec
            )

            if len(y) < 1600:

                y = np.pad(
                    y,
                    (0, 1600 - len(y))
                )

            feature = fx.extract(y, SR)

            specs.append(feature)
            ys.append(lab)

        except Exception as e:

            bad += 1

            if bad <= 3:

                print(
                    f"Skipped {path}: {e}"
                )

        if (k + 1) % 500 == 0:

            print(
                f"Processed {k+1}/{len(chosen)}"
            )

    if not specs:

        raise RuntimeError(
            "No valid audio features were extracted."
        )

    if bad:

        print(
            f"Skipped {bad} unreadable audio clips."
        )

    X = np.stack(specs)[:, None, :, :].astype(
        np.float16
    )

    Y = np.array(
        ys,
        dtype=np.int64
    )

    return X, Y


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

@torch.no_grad()
def predict(model, loader):

    model.eval()

    scores = []
    ys = []

    for x, y in loader:

        probs = model(x.float())

        scores += probs[:, 1].tolist()

        ys += y.tolist()

    return scores, ys


# --------------------------------------------------
# CALCULATE VALIDATION LOSS
# --------------------------------------------------

@torch.no_grad()
def calculate_loss(model, loader):

    model.eval()

    total_loss = 0.0
    total_samples = 0

    for x, y in loader:

        probs = model(x.float())

        loss = F.nll_loss(
            torch.log(probs + 1e-9),
            y
        )

        total_loss += loss.item() * len(y)

        total_samples += len(y)

    return total_loss / total_samples


# --------------------------------------------------
# GENERATE ACCURACY AND LOSS CURVES
# --------------------------------------------------

def generate_curves(history, ckpt):

    epochs = range(
        1,
        len(history["train_accuracy"]) + 1
    )

    # ----------------------------------------------
    # ACCURACY CURVE
    # ----------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs,
        history["train_accuracy"],
        label="Training Accuracy",
        marker="o",
        linewidth=2
    )

    plt.plot(
        epochs,
        history["val_accuracy"],
        label="Validation Accuracy",
        marker="o",
        linewidth=2
    )

    plt.title(
        "Audio Deepfake Detection - Accuracy Curve",
        fontsize=14
    )

    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)

    plt.xticks(list(epochs))

    plt.ylim(0, 1.05)

    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    accuracy_path = (
        ckpt / "audio_accuracy_curve.png"
    )

    plt.savefig(
        accuracy_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ----------------------------------------------
    # LOSS CURVE
    # ----------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs,
        history["train_loss"],
        label="Training Loss",
        marker="o",
        linewidth=2
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation Loss",
        marker="o",
        linewidth=2
    )

    plt.title(
        "Audio Deepfake Detection - Loss Curve",
        fontsize=14
    )

    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss", fontsize=12)

    plt.xticks(list(epochs))

    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    loss_path = (
        ckpt / "audio_loss_curve.png"
    )

    plt.savefig(
        loss_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\nTraining curves saved:")

    print(f"Accuracy: {accuracy_path}")
    print(f"Loss: {loss_path}")


# --------------------------------------------------
# MAIN TRAINING FUNCTION
# --------------------------------------------------

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--ckpt",
        default="checkpoints"
    )

    ap.add_argument(
        "--dataset",
        default="garystafford/deepfake-audio-detection"
    )

    ap.add_argument(
        "--data_dir",
        default="data"
    )

    ap.add_argument(
        "--folder",
        default=None
    )

    ap.add_argument(
        "--max_per_class",
        type=int,
        default=0
    )

    ap.add_argument(
        "--max_sec",
        type=float,
        default=5.0
    )

    ap.add_argument(
        "--epochs",
        type=int,
        default=20
    )

    ap.add_argument(
        "--batch",
        type=int,
        default=32
    )

    ap.add_argument(
        "--lr",
        type=float,
        default=1e-3
    )

    ap.add_argument(
        "--patience",
        type=int,
        default=5
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = ap.parse_args()

    # ----------------------------------------------
    # REPRODUCIBILITY
    # ----------------------------------------------

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.set_num_threads(
        os.cpu_count() or 4
    )

    # ----------------------------------------------
    # CHECKPOINT DIRECTORY
    # ----------------------------------------------

    ckpt = Path(args.ckpt)

    ckpt.mkdir(
        parents=True,
        exist_ok=True
    )

    # ----------------------------------------------
    # DATASET
    # ----------------------------------------------

    if args.folder:

        root = Path(args.folder)

    else:

        root = prepare_hf(args)

    X, Y = build_features(
        collect(root),
        args
    )

    # Ensure both classes exist
    if len(np.unique(Y)) != 2:

        raise RuntimeError(
            "Dataset must contain both real and fake audio."
        )

    # ----------------------------------------------
    # TRAIN / VALIDATION / TEST SPLIT
    # ----------------------------------------------

    idx = np.arange(len(Y))

    tr_i, tmp_i = train_test_split(
        idx,
        test_size=0.30,
        stratify=Y,
        random_state=args.seed
    )

    va_i, te_i = train_test_split(
        tmp_i,
        test_size=0.50,
        stratify=Y[tmp_i],
        random_state=args.seed
    )

    print("\nDataset split:")

    print(f"Training samples:   {len(tr_i)}")
    print(f"Validation samples: {len(va_i)}")
    print(f"Testing samples:    {len(te_i)}")

    # ----------------------------------------------
    # DATA LOADERS
    # ----------------------------------------------

    def loader(ids, shuffle):

        return DataLoader(
            TensorDataset(
                torch.from_numpy(X[ids]),
                torch.from_numpy(Y[ids])
            ),
            batch_size=args.batch,
            shuffle=shuffle
        )

    tr = loader(tr_i, True)
    va = loader(va_i, False)
    te = loader(te_i, False)

    # ----------------------------------------------
    # MODEL
    # ----------------------------------------------

    model = AudioSpectrogramCNN()

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    # ----------------------------------------------
    # TRAINING HISTORY
    # ----------------------------------------------

    history = {

        "train_accuracy": [],
        "val_accuracy": [],

        "train_loss": [],
        "val_loss": []
    }

    best = -1.0
    best_state = None
    bad = 0

    print("\nStarting Audio CNN training...\n")

    # ----------------------------------------------
    # TRAINING LOOP
    # ----------------------------------------------

    for ep in range(args.epochs):

        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in tr:

            probs = model(x.float())

            loss = F.nll_loss(
                torch.log(probs + 1e-9),
                y
            )

            opt.zero_grad()

            loss.backward()

            opt.step()

            total_loss += loss.item() * len(y)

            correct += (
                probs.argmax(dim=1) == y
            ).sum().item()

            total += len(y)

        # Training metrics
        train_loss = total_loss / total

        train_acc = correct / total

        # Validation metrics
        v_s, v_y = predict(
            model,
            va
        )

        val_loss = calculate_loss(
            model,
            va
        )

        val_acc = accuracy_score(
            v_y,
            (np.array(v_s) >= 0.5).astype(int)
        )

        # Save history
        history["train_accuracy"].append(
            float(train_acc)
        )

        history["val_accuracy"].append(
            float(val_acc)
        )

        history["train_loss"].append(
            float(train_loss)
        )

        history["val_loss"].append(
            float(val_loss)
        )

        # Validation AUC
        m = metrics(
            v_y,
            v_s
        )

        print(
            f"Epoch {ep+1}/{args.epochs} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val AUC: {m['auc']:.4f}"
        )

        # ------------------------------------------
        # EARLY STOPPING
        # ------------------------------------------

        crit = (
            m["auc"]
            if not np.isnan(m["auc"])
            else m["acc"]
        )

        if crit > best:

            best = crit

            best_state = copy.deepcopy(
                model.state_dict()
            )

            bad = 0

        else:

            bad += 1

            if bad >= args.patience:

                print("\nEarly stopping triggered.")

                break

    # ----------------------------------------------
    # SAVE TRAINING HISTORY
    # ----------------------------------------------

    history_path = (
        ckpt / "audio_training_history.json"
    )

    with open(history_path, "w") as f:

        json.dump(
            history,
            f,
            indent=4
        )

    # ----------------------------------------------
    # GENERATE CURVES
    # ----------------------------------------------

    generate_curves(
        history,
        ckpt
    )

    # ----------------------------------------------
    # LOAD BEST MODEL
    # ----------------------------------------------

    if best_state is None:

        raise RuntimeError(
            "Training did not produce a valid model checkpoint."
        )

    model.load_state_dict(best_state)

    # ----------------------------------------------
    # SAVE TRAINED MODEL
    # ----------------------------------------------

    torch.save(
        model.state_dict(),
        ckpt / "audio.pt"
    )

    # ----------------------------------------------
    # FINAL VALIDATION AND TEST RESULTS
    # ----------------------------------------------

    v_s, v_y = predict(
        model,
        va
    )

    t_s, t_y = predict(
        model,
        te
    )

    tm = metrics(
        t_y,
        t_s
    )

    print("\n========== TEST RESULTS ==========")

    for key, value in tm.items():

        print(f"{key}: {value}")

    # ----------------------------------------------
    # SAVE PREDICTION SCORES
    # ----------------------------------------------

    with open(
        ckpt / "scores_audio.json",
        "w"
    ) as f:

        json.dump(
            {
                "val": {
                    "scores": v_s,
                    "labels": v_y
                },

                "test": {
                    "scores": t_s,
                    "labels": t_y
                }
            },
            f,
            indent=4
        )

    # ----------------------------------------------
    # UPDATE METRICS JSON
    # ----------------------------------------------

    mpath = ckpt / "metrics.json"

    if mpath.exists():

        with open(mpath, "r") as f:

            allm = json.load(f)

    else:

        allm = {}

    allm["audio"] = tm

    with open(mpath, "w") as f:

        json.dump(
            allm,
            f,
            indent=4
        )

    # ----------------------------------------------
    # COMPLETION
    # ----------------------------------------------

    print("\n========== TRAINING COMPLETED ==========")

    print(f"Model: {ckpt / 'audio1.pt'}")

    print(
        f"Accuracy curve: "
        f"{ckpt / 'audio_accuracy_curve.png'}"
    )

    print(
        f"Loss curve: "
        f"{ckpt / 'audio_loss_curve.png'}"
    )

    print(
        f"Training history: "
        f"{ckpt / 'audio_training_history.json'}"
    )


# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":

    main()
