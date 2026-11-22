"""
training_text_politifact.py - fine-tune DistilRoBERTa on PolitiFact-rated claims (3 epochs).

IMPORTANT: this trains a CLAIM CREDIBILITY classifier (false/misleading vs true), NOT an
AI-generated-text detector. It is saved under separate names so it never overwrites your
HC3 AI-text detector (checkpoints/text, scores_text.json):
    checkpoints/text_claims/            model + tokenizer
    checkpoints/scores_text_claims.json {"val": {...}, "test": {...}}  (label 1 = false/misleading)
    checkpoints/metrics.json["text_claims"]

Data (pick one):
    python training/training_text_politifact.py --csv data/politifact.csv --text_col statement --label_col label
    python training/training_text_politifact.py --hf <hf_dataset_id> --text_col statement --label_col label
Any PolitiFact-derived data works (e.g. the LIAR dataset). Check the dataset's licence and
cite it. Labels are normalised from names such as pants-fire / false / barely-true /
half-true / mostly-true / true:
    1 (false/misleading) = pants-fire, false, barely-true, mostly-false
    0 (true)             = mostly-true, true
    half-true is dropped (ambiguous).

Run:   python training/training_text_politifact.py --epochs 3
"""

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

FALSE_SET = {"pants-fire", "pants on fire", "pants_fire", "false",
             "barely-true", "barely true", "mostly-false", "mostly false"}
TRUE_SET = {"true", "mostly-true", "mostly true"}


def norm_label(x):
    return str(x).strip().lower().replace("_", "-")


def to_binary(x):
    s = norm_label(x)
    if s in FALSE_SET or s.replace("-", " ") in FALSE_SET:
        return 1
    if s in TRUE_SET or s.replace("-", " ") in TRUE_SET:
        return 0
    return None            # half-true / unknown -> dropped


def load_frame(args):
    if args.csv:
        df = pd.read_csv(args.csv)
    else:
        from datasets import ClassLabel, concatenate_datasets, load_dataset
        ds = load_dataset(args.hf)
        parts = [ds[k] for k in ds.keys()]
        feat = parts[0].features.get(args.label_col)
        full = concatenate_datasets(parts)
        df = full.to_pandas()
        if isinstance(feat, ClassLabel):
            df[args.label_col] = [feat.names[i] for i in df[args.label_col]]
    for c in (args.text_col, args.label_col):
        if c not in df.columns:
            raise SystemExit(f"Column '{c}' not found. Columns: {list(df.columns)}")
    df = df[[args.text_col, args.label_col]].rename(
        columns={args.text_col: "text", args.label_col: "raw"})
    df["label"] = df["raw"].map(to_binary)
    n0 = len(df)
    df = df.dropna(subset=["text", "label"]).drop_duplicates(subset="text")   # dedupe avoids leakage
    df["label"] = df["label"].astype(int)
    print(f"Rows: {n0} -> {len(df)} after dropping half-true/unknown and duplicates; "
          f"false/misleading={int(df.label.sum())}, true={int((df.label == 0).sum())}")
    return df.reset_index(drop=True)


def metrics(y, p, thr=0.5):
    y, p = np.asarray(y), np.asarray(p)
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"acc": float(accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "auc": float(roc_auc_score(y, p)) if len(set(y.tolist())) == 2 else float("nan"),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}


class Ds(torch.utils.data.Dataset):
    def __init__(self, enc, labels):
        self.enc, self.labels = enc, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(int(self.labels[i]))
        return item


def softmax_pos(logits):
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return (e / e.sum(axis=1, keepdims=True))[:, 1]


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--csv")
    g.add_argument("--hf", help="Hugging Face dataset id")
    ap.add_argument("--text_col", default="statement")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--ckpt", default="checkpoints")
    ap.add_argument("--model", default="distilroberta-base")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    ckpt = Path(args.ckpt)
    ckpt.mkdir(parents=True, exist_ok=True)

    df = load_frame(args)
    tr, tmp = train_test_split(df, test_size=0.2, stratify=df.label, random_state=args.seed)
    va, te = train_test_split(tmp, test_size=0.5, stratify=tmp.label, random_state=args.seed)
    print(f"train={len(tr)} val={len(va)} test={len(te)}")

    tok = AutoTokenizer.from_pretrained(args.model)

    def enc(d):
        return Ds(tok(d.text.tolist(), truncation=True, max_length=args.max_len,
                      padding="max_length"), d.label.tolist())

    ds_tr, ds_va, ds_te = enc(tr), enc(va), enc(te)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

    def compute(ev):
        return metrics(ev.label_ids, softmax_pos(ev.predictions))

    ev_key = ("eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters
              else "evaluation_strategy")
    targs = TrainingArguments(
        output_dir=str(ckpt / "_tmp_text_claims"), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch, per_device_eval_batch_size=64,
        learning_rate=args.lr, weight_decay=0.01, logging_steps=50, report_to="none",
        save_strategy="epoch", save_total_limit=1, load_best_model_at_end=True,
        metric_for_best_model="auc", greater_is_better=True, seed=args.seed,
        **{ev_key: "epoch"})
    trainer = Trainer(model=model, args=targs, train_dataset=ds_tr, eval_dataset=ds_va,
                      compute_metrics=compute)
    trainer.train()

    v = trainer.predict(ds_va)
    t = trainer.predict(ds_te)
    v_s, t_s = softmax_pos(v.predictions).tolist(), softmax_pos(t.predictions).tolist()
    tm = metrics(te.label.values, t_s)
    print(f"\nTEST RESULTS (claims): {tm}")

    out = ckpt / "text_claims"
    trainer.save_model(str(out))
    tok.save_pretrained(str(out))
    json.dump({"val": {"scores": v_s, "labels": va.label.tolist()},
               "test": {"scores": t_s, "labels": te.label.tolist()}},
              open(ckpt / "scores_text_claims.json", "w"))
    mpath = ckpt / "metrics.json"
    allm = json.load(open(mpath)) if mpath.exists() else {}
    allm["text_claims"] = tm
    json.dump(allm, open(mpath, "w"), indent=2)
    print(f"Saved {out}, scores_text_claims.json, metrics.json['text_claims']")


if __name__ == "__main__":
    main()
