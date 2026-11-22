"""
training_text_hc3_politifact.py - train ONE text model on HC3 + PolitiFact (LIAR claims).

Label 1 means: ChatGPT-written answer (HC3)  OR  false/misleading claim (PolitiFact).
Label 0 means: human-written answer (HC3)    OR  true claim (PolitiFact).
So the model's "fake" score now means "AI-written or false claim". Say this in your report.
Per-source metrics are printed so you can see how it does on each task separately.

Data
  HC3      : downloaded automatically (Hello-SimpleAI/HC3, all.jsonl), or --hc3_jsonl path.
  PolitiFact: --liar_csv data/politifact.csv (columns statement,label; made by make_politifact_csv.py).

Safeguards built in
  * HC3 answers are split BY QUESTION (all answers to one question stay in one split).
  * Texts that appear under both labels in HC3 are removed, then exact duplicates.
  * Each source/class is capped (--hc3_per_class, --liar_per_class) so one source cannot dominate.
  * Prints a text-length-only baseline: if length alone scores high, the model may be learning
    "which dataset is this" instead of "AI or false".

Outputs (--ckpt, default checkpoints). With the default --name text it REPLACES the old text model;
the previous folder/files are renamed to *_prev first.
    checkpoints/<name>/            model + tokenizer (what backend/main.py loads when name = text)
    checkpoints/scores_<name>.json {"val": {...}, "test": {...}}  (used by training_fusion.py)
    checkpoints/metrics.json["<name>"]

Run:
    python training/training_text_hc3_politifact.py --liar_csv data/politifact.csv --epochs 3
After this, re-run training_fusion.py (text scores changed).
NOTE: set --max_len to the same value model.py uses when it tokenises text at inference.
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

FALSE_SET = {"pants-fire", "pants on fire", "pants_fire", "false",
             "barely-true", "barely true", "mostly-false", "mostly false"}
TRUE_SET = {"true", "mostly-true", "mostly true"}


def clean(t):
    return re.sub(r"\s+", " ", t).strip() if isinstance(t, str) else ""


def to_binary(x):
    s = str(x).strip().lower().replace("_", "-")
    if s in FALSE_SET or s.replace("-", " ") in FALSE_SET:
        return 1
    if s in TRUE_SET or s.replace("-", " ") in TRUE_SET:
        return 0
    return None            # half-true / unknown -> dropped


def load_hc3(args):
    path = args.hc3_jsonl
    if not path:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download("Hello-SimpleAI/HC3", "all.jsonl", repo_type="dataset")
    rows = []
    with open(path, encoding="utf-8") as f:
        for gid, line in enumerate(f):
            d = json.loads(line)
            for lab, key in ((0, "human_answers"), (1, "chatgpt_answers")):
                for a in d.get(key) or []:
                    t = clean(a)
                    if len(t) >= 20:
                        rows.append((t, lab, "hc3", f"hc3_{gid}"))
    df = pd.DataFrame(rows, columns=["text", "label", "source", "group"])
    both = set(df[df.label == 0].text) & set(df[df.label == 1].text)
    df = df[~df.text.isin(both)].drop_duplicates("text")
    print(f"HC3: {len(df)} unique answers (human={int((df.label == 0).sum())}, "
          f"chatgpt={int((df.label == 1).sum())})")
    return df


def load_liar(args):
    df = pd.read_csv(args.liar_csv)
    for c in (args.liar_text_col, args.liar_label_col):
        if c not in df.columns:
            raise SystemExit(f"Column '{c}' not in {args.liar_csv}. Columns: {list(df.columns)}")
    df = df[[args.liar_text_col, args.liar_label_col]].rename(
        columns={args.liar_text_col: "text", args.liar_label_col: "raw"})
    df["label"] = df.raw.map(to_binary)
    df = df.dropna(subset=["text", "label"])
    df["text"] = df.text.map(clean)
    df = df[df.text.str.len() >= 10].drop_duplicates("text").copy()
    df["label"] = df.label.astype(int)
    df["source"] = "liar"
    df["group"] = [f"liar_{i}" for i in range(len(df))]
    print(f"PolitiFact/LIAR: {len(df)} claims (false={int(df.label.sum())}, "
          f"true={int((df.label == 0).sum())})")
    return df[["text", "label", "source", "group"]]


def cap(df, per_class, seed):
    if not per_class:
        return df
    parts = []
    for lab in (0, 1):
        d = df[df.label == lab]
        parts.append(d.sample(min(len(d), per_class), random_state=seed))
    return pd.concat(parts)


def group_split(df, seed):
    """80/10/10 split by group (HC3 question / single LIAR claim): no answer-level leakage."""
    rng = np.random.default_rng(seed)
    g = np.array(sorted(df.group.unique()), dtype=object)   # plain numpy array: safe to shuffle
    rng.shuffle(g)
    n = len(g)
    tr, va = set(g[:int(0.8 * n)]), set(g[int(0.8 * n):int(0.9 * n)])
    split = np.where(df.group.isin(tr), "train", np.where(df.group.isin(va), "val", "test"))
    return df[split == "train"], df[split == "val"], df[split == "test"]


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


def softmax_pos(logits):
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return (e / e.sum(axis=1, keepdims=True))[:, 1]


def archive(path):
    """Rename an existing output to *_prev so it is not overwritten."""
    path = Path(path)
    if path.exists():
        prev = path.with_name(path.stem + "_prev" + path.suffix) if path.is_file() \
            else path.with_name(path.name + "_prev")
        if prev.exists():
            shutil.rmtree(prev) if prev.is_dir() else prev.unlink()
        path.rename(prev)
        print(f"Kept previous output as {prev}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints")
    ap.add_argument("--name", default="text", help="output name; 'text' replaces the current text model")
    ap.add_argument("--liar_csv", default="data/politifact.csv")
    ap.add_argument("--liar_text_col", default="statement")
    ap.add_argument("--liar_label_col", default="label")
    ap.add_argument("--hc3_jsonl", default=None)
    ap.add_argument("--hc3_per_class", type=int, default=8000, help="0 = all")
    ap.add_argument("--liar_per_class", type=int, default=4500, help="0 = all")
    ap.add_argument("--model", default="distilroberta-base")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import inspect
    import torch
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer, TrainingArguments)

    torch.manual_seed(args.seed)
    ckpt = Path(args.ckpt)
    ckpt.mkdir(parents=True, exist_ok=True)

    hc3 = cap(load_hc3(args), args.hc3_per_class, args.seed)
    liar = cap(load_liar(args), args.liar_per_class, args.seed)
    df = pd.concat([hc3, liar], ignore_index=True)
    tr, va, te = group_split(df, args.seed)
    for nm, d in (("train", tr), ("val", va), ("test", te)):
        print(f"{nm}: {len(d)}  " + "  ".join(
            f"{s}: {int((d.source == s).sum())} (fake {int(d[d.source == s].label.sum())})"
            for s in d.source.unique()))

    tok = AutoTokenizer.from_pretrained(args.model)

    class Ds(torch.utils.data.Dataset):
        def __init__(self, d):
            self.enc = tok(d.text.tolist(), truncation=True, max_length=args.max_len)
            self.labels = d.label.tolist()

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            return {"input_ids": self.enc["input_ids"][i],
                    "attention_mask": self.enc["attention_mask"][i],
                    "labels": int(self.labels[i])}

    ds_tr, ds_va, ds_te = Ds(tr), Ds(va), Ds(te)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

    def compute(ev):
        return metrics(ev.label_ids, softmax_pos(ev.predictions))

    ev_key = ("eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters
              else "evaluation_strategy")
    targs = TrainingArguments(
        output_dir=str(ckpt / f"_tmp_{args.name}"), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch, per_device_eval_batch_size=64,
        learning_rate=args.lr, weight_decay=0.01, logging_steps=50, report_to="none",
        save_strategy="epoch", save_total_limit=1, load_best_model_at_end=True,
        metric_for_best_model="auc", greater_is_better=True, seed=args.seed,
        fp16=torch.cuda.is_available(), **{ev_key: "epoch"})
    trainer = Trainer(model=model, args=targs, train_dataset=ds_tr, eval_dataset=ds_va,
                      data_collator=DataCollatorWithPadding(tok), compute_metrics=compute)
    trainer.train()

    v_s = softmax_pos(trainer.predict(ds_va).predictions)
    t_s = softmax_pos(trainer.predict(ds_te).predictions)
    tm = metrics(te.label.values, t_s)
    per_source = {s: metrics(te.label.values[(te.source == s).values], t_s[(te.source == s).values])
                  for s in te.source.unique()}
    tm["per_source"] = per_source
    print("\nTEST (all):", {k: round(v, 4) if isinstance(v, float) else v
                            for k, v in tm.items() if k != "per_source"})
    for s, m in per_source.items():
        print(f"TEST ({s}): acc {m['acc']:.4f}  prec {m['precision']:.4f}  rec {m['recall']:.4f}  "
              f"F1 {m['f1']:.4f}  AUC {m['auc']:.4f}")

    # shortcut check: how well does plain word count separate the classes?
    wc = te.text.str.split().str.len().values.astype(float)
    print(f"Word-count-only AUC: all {roc_auc_score(te.label.values, wc):.3f}  " + "  ".join(
        f"{s} {roc_auc_score(te.label.values[(te.source == s).values], wc[(te.source == s).values]):.3f}"
        for s in te.source.unique()))

    out = ckpt / args.name
    scores_path = ckpt / f"scores_{args.name}.json"
    archive(out)
    archive(scores_path)
    trainer.save_model(str(out))
    tok.save_pretrained(str(out))
    json.dump({"val": {"scores": v_s.tolist(), "labels": va.label.tolist(), "sources": va.source.tolist()},
               "test": {"scores": t_s.tolist(), "labels": te.label.tolist(), "sources": te.source.tolist()}},
              open(scores_path, "w"))
    mpath = ckpt / "metrics.json"
    allm = json.load(open(mpath)) if mpath.exists() else {}
    allm[args.name] = tm
    json.dump(allm, open(mpath, "w"), indent=2)
    print(f"\nSaved {out}, {scores_path.name}, metrics.json['{args.name}']")


if __name__ == "__main__":
    main()
