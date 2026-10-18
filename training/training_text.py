"""
training_text.py - fine-tune DistilRoBERTa on HC3 (human vs ChatGPT), CPU-friendly defaults.

Changes vs. the earlier script:
  * Split BY QUESTION (GroupShuffleSplit), so the same question never appears
    in train and test -> no topic leakage, honest metrics.
  * Dynamic padding (much faster on CPU) with max_length=256.
  * Reports accuracy / precision / recall / F1 / AUC.
  * Saves the model to <ckpt>/text (what train.py's load_trained expects).
  * Exports <ckpt>/scores_text.json (val/test AI-probabilities) so the
    fusion layer in train.py can use the text modality.
  * Writes test metrics to <ckpt>/metrics.json under "text".

Usage (run from the project root, e.g. C:\\Users\\mayan\\Multimodel):
    python training\\training_text.py                          # CPU defaults: 3000 questions, 2 epochs, len 128
    python training\\training_text.py --max_questions 0 --epochs 3 --max_len 256   # full run (use a GPU / Colab)
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, load_dataset
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "distilroberta-base"
HC3_URL = "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/all.jsonl"


def metrics_from_probs(labels, probs):
    labels, probs = np.asarray(labels), np.asarray(probs)
    preds = (probs >= 0.5).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0)
    try:
        auc = float(roc_auc_score(labels, probs))
    except ValueError:
        auc = float("nan")
    return {"acc": float(accuracy_score(labels, preds)), "precision": float(p),
            "recall": float(r), "f1": float(f1), "auc": auc}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = softmax(logits, axis=-1)[:, 1]
    m = metrics_from_probs(labels, probs)
    return {"accuracy": m["acc"], "precision": m["precision"],
            "recall": m["recall"], "f1": m["f1"], "auc": m["auc"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--max_questions", type=int, default=3000, help="0 = all questions")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.set_num_threads(os.cpu_count() or 4)
    ckpt = Path(args.ckpt)
    ckpt.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load HC3 ----------------------------------------------------
    print("Loading HC3...")
    ds = load_dataset("json", data_files=HC3_URL)["train"]
    qids = list(range(len(ds)))
    if args.max_questions and args.max_questions < len(qids):
        rng = np.random.default_rng(args.seed)
        qids = sorted(rng.choice(qids, args.max_questions, replace=False).tolist())

    # ---- 2. Build records, keeping the question id ---------------------
    records = []
    for qid in qids:
        row = ds[qid]
        for a in row["human_answers"]:
            if a and a.strip():
                records.append({"text": a.strip(), "label": 0, "qid": qid})
        for a in row["chatgpt_answers"]:
            if a and a.strip():
                records.append({"text": a.strip(), "label": 1, "qid": qid})
    print(f"Questions: {len(qids)}  Answers: {len(records)}")

    # ---- 3. Split BY QUESTION: 85 / 7.5 / 7.5 --------------------------
    idx = np.arange(len(records))
    groups = np.array([r["qid"] for r in records])
    g1 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=args.seed)
    tr_i, tmp_i = next(g1.split(idx, groups=groups))
    g2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=args.seed)
    va_rel, te_rel = next(g2.split(tmp_i, groups=groups[tmp_i]))
    va_i, te_i = tmp_i[va_rel], tmp_i[te_rel]

    tr_q, va_q, te_q = (set(groups[i]) for i in (tr_i, va_i, te_i))
    assert not (tr_q & va_q or tr_q & te_q or va_q & te_q), "question leakage!"

    def subset(ids):
        return Dataset.from_list(
            [{"text": records[i]["text"], "label": records[i]["label"]} for i in ids])

    train_ds, val_ds, test_ds = subset(tr_i), subset(va_i), subset(te_i)
    for name, d in [("Train", train_ds), ("Validation", val_ds), ("Test", test_ds)]:
        print(f"{name}: {len(d)} (AI fraction {np.mean(d['label']):.2f})")

    # ---- 4. Tokenise (dynamic padding) ---------------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_len)

    def prep(d):
        d = d.map(tok, batched=True)
        return d.remove_columns(["text"]).rename_column("label", "labels")

    train_t, val_t, test_t = prep(train_ds), prep(val_ds), prep(test_ds)

    # ---- 5. Model + Trainer --------------------------------------------
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    training_args = TrainingArguments(
        output_dir=str(ckpt / "text_trainer_runs"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=100,
        fp16=False,
        seed=args.seed,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_t,
        eval_dataset=val_t,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    print("\n=== TRAINING ===")
    trainer.train()

    # ---- 6. Evaluate + export scores -----------------------------------
    out = {}
    for name, d in [("val", val_t), ("test", test_t)]:
        pred = trainer.predict(d)
        probs = softmax(pred.predictions, axis=-1)[:, 1]
        out[name] = {"scores": probs.tolist(), "labels": pred.label_ids.tolist()}
        m = metrics_from_probs(pred.label_ids, probs)
        print(f"\n{name.upper()} RESULTS: {m}")
        if name == "test":
            test_metrics = m

    with open(ckpt / "scores_text.json", "w") as f:
        json.dump(out, f)

    mpath = ckpt / "metrics.json"
    allm = json.load(open(mpath)) if mpath.exists() else {}
    allm["text"] = test_metrics
    json.dump(allm, open(mpath, "w"), indent=2)

    # ---- 7. Save model --------------------------------------------------
    trainer.save_model(str(ckpt / "text"))
    tokenizer.save_pretrained(str(ckpt / "text"))
    print(f"\nModel saved to {ckpt / 'text'}")
    print(f"Scores saved to {ckpt / 'scores_text.json'} (for fusion)")


if __name__ == "__main__":
    main()