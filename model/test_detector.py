"""
test_detector.py - quick sanity test of the trained multimodal detector.

Put this next to multimodal_detector.py and run from that folder:

    python test_detector.py                          # built-in text samples
    python test_detector.py --text "Paste any text here"
    python test_detector.py --text_file essay.txt
    python test_detector.py --image photo.jpg
    python test_detector.py --audio clip.wav
    python test_detector.py --video clip.mp4
    python test_detector.py --image a.jpg --text "caption"   # several at once

Only modalities whose weights exist in --ckpt are meaningful. Until
checkpoints/fusion.pt exists, ignore the "Final Decision" and read the
per-modality scores instead.
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification

from model import InputPayload, MultimodalDetector

# Written for this test; (expected_label, text). 1 = AI-style, 0 = human-style.
SAMPLES = [
    (0, "honestly no idea why my sourdough keeps coming out flat. tried a warmer spot, "
        "more feeding, nothing. my buddy says its the flour but i think my starter's just lazy lol"),
    (0, "we drove up there last summer and the road was basically gravel the whole way. "
        "the view was worth it tho, even if my car hated me by the end"),
    (1, "There are several factors that can cause sourdough bread to turn out flat. First, "
        "ensure your starter is active and has doubled in size before use. Second, consider "
        "the hydration level of your dough, as too much water can make it difficult to hold "
        "its shape. Finally, make sure to proof the dough for an appropriate amount of time."),
    (1, "Regular exercise offers numerous benefits for both physical and mental health. It can "
        "help improve cardiovascular fitness, strengthen muscles, and boost mood by releasing "
        "endorphins. Additionally, maintaining a consistent routine can enhance sleep quality "
        "and overall well-being."),
]


def load_trained(detector, ckpt):
    """Load whatever trained weights exist in ckpt; report what was found."""
    ck = Path(ckpt)
    found = []
    if (ck / "text").exists():
        detector.text_det._hf_model = AutoModelForSequenceClassification.from_pretrained(
            ck / "text").eval()
        found.append("text")
    for attr, fname, name in [("img_det", "image.pt", "image"), ("audio_det", "audio.pt", "audio"),
                              ("video_det", "video.pt", "video"), ("fusion", "fusion.pt", "fusion")]:
        if (ck / fname).exists():
            getattr(detector, attr).load_state_dict(torch.load(ck / fname, map_location="cpu"))
            getattr(detector, attr).eval()
            found.append(name)
    print(f"Loaded trained weights for: {found or 'NOTHING (all random!)'}")
    return detector, found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints")
    ap.add_argument("--text")
    ap.add_argument("--text_file")
    ap.add_argument("--image")
    ap.add_argument("--audio")
    ap.add_argument("--video")
    args = ap.parse_args()

    detector, found = load_trained(MultimodalDetector(), args.ckpt)

    mpath = Path(args.ckpt) / "metrics.json"
    if mpath.exists():
        print("\nSaved test metrics (from training):")
        print(json.dumps(json.load(open(mpath)), indent=2))

    text = args.text
    if args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8", errors="ignore")

    if not any([text, args.image, args.audio, args.video]):
        if "text" not in found:
            print("\nNo trained text model found in", args.ckpt)
            return
        print("\n--- Built-in text sanity check (expected: human-style low, AI-style high) ---")
        ok = 0
        for expected, t in SAMPLES:
            score = detector.predict(InputPayload(text=t)).modality_scores["text"]
            hit = (score >= 0.5) == bool(expected)
            ok += hit
            print(f"[{'OK ' if hit else 'MISS'}] expected={'AI   ' if expected else 'human'} "
                  f"text_score={score:.3f}  | {t[:60]}...")
        print(f"\n{ok}/{len(SAMPLES)} as expected. This is a sanity check on 4 short texts, "
              "NOT an evaluation - use metrics.json for that.")
        return

    result = detector.predict(InputPayload(
        text=text, image_path=args.image, audio_path=args.audio, video_path=args.video))
    print("\n" + result.report)
    print("Per-modality scores:", result.modality_scores)
    if "fusion" not in found:
        print("\nNOTE: fusion.pt not found, so the Final Decision above is from random "
              "weights. Read the per-modality scores instead.")
    print("\nExplanations:")
    for k, v in result.explanations.items():
        print(f"  {k}: {v['explanation']}")


if __name__ == "__main__":
    main()