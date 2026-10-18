"""
video_effnet.py - inference helper for the EfficientNet-B0 video detector.

Everything here (frame sampling, face crop, size, normalisation, mean of per-frame
softmax) MUST stay identical to training_video_effnet.py, otherwise the scores the
fusion layer was trained on will not match what the detector produces at inference.

Put this file next to your model file, then make three small edits there
(see the instructions in the reply):  import, load_weights, _detect_video.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

SIZE = 160
T_FRAMES = 16
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def make_model():
    """Same architecture as training; weights=None because the trained state_dict is loaded."""
    m = torchvision.models.efficientnet_b0(weights=None)
    m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(1280, 2))
    return m


def read_frames(path, n=T_FRAMES, max_sequential=4000):
    """n evenly spaced BGR frames from the whole video.

    Videos up to `max_sequential` frames are decoded sequentially. Seeking inside H.264
    streams (typical for social-media reels) can hit missing reference frames and return
    garbled frames ("mmco: unref short failure"), which a detector may read as fake.
    Longer videos fall back to seeking to keep the decoding time reasonable."""
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = np.linspace(0, max(total - 1, 0), n).astype(int)
    frames = []
    if 0 < total <= max_sequential:
        wanted, last = set(idx.tolist()), int(idx.max())
        for i in range(last + 1):
            if not cap.grab():
                break
            if i in wanted:
                ok, f = cap.retrieve()
                if ok:
                    frames.append(f)
    else:
        for i in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, f = cap.read()
            if ok:
                frames.append(f)
    cap.release()
    return frames


def crop_faces(frames, size=SIZE, margin=1.4):
    """Median face box over the clip, crop + resize to size x size RGB (centre crop if no face)."""
    boxes = []
    for f in frames:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        d = CASCADE.detectMultiScale(g, 1.1, 5, minSize=(60, 60))
        if len(d):
            boxes.append(max(d, key=lambda b: b[2] * b[3]))
    h, w = frames[0].shape[:2]
    found = len(boxes) > 0
    if found:
        x, y, bw, bh = np.median(np.array(boxes, dtype=np.float32), axis=0)
        cx, cy, s = x + bw / 2, y + bh / 2, max(bw, bh) * margin
    else:
        cx, cy, s = w / 2, h / 2, min(h, w) * 0.6
    x0, y0 = int(max(cx - s / 2, 0)), int(max(cy - s / 2, 0))
    x1, y1 = int(min(cx + s / 2, w)), int(min(cy + s / 2, h))
    out = []
    for f in frames:
        c = cv2.resize(f[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_AREA)
        out.append(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
    while len(out) < T_FRAMES:
        out.append(out[-1])
    return np.stack(out[:T_FRAMES]), found


def load_video_effnet(path, device="cpu"):
    model = make_model()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.to(device).eval()


@torch.no_grad()
def score_video(model, path):
    """P(fake) for one video file = mean of the per-frame softmax over 16 face crops."""
    frames = read_frames(path)
    if not frames:
        raise ValueError(f"Cannot decode any frames from: {path}")
    clip, _ = crop_faces(frames, SIZE)
    x = torch.from_numpy(clip).permute(0, 3, 1, 2).float() / 255.0
    x = (x - MEAN) / STD
    dev = next(model.parameters()).device
    probs = F.softmax(model(x.to(dev)).float(), dim=-1)[:, 1]
    return float(probs.mean())


def explain_video(model, path, top_k=3):
    """Per-frame fake probabilities + Grad-CAM on the most informative frames.
    Uses exactly the same frames, face crop and normalisation as score_video()."""
    import explainability as ex                      # lives next to this file

    frames = read_frames(path)
    if not frames:
        raise ValueError(f"Cannot decode any frames from: {path}")
    clip, found = crop_faces(frames, SIZE)
    n = len(frames)                                  # real frames; the rest of the clip is padding
    x = torch.from_numpy(clip).permute(0, 3, 1, 2).float() / 255.0
    x = (x - MEAN) / STD
    dev = next(model.parameters()).device
    x = x.to(dev)[:n]

    model.eval()
    with torch.no_grad():
        p_fake = F.softmax(model(x).float(), dim=-1)[:, 1].cpu().numpy()
    target = 1 if float(p_fake.mean()) >= 0.5 else 0
    cams, _ = ex.gradcam(model.features[-1], lambda: F.softmax(model(x).float(), dim=-1), target)
    return ex.video_report(clip[:n], found, p_fake, cams, top_k=top_k)