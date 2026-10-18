"""
explainability.py - Layer 5: real, model-based explanations for every detection.

What each modality gets
  text   : LIME - a local weighted linear surrogate fitted on word-removal perturbations
           (own implementation of the LIME algorithm, so no extra package is needed).
           Gives a signed weight per word: + pushes toward AI-generated, - toward human.
  image  : Grad-CAM on the spatial branch and on the frequency (DFT) branch of the CNN.
  audio  : Grad-CAM on the log-mel spectrogram -> which seconds / pitch range mattered.
  video  : per-frame fake probability + Grad-CAM on the most informative frames.
  fusion : EXACT Shapley values (the SHAP definition) over the modalities, i.e. how many
           probability points each modality added to or removed from the final verdict.

All functions return plain JSON-serialisable dicts; images are base64-encoded PNG strings.
The numpy/OpenCV parts are kept separate from the PyTorch plumbing (gradcam, explain_*_cnn)
so they can be unit-tested without a model.

Honest scope: these methods show what influenced THIS model's output. They are evidence
about the model's behaviour, not proof about the content, and the CNN heatmaps are coarse
because the detectors run on small inputs (64x64 image/audio, 160x160 face crops).
"""

import base64
import itertools
import math

import cv2
import numpy as np

TARGET_NAMES = {1: "AI-generated", 0: "authentic"}
NOTE = ("Explanations describe what influenced this model's output - evidence about the "
        "model's behaviour, not proof about the content.")


# ---------------------------------------------------------------------------
# Image helpers (numpy + OpenCV only)
# ---------------------------------------------------------------------------

def png_b64(rgb):
    """RGB uint8 array -> base64 PNG string."""
    bgr = cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def fit_display(rgb, max_side=384, min_side=224):
    """Scale so the longest side is within [min_side, max_side] (small inputs are enlarged)."""
    h, w = rgb.shape[:2]
    longest = max(h, w)
    if longest > max_side:
        s, interp = max_side / longest, cv2.INTER_AREA
    elif longest < min_side:
        s, interp = min_side / longest, cv2.INTER_CUBIC
    else:
        return rgb
    return cv2.resize(rgb, (max(1, int(round(w * s))), max(1, int(round(h * s)))), interpolation=interp)


def overlay_heatmap(rgb, cam, strength=0.65):
    """Blend a [0,1] heatmap (any size) over an RGB image; cold areas stay untouched."""
    h, w = rgb.shape[:2]
    big = np.clip(cv2.resize(np.asarray(cam, np.float32), (w, h), interpolation=cv2.INTER_CUBIC), 0, 1)
    heat = cv2.cvtColor(cv2.applyColorMap((big * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    a = (strength * big)[..., None]
    return np.clip(rgb.astype(np.float32) * (1 - a) + heat.astype(np.float32) * a, 0, 255).astype(np.uint8)


def region_name(cam):
    """Where the heatmap peaks, as words (3x3 grid)."""
    h, w = cam.shape
    y, x = np.unravel_index(int(np.argmax(cam)), cam.shape)
    v = ("upper", "middle", "lower")[min(2, int(3 * y / h))]
    hz = ("left", "centre", "right")[min(2, int(3 * x / w))]
    if v == "middle":
        return "centre" if hz == "centre" else hz
    return f"{v} {hz}" if hz != "centre" else f"{v} centre"


def radial_band(cam):
    """Peak of a centred DFT-magnitude heatmap: low / mid / high spatial frequency."""
    h, w = cam.shape
    y, x = np.unravel_index(int(np.argmax(cam)), cam.shape)
    r = math.hypot((y + 0.5) / h - 0.5, (x + 0.5) / w - 0.5) / 0.5
    return "low" if r < 0.33 else "mid" if r < 0.66 else "high"


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------

def cam_from_arrays(acts, grads):
    """Grad-CAM maths on numpy arrays of shape (B,C,H,W): channel weights = spatial mean of
    the gradients, map = ReLU(sum_c w_c * A_c), then scaled to [0,1] per sample."""
    acts, grads = np.asarray(acts, np.float64), np.asarray(grads, np.float64)
    w = grads.mean(axis=(2, 3), keepdims=True)
    cam = np.maximum((w * acts).sum(axis=1), 0.0)
    mx = cam.reshape(len(cam), -1).max(axis=1)[:, None, None]
    return np.where(mx > 1e-12, cam / np.maximum(mx, 1e-12), 0.0).astype(np.float32)


def gradcam(layer, forward_fn, target_class=1):
    """Grad-CAM for `layer` of a 2-class model whose forward_fn() returns softmax probs (B,2).
    The target score is the log-odds margin log p_target - log p_other, which keeps a usable
    gradient even when the softmax is saturated. Returns (cams (B,h,w) in [0,1], probs (B,2))."""
    import torch
    store = {}
    handle = layer.register_forward_hook(lambda m, i, o: store.__setitem__("out", o))
    try:
        with torch.enable_grad():
            probs = forward_fn()
            out = store.get("out")
            if out is None or not out.requires_grad:
                raise RuntimeError("Grad-CAM layer produced no gradient (are the weights frozen?)")
            p, q = probs[:, target_class], probs[:, 1 - target_class]
            score = (torch.log(p + 1e-12) - torch.log(q + 1e-12)).sum()
            grads = torch.autograd.grad(score, out)[0]
    finally:
        handle.remove()
    cams = cam_from_arrays(out.detach().cpu().numpy(), grads.detach().cpu().numpy())
    return cams, probs.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Text: LIME
# ---------------------------------------------------------------------------

def lime_text(predict_fn, text, score=None, num_samples=150, max_words=80,
              kernel_width=25.0, ridge=1.0, seed=0):
    """LIME for text. predict_fn(list[str]) -> array of P(AI-generated).
    Randomly removes words, queries the detector, and fits a distance-weighted ridge
    regression on the 'word kept / removed' indicators. Its coefficients are the word weights."""
    all_words = text.split()
    words = all_words[:max_words]
    n = len(words)
    if n == 0:
        raise ValueError("empty text")
    rng = np.random.default_rng(seed)

    Z = np.ones((num_samples, n))
    for i in range(1, num_samples):
        k = 1 if n == 1 else int(rng.integers(1, n))          # remove 1 .. n-1 words
        Z[i, rng.choice(n, k, replace=False)] = 0.0
    texts = [" ".join(w for w, keep in zip(words, row) if keep) for row in Z]
    y = np.asarray(predict_fn(texts), dtype=np.float64)

    # LIME's kernel: cosine distance (x100) to the original, exponential kernel width 25
    cos = Z.sum(axis=1) / (np.sqrt(Z.sum(axis=1)) * math.sqrt(n) + 1e-12)
    d = (1.0 - cos) * 100.0
    sw = np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

    A = np.hstack([Z, np.ones((num_samples, 1))])             # last column = intercept
    reg = np.diag([ridge] * n + [0.0])
    M = A.T @ (A * sw[:, None]) + reg + 1e-9 * np.eye(n + 1)
    beta = np.linalg.solve(M, A.T @ (sw * y))
    coef = beta[:n]

    resid = y - A @ beta
    ybar = float(np.average(y, weights=sw))
    ss_tot = float(np.sum(sw * (y - ybar) ** 2))
    r2 = float(1.0 - np.sum(sw * resid ** 2) / ss_tot) if ss_tot > 1e-12 else None

    ai_score = float(y[0] if score is None else score)
    order = np.argsort(-coef)
    toward_ai = [words[i] for i in order if coef[i] > 0][:5]
    toward_human = [words[i] for i in order[::-1] if coef[i] < 0][:5]
    parts = [f"AI-probability {ai_score:.1%}."]
    if toward_ai:
        parts.append("Words pushing toward AI-generated: " + ", ".join(toward_ai) + ".")
    if toward_human:
        parts.append("Words pushing toward human: " + ", ".join(toward_human) + ".")
    if r2 is not None:
        parts.append(f"The local surrogate reproduces the detector with R² = {r2:.2f}"
                     + (" (low: treat the weights as rough)." if r2 < 0.5 else "."))
    return {
        "method": f"LIME (weighted linear surrogate, {num_samples} word-removal perturbations)",
        "ai_score": round(ai_score, 4),
        "words": [{"word": w, "weight": round(float(c), 5)} for w, c in zip(words, coef)],
        "highlighted_tokens": toward_ai,
        "supports_human": toward_human,
        "fidelity_r2": None if r2 is None else round(r2, 3),
        "truncated": len(all_words) > max_words,
        "explanation": " ".join(parts),
        "note": NOTE + (f" Only the first {max_words} words were analysed." if len(all_words) > max_words else ""),
    }


# ---------------------------------------------------------------------------
# Fusion: exact Shapley values
# ---------------------------------------------------------------------------

def exact_shapley(value_fn, players):
    """Exact Shapley values. value_fn(set_of_players) -> float. Returns (phi, v(empty), v(all)).
    sum(phi) == v(all) - v(empty) (the efficiency property)."""
    n = len(players)
    cache = {}

    def v(S):
        key = frozenset(S)
        if key not in cache:
            cache[key] = float(value_fn(set(key)))
        return cache[key]

    phi = {p: 0.0 for p in players}
    for p in players:
        others = [q for q in players if q != p]
        for r in range(len(others) + 1):
            weight = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
            for S in itertools.combinations(others, r):
                phi[p] += weight * (v(set(S) | {p}) - v(set(S)))
    return phi, v(set()), v(set(players))


def explain_fusion(fusion_fn, scores, active, threshold):
    """fusion_fn(scores_dict, active_list) -> P(AI). Absent modalities are masked out exactly
    the way the fusion layer was trained to handle them (neutral score + presence flag 0)."""
    order = ("text", "image", "audio", "video")
    players = [m for m in order if m in active and m in scores]
    if not players:
        raise ValueError("no modality scores to attribute")

    def value(S):
        return fusion_fn({m: scores[m] for m in S}, [m for m in players if m in S])

    phi, base, full = exact_shapley(value, players)
    verdict = "AI-generated" if full >= threshold else "human/authentic"
    ranked = sorted(players, key=lambda m: -abs(phi[m]))
    moves = ", ".join(f"{m} {phi[m] * 100:+.1f} pts" for m in ranked)
    return {
        "method": "Exact Shapley values (SHAP definition) over the modalities given to the fusion layer",
        "base_value": round(base, 4),
        "final_value": round(full, 4),
        "threshold": threshold,
        "contributions": {m: round(float(phi[m]), 4) for m in players},
        "modality_scores": {m: round(float(scores[m]), 4) for m in players},
        "explanation": (f"With no evidence the fusion layer outputs {base:.1%}. Each modality moves the "
                        f"AI-probability by: {moves}. Result {full:.1%} -> {verdict} "
                        f"(threshold {threshold:.0%})."),
        "note": ("Contributions are in probability points and add up exactly to final minus baseline. "
                 "Shapley values describe the fusion layer, which was trained on synthetically paired scores."),
    }


# ---------------------------------------------------------------------------
# Reports built from heatmaps (numpy/OpenCV only -> unit-testable)
# ---------------------------------------------------------------------------

def image_report(img_bgr, freq, cam_s, cam_f, p, target):
    img = np.asarray(img_bgr)
    rgb = (cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if img.ndim == 2
           else cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB))
    spatial = overlay_heatmap(fit_display(rgb, 384, 256), cam_s)
    f_gray = (np.clip(freq, 0, 1) * 255).astype(np.uint8)
    spectrum = overlay_heatmap(fit_display(cv2.cvtColor(f_gray, cv2.COLOR_GRAY2RGB), 256, 256), cam_f)
    name = TARGET_NAMES[target]
    coverage = float((cam_s >= 0.5).mean())
    return {
        "method": "Grad-CAM on the spatial branch and the frequency (DFT) branch of the image CNN",
        "ai_score": round(float(p), 4),
        "target": name,
        "images": {"spatial_heatmap": png_b64(spatial), "frequency_heatmap": png_b64(spectrum)},
        "explanation": (f"Heatmaps show the evidence behind the '{name}' verdict (red = strongest). "
                        f"Spatially it concentrated on the {region_name(cam_s)} of the image "
                        f"(the strongest evidence covers about {coverage:.0%} of the picture). "
                        f"In the frequency spectrum the peak is in the {radial_band(cam_f)}-frequency region "
                        f"(centre of the spectrum = low frequencies)."),
        "note": NOTE + " The maps are coarse: the detector sees a 64x64 version of the image.",
    }


def audio_report(spec, cam, duration_s, p, target):
    spec = np.clip(np.asarray(spec, np.float32), 0, 1)
    base = cv2.cvtColor((np.flipud(spec) * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)   # low pitch at bottom
    img = overlay_heatmap(fit_display(base, 384, 256), np.flipud(cam))

    h, w = cam.shape
    curve = np.interp(np.linspace(0, w - 1, 64), np.arange(w), cam.mean(axis=0))
    top = float(curve.max())
    curve = curve / top if top > 1e-12 else curve * 0.0
    segs, i = [], 0
    while i < 64:
        if curve[i] >= 0.6:
            j = i
            while j + 1 < 64 and curve[j + 1] >= 0.6:
                j += 1
            segs.append({"start_s": round(i / 64 * duration_s, 2), "end_s": round((j + 1) / 64 * duration_s, 2),
                         "strength": round(float(curve[i:j + 1].mean()), 3)})
            i = j + 1
        else:
            i += 1
    segs = sorted(segs, key=lambda s: -s["strength"])[:3]

    rows = cam.mean(axis=1)
    k = (int(np.argmax(rows)) + 0.5) / h
    band = "low" if k < 1 / 3 else "mid" if k < 2 / 3 else "high"
    name = TARGET_NAMES[target]
    if segs:
        s = segs[0]
        where = (f"The strongest evidence is at {s['start_s']:.1f}-{s['end_s']:.1f} s of the "
                 f"{duration_s:.1f} s analysed, mainly in the {band} mel-frequency range.")
    else:
        where = "No localised segment stood out."
    return {
        "method": "Grad-CAM on the log-mel spectrogram CNN",
        "ai_score": round(float(p), 4),
        "target": name,
        "images": {"spectrogram_heatmap": png_b64(img)},
        "time_importance": [round(float(v), 3) for v in curve[::2]],
        "duration_s": round(float(duration_s), 2),
        "top_segments": segs,
        "explanation": f"Evidence behind the '{name}' verdict. {where}",
        "note": NOTE + " Only the first 5 seconds of the clip are analysed, as in training.",
    }


def video_report(clip, found, p_fake, cams, top_k=3):
    """clip: (T,S,S,3) RGB uint8 face crops; p_fake: (n,) per-frame P(fake); cams: (n,h,w)."""
    p_fake = np.asarray(p_fake, np.float64)
    n = len(p_fake)
    mean_p = float(p_fake.mean())
    target = 1 if mean_p >= 0.5 else 0
    p_t = p_fake if target == 1 else 1.0 - p_fake
    chosen = sorted(np.argsort(-p_t)[:top_k].tolist())
    frames = []
    for i in chosen:
        frames.append({
            "index": int(i),
            "position_pct": int(round(100 * i / max(n - 1, 1))),
            "p_fake": round(float(p_fake[i]), 4),
            "image": png_b64(overlay_heatmap(fit_display(clip[i], 320, 256), cams[i])),
        })
    name = TARGET_NAMES[target]
    over = int((p_fake >= 0.5).sum())
    text = (f"Mean fake-probability {mean_p:.1%} over {n} sampled frames (per-frame range "
            f"{p_fake.min():.0%}-{p_fake.max():.0%}); {over} of {n} frames scored above 50%. "
            f"The frames below are the ones giving the strongest evidence for '{name}'; red marks the "
            f"face regions the network relied on.")
    if not found:
        text += " WARNING: no face was detected, so a centre crop was used and the result is less reliable."
    return {
        "method": "Per-frame EfficientNet-B0 scores + Grad-CAM on the most informative frames",
        "ai_score": round(mean_p, 4),
        "target": name,
        "face_found": bool(found),
        "frame_scores": [round(float(v), 4) for v in p_fake],
        "top_frames": frames,
        "explanation": text,
        "note": NOTE,
    }


# ---------------------------------------------------------------------------
# PyTorch wrappers (image / audio)
# ---------------------------------------------------------------------------

def explain_image_cnn(det, feat, img_bgr):
    """det = FreqAwareCNN, feat = ImageFeatureExtractor (the same objects used for scoring)."""
    import torch
    spatial, freq = feat.extract(img_bgr)
    sp = torch.tensor(spatial).permute(2, 0, 1).unsqueeze(0)
    fr = torch.tensor(freq).unsqueeze(0).unsqueeze(0)
    det.eval()
    with torch.no_grad():
        p = float(det(sp, fr)[0, 1])
    target = 1 if p >= 0.5 else 0
    cam_s, _ = gradcam(det.spatial[-2], lambda: det(sp, fr), target)
    cam_f, _ = gradcam(det.freq[-2], lambda: det(sp, fr), target)
    return image_report(img_bgr, freq, cam_s[0], cam_f[0], p, target)


def explain_audio_cnn(det, feat, audio, sr):
    """det = AudioSpectrogramCNN, feat = AudioFeatureExtractor; audio already cut to <= 5 s."""
    import torch
    spec = feat.extract(audio, sr)
    x = torch.tensor(spec).unsqueeze(0).unsqueeze(0)
    det.eval()
    with torch.no_grad():
        p = float(det(x)[0, 1])
    target = 1 if p >= 0.5 else 0
    cams, _ = gradcam(det.net[-2], lambda: det(x), target)
    return audio_report(spec, cams[0], len(audio) / float(sr), p, target)