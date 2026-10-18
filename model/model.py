"""
=============================================================
  Multimodal AI-Content Detection Model  —  6-Layer Pipeline
=============================================================
Dissertation: "Combating the Rise of AI-Synthesized Malicious Content"

LAYER 1 : Input          — accepts text / image / audio / video
LAYER 2 : Feature Extract— tokenization, spectrogram, frame/face-align, freq-domain
LAYER 3 : Detection      — per-modality deep-learning detectors
LAYER 4 : Fusion         — learned meta-classifier combines modality scores
LAYER 5 : Explainability — LIME / SHAP / attention maps
LAYER 6 : Output         — final decision + confidence + report + feedback loop

CPU-friendly: all models are lightweight / quantised variants.
"""

import io
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import cv2
from PIL import Image
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _video_effnet():
    """Import video_effnet.py from the same folder as this file, wherever the app is run from."""
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    try:
        import video_effnet
    except ImportError as e:
        raise ImportError(
            f"Could not import video_effnet.py from {here}. Copy video_effnet.py into that "
            f"folder and run `pip install torchvision`. Original error: {e}") from e
    return video_effnet


def _explainability():
    """Import explainability.py from the same folder as this file."""
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    try:
        import explainability
    except ImportError as e:
        raise ImportError(f"Could not import explainability.py from {here}. Copy it next to "
                          f"model.py. Original error: {e}") from e
    return explainability


# ---------------------------------------------------------------------------
# LAYER 1 — INPUT LAYER
# ---------------------------------------------------------------------------

@dataclass
class InputPayload:
    """Unified input container for all modalities."""
    text: Optional[str] = None
    image_path: Optional[str] = None          # path or PIL.Image
    audio_path: Optional[str] = None          # path or np.ndarray (sr=16000)
    video_path: Optional[str] = None          # path to video file
    image_array: Optional[np.ndarray] = None
    audio_array: Optional[np.ndarray] = None
    audio_sr: int = 16000

    @property
    def active_modalities(self) -> List[str]:
        mods = []
        if self.text:
            mods.append("text")
        if self.image_path or self.image_array is not None:
            mods.append("image")
        if self.audio_path or self.audio_array is not None:
            mods.append("audio")
        if self.video_path:
            mods.append("video")
        return mods


@dataclass
class DetectionResult:
    """Final output container (Layer 6)."""
    final_label: str = "unknown"           # "AI-generated" | "human/authentic"
    confidence: float = 0.0               # 0–1
    modality_scores: Dict[str, float] = field(default_factory=dict)
    explanations: Dict[str, Any] = field(default_factory=dict)
    report: str = ""
    raw_features: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# LAYER 2 — FEATURE EXTRACTION LAYER
# ---------------------------------------------------------------------------

class TextFeatureExtractor:
    """Tokenises text and returns model-ready encodings."""

    def __init__(self, model_name: str = "distilroberta-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def extract(self, text: str, max_length: int = 128) -> Dict:   # = --max_len used in training_text.py
        encoding = self.tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        # Perplexity proxy: token variance (fast, CPU-friendly)
        ids = encoding["input_ids"][0].numpy()
        perplexity_proxy = float(np.std(ids))
        return {"encoding": encoding, "perplexity_proxy": perplexity_proxy}


class AudioFeatureExtractor:
    """Extracts log-mel spectrograms from audio (CPU-safe via librosa)."""

    def __init__(self, sr: int = 16000, n_mels: int = 64, hop_length: int = 512):
        self.sr = sr
        self.n_mels = n_mels
        self.hop_length = hop_length

    def extract(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if sr != self.sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sr)
        mel = librosa.feature.melspectrogram(
            y=audio, sr=self.sr, n_mels=self.n_mels, hop_length=self.hop_length
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        # Normalise to [0, 1] and resize to fixed 64×64
        log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-9)
        log_mel_resized = cv2.resize(log_mel, (64, 64))
        return log_mel_resized.astype(np.float32)


class ImageFeatureExtractor:
    """Extracts spatial + frequency-domain features from images."""

    def extract(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        # Spatial: resize to 64×64
        spatial = cv2.resize(img, (64, 64)).astype(np.float32) / 255.0
        # Frequency domain (DFT magnitude spectrum)
        dft = np.fft.fft2(gray.astype(np.float32))
        dft_shift = np.fft.fftshift(dft)
        magnitude = 20 * np.log(np.abs(dft_shift) + 1e-9)
        freq = cv2.resize(magnitude, (64, 64)).astype(np.float32)
        freq = (freq - freq.min()) / (freq.max() - freq.min() + 1e-9)
        return spatial, freq


class VideoFeatureExtractor:
    """Samples key frames, crops the largest detected face (Haar cascade), returns frame stack."""

    FACE_CROP = True   # training_video.py checks for this flag

    def __init__(self, max_frames: int = 16):
        self.max_frames = max_frames
        self.img_feat = ImageFeatureExtractor()
        self._cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def _crop_face(self, frame: np.ndarray) -> np.ndarray:
        """Crop the largest face (+20% margin); fall back to the full frame if none found."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        if len(faces) == 0:
            return frame
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        pad = int(0.2 * max(w, h))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
        return frame[y0:y1, x0:x1]

    def extract(self, video_path: str) -> List[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total // self.max_frames)
        frames = []
        idx = 0
        while cap.isOpened() and len(frames) < self.max_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            frame = self._crop_face(frame)
            spatial, _ = self.img_feat.extract(frame)
            frames.append(spatial)
            idx += step
        cap.release()
        if not frames:
            frames = [np.zeros((64, 64, 3), dtype=np.float32)]
        return frames


# ---------------------------------------------------------------------------
# LAYER 3 — DETECTION LAYER (per-modality detectors)
# ---------------------------------------------------------------------------

# ---- 3a. Text detector (fine-tuned DistilRoBERTa, CPU) ----

class TextDetector(nn.Module):
    """DistilRoBERTa-base fine-tuned for AI-text detection.
    Falls back to a lightweight linear classifier on perplexity features
    when HuggingFace weights are not yet fine-tuned (zero-shot mode).
    """

    def __init__(self, model_name: str = "distilroberta-base"):
        super().__init__()
        self.model_name = model_name
        self._hf_model = None
        # Fallback linear layer on 768-dim CLS embedding
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def load_pretrained(self):
        """Call once to load HuggingFace backbone."""
        self._hf_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=2
        )
        self._hf_model.eval()

    def forward(self, encoding: Dict) -> torch.Tensor:
        """Returns [prob_human, prob_ai_generated]."""
        if self._hf_model is not None:
            with torch.no_grad():
                out = self._hf_model(**encoding)
            return F.softmax(out.logits, dim=-1)
        # Fallback: random baseline (replace with trained weights)
        dummy = torch.randn(1, 768)
        logits = self.classifier(dummy)
        return F.softmax(logits, dim=-1)


# ---- 3b. Image detector (Frequency-aware CNN) ----

class FreqAwareCNN(nn.Module):
    """Lightweight CNN fusing spatial + frequency-domain features.
    6 conv blocks — CPU-trainable at 64×64 resolution.
    """

    def __init__(self):
        super().__init__()
        # Spatial branch
        self.spatial = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        )
        # Frequency branch
        self.freq = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        )
        # Fusion head
        self.head = nn.Sequential(
            nn.Linear(64 * 8 * 8 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2),
        )

    def forward(self, spatial: torch.Tensor, freq: torch.Tensor) -> torch.Tensor:
        s = self.spatial(spatial).flatten(1)
        f = self.freq(freq).flatten(1)
        return F.softmax(self.head(torch.cat([s, f], dim=1)), dim=-1)


# ---- 3c. Audio detector (Spectrogram CNN) ----

class AudioSpectrogramCNN(nn.Module):
    """Shallow CNN on log-mel spectrogram — runs fast on CPU."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        x = self.net(spec).flatten(1)
        return F.softmax(self.classifier(x), dim=-1)


# ---- 3d. Video detector (frame-averaged CNN/ViT-lite) ----

class VideoFrameCNN(nn.Module):
    """Applies FreqAwareCNN to each frame, temporally averages scores."""

    def __init__(self):
        super().__init__()
        self.frame_model = FreqAwareCNN()

    def forward(self, frames_spatial: torch.Tensor, frames_freq: torch.Tensor) -> torch.Tensor:
        # frames_spatial: (T, 3, 64, 64), frames_freq: (T, 1, 64, 64)
        scores = []
        for i in range(frames_spatial.shape[0]):
            s = frames_spatial[i].unsqueeze(0)
            f = frames_freq[i].unsqueeze(0)
            scores.append(self.frame_model(s, f))
        return torch.stack(scores).mean(0)  # temporal average


# ---------------------------------------------------------------------------
# LAYER 4 — FUSION LAYER (meta-classifier)
# ---------------------------------------------------------------------------

class FusionLayer(nn.Module):
    """Learns optimal weights to combine per-modality AI-scores.
    Input: up to 4 scores (one per modality), concatenated with one-hot
    mask indicating which modalities are active.
    """

    def __init__(self, n_modalities: int = 4):
        super().__init__()
        self.n = n_modalities
        self.net = nn.Sequential(
            nn.Linear(n_modalities * 2, 32),   # scores + presence mask
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(
        self, scores: Dict[str, float], active: List[str]
    ) -> Tuple[torch.Tensor, float]:
        """Returns (prob_tensor[2], ai_confidence_float)."""
        order = ["text", "image", "audio", "video"]
        score_vec = []
        mask_vec = []
        for m in order:
            if m in active and m in scores:
                score_vec.append(scores[m])
                mask_vec.append(1.0)
            else:
                score_vec.append(0.5)   # neutral prior
                mask_vec.append(0.0)
        x = torch.tensor(score_vec + mask_vec, dtype=torch.float32).unsqueeze(0)
        logits = self.net(x)
        probs = F.softmax(logits, dim=-1)
        return probs, float(probs[0, 1].item())


# ---------------------------------------------------------------------------
# LAYER 5 — EXPLAINABILITY LAYER
# ---------------------------------------------------------------------------

class ExplainabilityLayer:
    """Real, model-based explanations (implemented in explainability.py):
    text -> LIME word weights | image / audio / video -> Grad-CAM heatmaps |
    fusion -> exact Shapley values (SHAP definition) per modality.
    An explanation can never break a detection: failures come back as a short message."""

    TEXT_MAX_LEN = 128      # keep equal to TextFeatureExtractor.extract's default and training --max_len
    LIME_SAMPLES = 150      # more = steadier word weights, slower

    def __init__(self, detector=None):
        self.det = detector

    @staticmethod
    def _unavailable(score, reason):
        return {"method": "unavailable",
                "ai_score": None if score is None else round(float(score), 4),
                "explanation": reason}

    def _safe(self, name, score, fn):
        try:
            out = fn()
            if isinstance(out, dict) and score is not None and out.get("method") != "unavailable":
                out["ai_score"] = round(float(score), 4)       # the exact score the detector reported
            return out
        except Exception as e:
            logger.exception("Explanation for %s failed", name)
            return self._unavailable(score, f"Explanation could not be generated ({type(e).__name__}: {e}).")

    def explain_text(self, text: str, score: float) -> Dict:
        def run():
            hf = self.det.text_det._hf_model
            if hf is None:
                return self._unavailable(score, "The text model weights are not loaded, so there is nothing to explain.")
            tok = self.det.text_feat.tokenizer

            def predict_fn(texts):
                probs = []
                for i in range(0, len(texts), 32):
                    enc = tok(texts[i:i + 32], max_length=self.TEXT_MAX_LEN, truncation=True,
                              padding=True, return_tensors="pt")
                    with torch.no_grad():
                        probs += F.softmax(hf(**enc).logits, dim=-1)[:, 1].tolist()
                return np.array(probs)

            return _explainability().lime_text(predict_fn, text, score=score, num_samples=self.LIME_SAMPLES)
        return self._safe("text", score, run)

    def explain_image(self, img: np.ndarray, score: float) -> Dict:
        return self._safe("image", score, lambda: _explainability().explain_image_cnn(
            self.det.img_det, self.det.img_feat, img))

    def explain_audio(self, audio: np.ndarray, sr: int, score: float) -> Dict:
        return self._safe("audio", score, lambda: _explainability().explain_audio_cnn(
            self.det.audio_det, self.det.audio_feat, audio, sr))

    def explain_video(self, path: str, score: float) -> Dict:
        def run():
            if self.det.video_effnet is None:
                return self._unavailable(score, "Video explanations need the EfficientNet model "
                                                "(checkpoints/video_effnet.pt).")
            return _video_effnet().explain_video(self.det.video_effnet, path)
        return self._safe("video", score, run)

    def explain_fusion(self, scores: Dict[str, float], active: List[str], threshold: float) -> Dict:
        def run():
            def fusion_fn(sc, act):
                with torch.no_grad():
                    return self.det.fusion(sc, act)[1]
            return _explainability().explain_fusion(fusion_fn, scores, active, threshold)
        return self._safe("fusion", None, run)


# ---------------------------------------------------------------------------
# LAYER 6 — OUTPUT LAYER (orchestrator)
# ---------------------------------------------------------------------------

class MultimodalDetector:
    """Main 6-layer orchestrator. CPU-safe inference pipeline."""

    THRESHOLD = 0.55   # probability above which content is flagged AI-generated

    def __init__(self):
        logger.info("Initialising 6-layer multimodal detector …")

        # Layer 2 – feature extractors
        self.text_feat = TextFeatureExtractor()
        self.audio_feat = AudioFeatureExtractor()
        self.img_feat = ImageFeatureExtractor()
        self.video_feat = VideoFeatureExtractor()

        # Layer 3 – detectors
        self.text_det = TextDetector()
        self.img_det = FreqAwareCNN()
        self.audio_det = AudioSpectrogramCNN()
        self.video_det = VideoFrameCNN()

        # Layer 4 – fusion
        self.fusion = FusionLayer()

        # EfficientNet-B0 video detector (loaded in load_weights from video_effnet.pt;
        # when present it replaces self.video_det for video scoring)
        self.video_effnet = None

        # Layer 5 – explainability
        self.explain = ExplainabilityLayer(self)

        # Set all to eval mode
        for m in [self.text_det, self.img_det, self.audio_det, self.video_det, self.fusion]:
            m.eval()

        logger.info("Detector ready.")

    def load_weights(self, ckpt: str = "checkpoints"):
        """Load whatever trained weights exist in `ckpt` (text/, image.pt, audio.pt,
        video.pt, fusion.pt). Modalities without weights keep random weights."""
        ck = Path(ckpt)
        if (ck / "text").exists():
            self.text_det._hf_model = AutoModelForSequenceClassification.from_pretrained(
                ck / "text").eval()
        for attr, fname in [("img_det", "image.pt"), ("audio_det", "audio.pt"),
                            ("video_det", "video.pt"), ("fusion", "fusion.pt")]:
            if (ck / fname).exists():
                getattr(self, attr).load_state_dict(torch.load(ck / fname, map_location="cpu"))
                getattr(self, attr).eval()
        if (ck / "video_effnet.pt").exists():
            self.video_effnet = _video_effnet().load_video_effnet(ck / "video_effnet.pt")
        return self

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, payload: InputPayload, explain: bool = True) -> DetectionResult:
        """Run full pipeline and return DetectionResult. explain=False skips Layer 5 (faster)."""
        result = DetectionResult()
        modality_scores: Dict[str, float] = {}
        explanations: Dict[str, Any] = {}
        inputs: Dict[str, Any] = {}          # decoded inputs, reused by the explainers
        active = payload.active_modalities

        # ── LAYER 2 + 3: per-modality feature extraction + detection ──

        if "text" in active:
            score = self._detect_text(payload.text)
            modality_scores["text"] = score

        if "image" in active:
            img = self._load_image(payload)
            inputs["image"] = img
            score = self._detect_image(img)
            modality_scores["image"] = score

        if "audio" in active:
            audio, sr = self._load_audio(payload)
            inputs["audio"] = (audio, sr)
            score = self._detect_audio(audio, sr)
            modality_scores["audio"] = score

        if "video" in active:
            score = self._detect_video(payload.video_path)
            modality_scores["video"] = score

        # ── LAYER 4: fusion ──
        probs, ai_conf = self.fusion(modality_scores, active)

        # ── LAYER 5: explanations (fusion first = the overall verdict, then each modality) ──
        if explain:
            explanations["fusion"] = self.explain.explain_fusion(modality_scores, active, self.THRESHOLD)
            if "text" in modality_scores:
                explanations["text"] = self.explain.explain_text(payload.text, modality_scores["text"])
            if "image" in modality_scores:
                explanations["image"] = self.explain.explain_image(inputs["image"], modality_scores["image"])
            if "audio" in modality_scores:
                aud, aud_sr = inputs["audio"]
                explanations["audio"] = self.explain.explain_audio(aud, aud_sr, modality_scores["audio"])
            if "video" in modality_scores:
                explanations["video"] = self.explain.explain_video(payload.video_path, modality_scores["video"])

        # ── LAYER 6: output ──
        label = "AI-generated" if ai_conf >= self.THRESHOLD else "Human/Authentic"
        result.final_label = label
        result.confidence = round(ai_conf, 4)
        result.modality_scores = {k: round(v, 4) for k, v in modality_scores.items()}
        result.explanations = explanations
        result.report = self._build_report(label, ai_conf, modality_scores, active)
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _detect_text(self, text: str) -> float:
        feats = self.text_feat.extract(text)
        with torch.no_grad():
            probs = self.text_det(feats["encoding"])
        return float(probs[0, 1].item())   # prob of AI-generated

    def _detect_image(self, img: np.ndarray) -> float:
        spatial, freq = self.img_feat.extract(img)
        sp_t = torch.tensor(spatial).permute(2, 0, 1).unsqueeze(0)  # (1,3,64,64)
        fr_t = torch.tensor(freq).unsqueeze(0).unsqueeze(0)          # (1,1,64,64)
        with torch.no_grad():
            probs = self.img_det(sp_t, fr_t)
        return float(probs[0, 1].item())

    def _detect_audio(self, audio: np.ndarray, sr: int) -> float:
        spec = self.audio_feat.extract(audio, sr)
        sp_t = torch.tensor(spec).unsqueeze(0).unsqueeze(0)           # (1,1,64,64)
        with torch.no_grad():
            probs = self.audio_det(sp_t)
        return float(probs[0, 1].item())

    def _detect_video(self, path: str) -> float:
        if self.video_effnet is not None:                    # trained EfficientNet-B0 path
            return _video_effnet().score_video(self.video_effnet, path)
        frames = self.video_feat.extract(path)
        spatials = [torch.tensor(f).permute(2, 0, 1) for f in frames]
        sp_t = torch.stack(spatials)                                   # (T,3,64,64)
        # Compute freq per frame
        freq_list = []
        for f in frames:
            gray = cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            _, freq = self.img_feat.extract(gray)
            freq_list.append(torch.tensor(freq).unsqueeze(0))
        fr_t = torch.stack(freq_list)                                  # (T,1,64,64)
        with torch.no_grad():
            probs = self.video_det(sp_t, fr_t)
        return float(probs[0, 1].item())

    def _load_image(self, payload: InputPayload) -> np.ndarray:
        if payload.image_array is not None:
            return payload.image_array
        img = cv2.imread(payload.image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {payload.image_path}")
        return img

    def _load_audio(self, payload: InputPayload) -> Tuple[np.ndarray, int]:
        max_sec = 5.0   # training used the first 5 s of each clip
        if payload.audio_array is not None:
            return payload.audio_array[: int(max_sec * payload.audio_sr)], payload.audio_sr
        audio, sr = librosa.load(payload.audio_path, sr=None, mono=True, duration=max_sec)
        return audio, sr

    def _build_report(
        self, label: str, conf: float, scores: Dict[str, float], active: List[str]
    ) -> str:
        lines = [
            "=" * 60,
            "   MULTIMODAL AI-CONTENT DETECTION REPORT",
            "=" * 60,
            f"  Final Decision  : {label}",
            f"  Confidence      : {conf:.1%}",
            f"  Active Modalities: {', '.join(active) or 'none'}",
            "-" * 60,
            "  Per-Modality Scores (AI-probability):",
        ]
        for mod in ["text", "image", "audio", "video"]:
            if mod in scores:
                lines.append(f"    {mod.capitalize():<10}: {scores[mod]:.1%}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    detector = MultimodalDetector()

    # Text-only smoke test
    payload = InputPayload(
        text=(
            "Certainly, I would be delighted to facilitate your request. "
            "Moreover, leveraging the synergies will ensure optimal outcomes."
        )
    )
    result = detector.predict(payload)
    print(result.report)
    print("Explanations:", result.explanations)