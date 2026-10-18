"""
FastAPI Backend for the Multimodal AI-Content Detector (text, image, audio, video).
MTech Dissertation: "Combating the Rise of AI-Synthesized Malicious Content"

Endpoints:
  POST /detect/text          — detect AI-generated text
  POST /detect/image         — detect GAN/diffusion image
  POST /detect/audio         — detect voice cloning
  POST /detect/video         — detect deepfake video
  POST /detect/multimodal    — detect using any combination of modalities
  GET  /health               — liveness check
  GET  /modalities           — list supported modalities + dataset info
"""

import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import librosa
import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure model directory is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "model"))
from model import MultimodalDetector, InputPayload, DetectionResult

# Folder with the trained weights (text/, image.pt, audio.pt, video_effnet.pt, fusion.pt).
# Default: <project root>/checkpoints. Override with the CKPT_DIR environment variable.
CKPT = Path(os.environ.get("CKPT_DIR", Path(__file__).resolve().parent.parent / "checkpoints"))

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Multimodal AI-Content Detector",
    description=(
        "It is a 6-layer detection pipeline to check text, image, audio and video whether they are ai generated or human. "
        "MTech Dissertation — Combating the Rise of AI-Synthesized Malicious Content."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton detector (loaded once on startup)
detector: Optional[MultimodalDetector] = None


@app.on_event("startup")
async def startup_event():
    global detector
    detector = MultimodalDetector().load_weights(CKPT)
    expected = ["text", "image.pt", "audio.pt", "video_effnet.pt", "fusion.pt"]
    missing = [n for n in expected if not (CKPT / n).exists()]
    if missing:
        print(f"WARNING: missing in {CKPT}: {missing}. Those parts run with UNTRAINED weights.")
    else:
        print(f"Loaded all trained weights from {CKPT}")


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class TextRequest(BaseModel):
    text: str


class DetectionResponse(BaseModel):
    final_label: str
    confidence: float
    modality_scores: dict
    explanations: dict
    report: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# ---------------------------------------------------------------------------
# Helper: convert DetectionResult → dict
# ---------------------------------------------------------------------------

def result_to_response(r: DetectionResult) -> DetectionResponse:
    return DetectionResponse(
        final_label=r.final_label,
        confidence=r.confidence,
        modality_scores=r.modality_scores,
        explanations=r.explanations,
        report=r.report,
    )


# ---------------------------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Liveness check."""
    return HealthResponse(status="ok", model_loaded=detector is not None)


@app.get("/modalities", tags=["System"])
async def modalities():
    """Return supported modalities with recommended CPU-friendly datasets."""
    return {
        "modalities": {
            "text": {
                "description": "Phishing emails, fake news, AI-written reviews",
                "detector": "DistilRoBERTa fine-tuned classifier",
                "datasets": [
                    {
                        "name": "HC3 (Human-ChatGPT Comparison Corpus)",
                        "url": "https://huggingface.co/datasets/Hello-SimpleAI/HC3",
                        "size": "~40k pairs",
                        "note": "Human vs ChatGPT answers across domains",
                    },
                    {
                        "name": "TuringBench",
                        "url": "https://huggingface.co/datasets/turingbench/TuringBench",
                        "size": "~200k texts",
                        "note": "19 AI generators vs human text",
                    },
                    {
                        "name": "RAID (Robust AI Detection)",
                        "url": "https://huggingface.co/datasets/liamdugan/raid",
                        "size": "~500k",
                        "note": "Covers paraphrase/attack robustness scenarios",
                    },
                    {
                        "name": "PhishTank CSV export",
                        "url": "https://www.phishtank.com/developer_info.php",
                        "size": "~10k phishing URLs + email text",
                        "note": "Combine with legitimate emails for binary classification",
                    },
                ],
            },
            "image": {
                "description": "GAN faces, diffusion-generated images, fake ID photos",
                "detector": "Frequency-Aware CNN (spatial + DFT branches)",
                "datasets": [
                    {
                        "name": "140k Real and Fake Faces (FFHQ + StyleGAN)",
                        "url": "https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces",
                        "size": "140k images (70k real, 70k fake)",
                        "note": "Best starting point; fits in CPU RAM at 64×64",
                    },
                    {
                        "name": "CIFAKE (CIFAR-10 + Stable Diffusion)",
                        "url": "https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images",
                        "size": "120k images (60k real, 60k AI)",
                        "note": "Smaller images → fast CPU training",
                    },
                    {
                        "name": "GenImage",
                        "url": "https://huggingface.co/datasets/stablediffusion-community/genimage",
                        "size": "~1.3M (sub-sample for CPU)",
                        "note": "Sub-sample 50k per class for CPU experiments",
                    },
                ],
            },
            "audio": {
                "description": "Cloned/synthetic voice used in scam calls",
                "detector": "Log-Mel Spectrogram CNN (ASVspoof-style)",
                "datasets": [
                    {
                        "name": "ASVspoof 2019 (LA track)",
                        "url": "https://datashare.ed.ac.uk/handle/10283/3336",
                        "size": "~121k utterances",
                        "note": "Gold-standard voice spoofing benchmark",
                    },
                    {
                        "name": "FakeAVCeleb (audio portion)",
                        "url": "https://github.com/DASH-Lab/FakeAVCeleb",
                        "size": "~500 videos → extract audio",
                        "note": "Real celebrity speech vs deepfake audio",
                    },
                    {
                        "name": "WaveFake",
                        "url": "https://github.com/RUB-SysSec/WaveFake",
                        "size": "~117k utterances",
                        "note": "7 different neural vocoders; CPU-safe at 16 kHz",
                    },
                ],
            },
            "video": {
                "description": "Deepfake face-swap and GAN/diffusion video",
                "detector": "EfficientNet-B0 (ImageNet-pretrained, fine-tuned) on face crops + per-frame score averaging",
                "datasets": [
                    {
                        "name": "FaceForensics++ (c40 compressed)",
                        "url": "https://github.com/ondyari/FaceForensics",
                        "size": "1000 videos × 4 manipulation types",
                        "note": "Use c40 (high compression) for CPU robustness testing",
                    },
                    {
                        "name": "Celeb-DF v2",
                        "url": "https://github.com/yuezunli/celeb-deepfakeforensics",
                        "size": "590 real + 5639 deepfake clips",
                        "note": "High visual quality deepfakes; sub-sample frames",
                    },
                    {
                        "name": "DFDC (DeepFake Detection Challenge) Preview",
                        "url": "https://ai.facebook.com/datasets/dfdc",
                        "size": "~5k preview videos",
                        "note": "Preview subset is manageable on CPU workstations",
                    },
                ],
            },
        }
    }


# ── Text ──────────────────────────────────────────────────────────────────

@app.post("/detect/text", response_model=DetectionResponse, tags=["Detection"])
async def detect_text(request: TextRequest, explain: bool = True):
    """Detect AI-generated text (phishing, fake news, AI-written content)."""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text field is empty.")
    try:
        payload = InputPayload(text=request.text)
        result = detector.predict(payload, explain=explain)
        return result_to_response(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Image ─────────────────────────────────────────────────────────────────

@app.post("/detect/image", response_model=DetectionResponse, tags=["Detection"])
async def detect_image(file: UploadFile = File(...), explain: bool = True):
    """Detect GAN/diffusion-generated images."""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Cannot decode image file.")
        payload = InputPayload(image_array=img)
        result = detector.predict(payload, explain=explain)
        return result_to_response(result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Audio ─────────────────────────────────────────────────────────────────

@app.post("/detect/audio", response_model=DetectionResponse, tags=["Detection"])
async def detect_audio(file: UploadFile = File(...), explain: bool = True):
    """Detect voice cloning / synthetic speech."""
    try:
        contents = await file.read()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        audio, sr = librosa.load(tmp_path, sr=None, mono=True)
        os.unlink(tmp_path)
        payload = InputPayload(audio_array=audio, audio_sr=sr)
        result = detector.predict(payload, explain=explain)
        return result_to_response(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Video ─────────────────────────────────────────────────────────────────

@app.post("/detect/video", response_model=DetectionResponse, tags=["Detection"])
async def detect_video(file: UploadFile = File(...), explain: bool = True):
    """Detect deepfake video (face-swap, GAN-generated)."""
    tmp_path = None
    try:
        contents = await file.read()
        suffix = Path(file.filename).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        payload = InputPayload(video_path=tmp_path)
        result = detector.predict(payload, explain=explain)
        return result_to_response(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ── Multimodal ────────────────────────────────────────────────────────────

@app.post("/detect/multimodal", response_model=DetectionResponse, tags=["Detection"])
async def detect_multimodal(
    text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None),
    explain: bool = True,
):
    """Fuse any combination of text / image / audio / video for a single authenticity score."""
    payload = InputPayload()
    tmp_files = []

    if text:
        payload.text = text

    if image:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        payload.image_array = img

    if audio:
        contents = await audio.read()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(contents)
            tmp_files.append(tmp.name)
        aud, sr = librosa.load(tmp_files[-1], sr=None, mono=True)
        payload.audio_array = aud
        payload.audio_sr = sr

    if video:
        contents = await video.read()
        suffix = Path(video.filename).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(contents)
            tmp_files.append(tmp.name)
        payload.video_path = tmp_files[-1]

    if not payload.active_modalities:
        raise HTTPException(status_code=400, detail="At least one modality input is required.")

    try:
        result = detector.predict(payload, explain=explain)
    finally:
        for f in tmp_files:
            try:
                os.unlink(f)
            except OSError:
                pass

    return result_to_response(result)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)