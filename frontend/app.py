"""
==========================================================
  Streamlit Frontend — Multimodal AI-Content Detector
==========================================================
Dissertation: "Combating the Rise of AI-Synthesized Malicious Content"

Run:  streamlit run app.py
Requires backend running at http://localhost:8000
"""

import io
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE = "http://localhost:8000"
PAGE_ICON = "🔍"

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Multimodal AI-Content Detector",
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1f3864; margin-bottom: 0; }
    .subtitle   { font-size: 1rem; color: #555; margin-top: 0.2rem; margin-bottom: 2rem; }
    .label-ai   { background:#ff4b4b; color:#fff; padding:6px 18px; border-radius:8px; font-weight:700; font-size:1.1rem; }
    .label-human{ background:#21c55d; color:#fff; padding:6px 18px; border-radius:8px; font-weight:700; font-size:1.1rem; }
    .metric-card{
        background:#f8f9fa; border:1px solid #dee2e6;
        border-radius:12px; padding:18px; margin:8px 0;
    }
    .section-header{ font-size:1.2rem; font-weight:600; color:#1f3864; margin-top:1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/artificial-intelligence.png", width=72
    )
    st.markdown("## ⚙️ Settings")
    api_url = st.text_input("Backend API URL", value=API_BASE)
    st.markdown("---")
    st.markdown("### 📖 About")
    st.info(
        "**MTech Dissertation**\n\n"
        "_Combating the Rise of AI-Synthesized Malicious Content_\n\n"
        "6-layer pipeline:\n"
        "1. Input\n2. Feature Extraction\n3. Detection\n4. Fusion\n"
        "5. Explainability\n6. Output"
    )
    st.markdown("---")
    if st.button("🔗 Check API Health"):
        try:
            r = requests.get(f"{api_url}/health", timeout=5)
            if r.status_code == 200 and r.json().get("model_loaded"):
                st.success("✅ API Online & Model Loaded")
            else:
                st.warning("⚠️ API reachable but model not loaded")
        except Exception:
            st.error("❌ Cannot reach API. Start the FastAPI backend.")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(
    '<p class="main-title">🔍 Multimodal AI-Content Detector</p>', unsafe_allow_html=True
)
st.markdown(
    '<p class="subtitle">Detect AI-generated Text · Images · Audio · Video — '
    "powered by a 6-layer deep-learning pipeline</p>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_text, tab_image, tab_audio, tab_video, tab_multi, tab_datasets = st.tabs(
    ["📝 Text", "🖼️ Image", "🎙️ Audio", "🎬 Video", "🔀 Multimodal", "📦 Datasets"]
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def render_result(result: Dict[str, Any]):
    """Render detection result in a structured UI block."""
    label = result.get("final_label", "unknown")
    conf  = result.get("confidence", 0.0)
    scores = result.get("modality_scores", {})
    explanations = result.get("explanations", {})
    report = result.get("report", "")

    st.markdown("---")
    st.markdown("### 📊 Detection Result")

    # Final verdict
    col_lbl, col_conf = st.columns([2, 3])
    with col_lbl:
        if "AI" in label:
            st.markdown(f'<span class="label-ai">🤖 {label}</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="label-human">✅ {label}</span>', unsafe_allow_html=True)
    with col_conf:
        st.metric("Fusion Confidence", f"{conf:.1%}")

    # Modality scores
    if scores:
        st.markdown('<p class="section-header">Per-Modality AI Probability</p>', unsafe_allow_html=True)
        cols = st.columns(len(scores))
        icons = {"text": "📝", "image": "🖼️", "audio": "🎙️", "video": "🎬"}
        for i, (mod, sc) in enumerate(scores.items()):
            with cols[i]:
                delta_color = "inverse" if sc < 0.5 else "normal"
                st.metric(
                    label=f"{icons.get(mod,'🔵')} {mod.title()}",
                    value=f"{sc:.1%}",
                    delta=f"{'⚠️ High' if sc >= 0.55 else '✅ Low'} Risk",
                    delta_color=delta_color,
                )

    # Progress bars
    st.markdown('<p class="section-header">Score Bars</p>', unsafe_allow_html=True)
    for mod, sc in scores.items():
        st.write(f"{icons.get(mod,'🔵')} **{mod.title()}**")
        st.progress(sc)

    # Explainability
    if explanations:
        st.markdown('<p class="section-header">💡 Explanations (Layer 5)</p>', unsafe_allow_html=True)
        for mod, exp in explanations.items():
            with st.expander(f"{icons.get(mod,'🔵')} {mod.title()} — {exp.get('method','')}"):
                st.write(exp.get("explanation", ""))
                if "highlighted_tokens" in exp and exp["highlighted_tokens"]:
                    st.warning("⚠️ AI-marker tokens: " + ", ".join(exp["highlighted_tokens"]))

    # Full report
    with st.expander("📋 Full Report"):
        st.code(report, language=None)


def call_api(endpoint: str, **kwargs) -> Optional[Dict]:
    """Generic API caller with error handling."""
    url = f"{api_url}{endpoint}"
    try:
        with st.spinner("Running detection pipeline …"):
            t0 = time.time()
            if "json" in kwargs:
                resp = requests.post(url, json=kwargs["json"], timeout=60)
            else:
                resp = requests.post(url, **{k: v for k, v in kwargs.items() if k != "json"}, timeout=60)
            elapsed = time.time() - t0
        if resp.status_code == 200:
            st.caption(f"⏱️ Inference time: {elapsed:.2f}s")
            return resp.json()
        else:
            st.error(f"API Error {resp.status_code}: {resp.text}")
    except requests.ConnectionError:
        st.error("❌ Cannot connect to API. Ensure the FastAPI backend is running on port 8000.")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
    return None


# ---------------------------------------------------------------------------
# Tab: Text
# ---------------------------------------------------------------------------

with tab_text:
    st.markdown("### 📝 AI Text Detection")
    st.caption(
        "Detects GPT-generated phishing emails, fake reviews, and synthetic news articles."
    )
    example_ai = (
        "Certainly, I would be delighted to assist you. "
        "Moreover, leveraging these synergies will ensure we facilitate "
        "an optimal outcome and streamline your workflow effectively."
    )
    example_human = (
        "Hi John, just checking in on the report from last week. "
        "Could you send it over when you get a chance? Thanks!"
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🤖 Load AI Example"):
            st.session_state["text_input"] = example_ai
    with col2:
        if st.button("👤 Load Human Example"):
            st.session_state["text_input"] = example_human

    text_input = st.text_area(
        "Enter text to analyse:",
        value=st.session_state.get("text_input", ""),
        height=180,
        placeholder="Paste email, article, message, or review here …",
    )
    if st.button("🔍 Analyse Text", type="primary"):
        if not text_input.strip():
            st.warning("Please enter some text.")
        else:
            result = call_api("/detect/text", json={"text": text_input})
            if result:
                render_result(result)

# ---------------------------------------------------------------------------
# Tab: Image
# ---------------------------------------------------------------------------

with tab_image:
    st.markdown("### 🖼️ AI Image Detection")
    st.caption(
        "Detects GAN-generated faces, diffusion-model images, and synthetic ID photos "
        "using frequency-domain artefact analysis."
    )
    uploaded_img = st.file_uploader(
        "Upload an image (JPG / PNG / WEBP):", type=["jpg", "jpeg", "png", "webp"]
    )
    if uploaded_img:
        st.image(uploaded_img, caption="Uploaded Image", use_container_width=True)
        if st.button("🔍 Analyse Image", type="primary"):
            files = {"file": (uploaded_img.name, uploaded_img.getvalue(), uploaded_img.type)}
            result = call_api("/detect/image", files=files)
            if result:
                render_result(result)

# ---------------------------------------------------------------------------
# Tab: Audio
# ---------------------------------------------------------------------------

with tab_audio:
    st.markdown("### 🎙️ AI Audio / Voice Detection")
    st.caption(
        "Detects cloned/synthetic voices used in scam calls. "
        "Analyses log-mel spectrogram patterns associated with neural vocoders."
    )
    uploaded_audio = st.file_uploader(
        "Upload an audio file (WAV / MP3 / FLAC):", type=["wav", "mp3", "flac", "ogg"]
    )
    if uploaded_audio:
        st.audio(uploaded_audio)
        if st.button("🔍 Analyse Audio", type="primary"):
            files = {"file": (uploaded_audio.name, uploaded_audio.getvalue(), "audio/wav")}
            result = call_api("/detect/audio", files=files)
            if result:
                render_result(result)

# ---------------------------------------------------------------------------
# Tab: Video
# ---------------------------------------------------------------------------

with tab_video:
    st.markdown("### 🎬 Deepfake Video Detection")
    st.caption(
        "Samples key frames, aligns faces, and applies frequency-aware CNN + temporal averaging "
        "to detect face-swaps and GAN-generated video."
    )
    uploaded_video = st.file_uploader(
        "Upload a video file (MP4 / AVI / MOV):", type=["mp4", "avi", "mov", "mkv"]
    )
    if uploaded_video:
        st.video(uploaded_video)
        if st.button("🔍 Analyse Video", type="primary"):
            files = {"file": (uploaded_video.name, uploaded_video.getvalue(), "video/mp4")}
            result = call_api("/detect/video", files=files)
            if result:
                render_result(result)

# ---------------------------------------------------------------------------
# Tab: Multimodal
# ---------------------------------------------------------------------------

with tab_multi:
    st.markdown("### 🔀 Multimodal Fusion Detection")
    st.caption(
        "Combine any or all modalities. The fusion layer (Layer 4) learns optimal weights "
        "to produce a single authenticity score."
    )
    st.info(
        "💡 Tip: Use multiple modalities together for higher accuracy than any single detector."
    )

    multi_text = st.text_area(
        "Text (optional):",
        height=100,
        placeholder="Paste suspicious email, caption, or message …",
    )
    col_img, col_aud, col_vid = st.columns(3)
    with col_img:
        multi_img = st.file_uploader("Image (optional)", type=["jpg", "jpeg", "png", "webp"], key="m_img")
    with col_aud:
        multi_aud = st.file_uploader("Audio (optional)", type=["wav", "mp3", "flac"], key="m_aud")
    with col_vid:
        multi_vid = st.file_uploader("Video (optional)", type=["mp4", "avi", "mov"], key="m_vid")

    if st.button("🔍 Run Multimodal Analysis", type="primary"):
        has_input = any([multi_text.strip(), multi_img, multi_aud, multi_vid])
        if not has_input:
            st.warning("Please provide at least one modality input.")
        else:
            data = {}
            files = {}
            if multi_text.strip():
                data["text"] = multi_text
            if multi_img:
                files["image"] = (multi_img.name, multi_img.getvalue(), multi_img.type)
            if multi_aud:
                files["audio"] = (multi_aud.name, multi_aud.getvalue(), "audio/wav")
            if multi_vid:
                files["video"] = (multi_vid.name, multi_vid.getvalue(), "video/mp4")

            result = call_api("/detect/multimodal", data=data, files=files if files else None)
            if result:
                render_result(result)

# ---------------------------------------------------------------------------
# Tab: Datasets
# ---------------------------------------------------------------------------

with tab_datasets:
    st.markdown("### 📦 Recommended CPU-Friendly Datasets")
    st.caption("All datasets listed are suitable for CPU-based training and experimentation.")

    try:
        resp = requests.get(f"{api_url}/modalities", timeout=10)
        mods = resp.json().get("modalities", {})
    except Exception:
        mods = {}
        st.warning("Could not fetch dataset info from API. Showing static data.")
        mods = {
            "text": {"datasets": [{"name": "HC3", "url": "https://huggingface.co/datasets/Hello-SimpleAI/HC3", "size": "~40k", "note": "Human vs ChatGPT pairs"}]},
        }

    icons = {"text": "📝", "image": "🖼️", "audio": "🎙️", "video": "🎬"}
    for mod, info in mods.items():
        with st.expander(f"{icons.get(mod,'🔵')} {mod.title()} — {info.get('description','')}"):
            st.markdown(f"**Detector:** `{info.get('detector','')}`")
            datasets = info.get("datasets", [])
            for ds in datasets:
                st.markdown(f"#### 📁 {ds['name']}")
                col_a, col_b = st.columns([2, 3])
                with col_a:
                    st.markdown(f"**Size:** {ds['size']}")
                    st.markdown(f"[🔗 Download / Access]({ds['url']})")
                with col_b:
                    st.info(ds["note"])
                st.markdown("---")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown(
    "<center style='color:#888; font-size:0.85rem;'>"
    "MTech Dissertation · Multimodal AI-Content Detection · "
    "6-Layer Pipeline: Input → Feature Extraction → Detection → Fusion → Explainability → Output"
    "</center>",
    unsafe_allow_html=True,
)
