"""
multimodal_tab.py - reusable fusion tab for the Streamlit app (frontend/, next to app.py).

Used by the "Multimodal" tab in app.py: any 2 of text/image/audio/video (endpoint /detect/multimodal).
Pass required=("text", "image") to make specific inputs mandatory instead.

It reuses call_api() and render_result() from app.py and adds: the mandatory-input note, a Run button
that stays disabled until the requirement is met, the exact-reasoning panel, an optional
PolitiFact/BOOM check and the digital trust score.
"""

import sys
from pathlib import Path

import streamlit as st

# NOTE: do not write "from model.multimodal_reasoning import ..." - model/ contains model.py,
# so the name "model" resolves to that module, not to a package. Import the file directly.
_MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
if str(_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_MODEL_DIR))

from multimodal_reasoning import (MANDATORY_NOTE, ORDER, THRESHOLD, can_run,  # noqa: E402
                                  explain_multimodal, lookup_claim, trust_score)

IMG_T = ["jpg", "jpeg", "png", "webp"]
AUD_T = ["wav", "mp3", "flac"]
VID_T = ["mp4", "avi", "mov"]


def render_multimodal_tab(call_api, render_result, fusion=None, threshold=THRESHOLD, *,
                          key="m", endpoint="/detect/multimodal",
                          modalities=("text", "image", "audio", "video"), required=None,
                          title="🔀 Multimodal Fusion Detection", note=None):
    st.markdown(f"### {title}")
    st.caption("The fusion layer (Layer 4) learns weights to combine the detectors into one "
               "authenticity score.")
    st.info("📌 " + (note or MANDATORY_NOTE))

    inputs = {}
    if "text" in modalities:
        inputs["text"] = st.text_area("Text:", key=f"{key}_text", height=110,
                                      placeholder="Paste the caption, claim, email or message …")
    media = [m for m in modalities if m != "text"]
    cols = st.columns(len(media)) if media else []
    types = {"image": IMG_T, "audio": AUD_T, "video": VID_T}
    for col, m in zip(cols, media):
        with col:
            inputs[m] = st.file_uploader(m.title(), type=types[m], key=f"{key}_{m}")
            if m == "image" and inputs[m] is not None:
                st.image(inputs[m], width=220)

    ok, msg = can_run(inputs, required=required, note=note)
    (st.success if ok else st.warning)(msg)
    st.caption("Tip: press Ctrl+Enter after typing text so the button updates.")

    text_val = (inputs.get("text") or "")
    use_fc = st.checkbox("Also compare the text with PolitiFact fact-checks "
                         "(offline archive; no API key needed)", key=f"{key}_fc",
                         disabled=not text_val.strip()) if "text" in modalities else False

    run = st.button("🔍 Run analysis", type="primary", disabled=not ok, key=f"{key}_run")

    if run:
        ok, msg = can_run(inputs, required=required, note=note)     # re-check on click
        if not ok:
            st.error(msg)
            return
        data, files = {}, {}
        if text_val.strip():
            data["text"] = text_val
        if inputs.get("image"):
            f = inputs["image"]
            files["image"] = (f.name, f.getvalue(), f.type)
        if inputs.get("audio"):
            f = inputs["audio"]
            files["audio"] = (f.name, f.getvalue(), "audio/wav")
        if inputs.get("video"):
            f = inputs["video"]
            files["video"] = (f.name, f.getvalue(), "video/mp4")
        result = call_api(endpoint, data=data, files=files if files else None)
        if result:
            fc = lookup_claim(text_val) if (use_fc and text_val.strip()) else None
            st.session_state[f"{key}_out"] = {"result": result, "fc": fc}

    out = st.session_state.get(f"{key}_out")
    if not out:
        return
    result, fc = out["result"], out["fc"]

    render_result(result)                               # your existing verdict / scores / report

    raw = result.get("modality_scores", {}) or {}
    scores = {k: (float(raw[k]) if raw.get(k) is not None else None) for k in ORDER}
    if sum(v is not None for v in scores.values()) < 2:
        st.warning("The backend returned fewer than 2 modality scores.")
        return

    expl = explain_multimodal(fusion, scores, threshold)
    trust = trust_score(expl["fused_p"], expl["agreement"], fc, None,
                        n_modalities=sum(v is not None for v in scores.values()))

    st.markdown('<p class="section-header">🧠 Reasoning for the multimodal output</p>',
                unsafe_allow_html=True)
    if not expl["exact"]:
        st.caption("Fusion layer not found in the backend process, so leave-one-out is unavailable "
                   "and the value below is the plain average of the modality scores.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Fusion P(fake)" if expl["exact"] else "Average score (proxy)", f"{expl['fused_p']:.3f}")
    c2.metric("Threshold", f"{expl['threshold']}")
    c3.metric("Digital trust score", f"{trust['score']:.0f} / 100", trust["band"])
    st.progress(min(max(trust["score"] / 100.0, 0.0), 1.0))

    st.dataframe([{"Modality": r["modality"],
                   "Own score (P fake)": round(r["score"], 3),
                   "Own verdict": r["own_verdict"],
                   "Fusion without it": None if r["fusion_without"] is None else round(r["fusion_without"], 3),
                   "Contribution": None if r["contribution"] is None else round(r["contribution"], 3)}
                  for r in expl["per_modality"]], use_container_width=True, hide_index=True)
    st.markdown(expl["text"])

    with st.expander("How the digital trust score was built"):
        st.write({k: round(v, 3) for k, v in trust["components"].items()})
        st.caption(f"Evidence coverage: {trust['evidence_coverage']:.0%}. Weights are design choices "
                   "(heuristic), not a calibrated probability.")
        for n in trust["notes"]:
            st.caption("- " + n)

    if fc is not None:
        with st.expander("Fact-check matches (PolitiFact)"):
            if not fc.get("available"):
                st.info(fc.get("reason", "Fact-check unavailable."))
            else:
                if fc.get("source") == "local":
                    st.caption(f"Offline comparison with {fc.get('n_archive', 0):,} PolitiFact-rated claims "
                               "(LIAR dataset, roughly 2007-2016). It cannot verify new claims. Only close "
                               "matches (similarity 0.75 or more) feed the trust score.")
                if not fc.get("reviews"):
                    st.info("No similar fact-checked claim found. That is not evidence the claim is true or false.")
                for r in fc.get("reviews", []):
                    if "error" in r:
                        st.caption(f"{r['site']}: lookup failed ({r['error']})")
                        continue
                    sim = f" · similarity {r['similarity']:.2f}" if "similarity" in r else ""
                    used = "" if r.get("used_for_score", True) else " (related only, not used in score)"
                    link = f" ([source]({r['url']}))" if r.get("url") else ""
                    st.markdown(f"**{r.get('publisher') or r['site']}** - {r.get('rating')}{sim}{used}{link}")
                    if r.get("claim"):
                        st.caption(f"Matched claim: {r['claim']}")