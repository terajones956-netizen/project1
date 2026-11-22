"""
multimodal_reasoning.py - gating, exact reasoning, fact-check validation and digital trust score
for the Multimodal tab.  Put it in model/ and import from your app.

1) GATING  - at least 2 inputs are mandatory, Run Analysis stays disabled until then.
       ok, msg = can_run(inputs)          # inputs = {"text": str|None, "image": obj|None, ...}
   Streamlit:  st.info(MANDATORY_NOTE); st.button("Run analysis", disabled=not ok)
   Gradio:     run_btn = gr.Button("Run analysis", interactive=False)
               # on every input change: run_btn.update(interactive=can_run(inputs)[0])

2) EXACT REASONING - explain_multimodal(fusion, scores) returns the numbers behind the verdict:
   the fusion probability, the threshold, each modality's own score and a leave-one-out
   contribution (re-run the SAME fusion layer with that modality masked out).

3) VALIDATION - fact_check_lookup(claim) queries the Google Fact Check Tools API restricted to
   politifact.com and boomlive.in (needs GOOGLE_FACTCHECK_API_KEY). Not tested against the live API
   here; check the response fields on your machine. Do not scrape the sites.

4) DIGITAL TRUST SCORE - trust_score(...) combines detector output, fact-check, provenance and
   cross-modality agreement into a 0-100 heuristic. Weights are design choices, not learned.
"""

import os
from pathlib import Path

import numpy as np

ORDER = ["text", "image", "audio", "video"]       # must match FusionLayer.forward
THRESHOLD = 0.55                                  # same as training_fusion.py
MIN_MODALITIES = 2
MANDATORY_NOTE = "Note: at least 2 inputs are mandatory for multimodal output."


# ---------------------------------------------------------------- 1) gating
def modalities_present(inputs):
    out = []
    for k in ORDER:
        v = inputs.get(k)
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        out.append(k)
    return out


def can_run(inputs, required=None, note=None):
    """required: optional tuple of modality names that must ALL be present (e.g. ("text", "image")).
    Without it, any MIN_MODALITIES (2) of the modalities are enough."""
    present = modalities_present(inputs)
    note = note or MANDATORY_NOTE
    if required:
        missing = [m for m in required if m not in present]
        if not missing:
            return True, f"Ready: {', '.join(present)}"
        return False, f"Add the missing input(s): {', '.join(missing)}. {note}"
    if len(present) >= MIN_MODALITIES:
        return True, f"Ready: {', '.join(present)}"
    need = MIN_MODALITIES - len(present)
    return False, f"Add {need} more input(s). {note}"


# ---------------------------------------------------------------- 2) exact reasoning
def _fuse(fusion, scores):
    import torch
    import torch.nn.functional as F
    s = np.full(4, 0.5, np.float32)
    m = np.zeros(4, np.float32)
    for i, k in enumerate(ORDER):
        if scores.get(k) is not None:
            s[i], m[i] = float(scores[k]), 1.0
    x = torch.from_numpy(np.concatenate([s, m]))[None]
    with torch.no_grad():
        return float(F.softmax(fusion.net(x), dim=-1)[0, 1])


def explain_multimodal(fusion, scores, threshold=THRESHOLD):
    """scores: {"text": 0.91, "image": 0.40, ...} (probability of AI/fake, missing = None)."""
    present = [k for k in ORDER if scores.get(k) is not None]
    if len(present) < MIN_MODALITIES:
        raise ValueError(MANDATORY_NOTE)
    if fusion is None:
        return explain_scores_only(scores, threshold)
    fused = _fuse(fusion, scores)
    verdict = "AI-generated / fake" if fused >= threshold else "likely real"

    rows = []
    for k in present:
        rest = {j: scores[j] for j in present if j != k}
        without = _fuse(fusion, rest)
        rows.append({"modality": k, "score": float(scores[k]),
                     "own_verdict": "fake" if scores[k] >= threshold else "real",
                     "fusion_without": without,
                     "contribution": fused - without})      # >0 pushed toward fake

    vals = [r["score"] for r in rows]
    spread = float(np.std(vals))
    conflict = max(vals) >= 0.8 and min(vals) <= 0.2
    agree = float(max(0.0, 1.0 - 2.0 * spread))              # 1 = same score, 0 = strong disagreement

    lines = [f"Fusion probability of fake = {fused:.3f} vs threshold {threshold} -> {verdict}.",
             f"Used {len(present)} modalities: {', '.join(present)} (the rest were masked out)."]
    for r in sorted(rows, key=lambda r: -abs(r["contribution"])):
        d = "toward fake" if r["contribution"] > 0 else "toward real"
        lines.append(f"- {r['modality']}: own score {r['score']:.3f} ({r['own_verdict']}); "
                     f"removing it moves fusion to {r['fusion_without']:.3f} "
                     f"({r['contribution']:+.3f}, pushes {d}).")
    lines.append("Modalities conflict strongly; treat the result with caution." if conflict
                 else "Modalities broadly agree." if spread < 0.2
                 else "Modalities partly disagree.")
    return {"fused_p": fused, "verdict": verdict, "threshold": threshold, "per_modality": rows,
            "agreement": agree, "conflict": conflict, "text": "\n".join(lines),
            "exact": True}


def explain_scores_only(scores, threshold=THRESHOLD):
    """Fallback when the fusion layer object is not available: no leave-one-out, and the
    'fused' value is the plain average of the modality scores (a proxy, not the real fusion)."""
    present = [k for k in ORDER if scores.get(k) is not None]
    if len(present) < MIN_MODALITIES:
        raise ValueError(MANDATORY_NOTE)
    vals = [float(scores[k]) for k in present]
    proxy = float(np.mean(vals))
    spread = float(np.std(vals))
    conflict = max(vals) >= 0.8 and min(vals) <= 0.2
    agree = float(max(0.0, 1.0 - 2.0 * spread))
    rows = [{"modality": k, "score": float(scores[k]),
             "own_verdict": "fake" if scores[k] >= threshold else "real",
             "fusion_without": None, "contribution": None} for k in present]
    lines = [f"Average of modality scores = {proxy:.3f} (proxy; the fusion layer object was not "
             f"available, so exact leave-one-out is not shown).",
             f"Used {len(present)} modalities: {', '.join(present)}."]
    for r in sorted(rows, key=lambda r: -r["score"]):
        lines.append(f"- {r['modality']}: own score {r['score']:.3f} ({r['own_verdict']}).")
    lines.append("Modalities conflict strongly; treat the result with caution." if conflict
                 else "Modalities broadly agree." if spread < 0.2
                 else "Modalities partly disagree.")
    return {"fused_p": proxy, "verdict": "AI-generated / fake" if proxy >= threshold else "likely real",
            "threshold": threshold, "per_modality": rows, "agreement": agree, "conflict": conflict,
            "text": "\n".join(lines), "exact": False}


# ---------------------------------------------------------------- 3) fact-check validation
def fact_check_lookup(claim, sites=("politifact.com", "boomlive.in"), api_key=None, timeout=10):
    import requests
    key = api_key or os.environ.get("GOOGLE_FACTCHECK_API_KEY")
    if not key:
        return {"available": False, "reason": "GOOGLE_FACTCHECK_API_KEY not set", "reviews": []}
    reviews = []
    for site in sites:
        try:
            r = requests.get("https://factchecktools.googleapis.com/v1alpha1/claims:search",
                             params={"query": claim[:300], "reviewPublisherSiteFilter": site,
                                     "pageSize": 3, "key": key}, timeout=timeout)
            r.raise_for_status()
            for c in r.json().get("claims", []):
                for cr in c.get("claimReview", []):
                    reviews.append({"site": site,
                                    "publisher": cr.get("publisher", {}).get("name"),
                                    "rating": cr.get("textualRating"),
                                    "url": cr.get("url"), "claim": c.get("text")})
        except Exception as e:                      # network / quota / key problems
            reviews.append({"site": site, "error": str(e)})
    ok = [r for r in reviews if "rating" in r]
    return {"available": True, "matched": bool(ok), "reviews": reviews}


def rating_to_credibility(text):
    """Map a fact-checker's wording to 0 (false) .. 1 (true). Returns None if unrecognised."""
    t = (text or "").lower()
    for key, v in (("pants on fire", 0.0), ("mostly false", 0.25), ("mostly true", 0.75),
                   ("half", 0.5), ("misleading", 0.25), ("false", 0.0), ("fake", 0.0),
                   ("incorrect", 0.0), ("true", 1.0), ("correct", 1.0), ("accurate", 1.0)):
        if key in t:
            return v
    return None


# ---------------------------------------------------------------- 3b) offline PolitiFact archive
# No API key needed: compares the claim with the PolitiFact-rated claims in data/politifact.csv
# (LIAR dataset: columns "statement" and "label"). Make the CSV with make_politifact_csv.py.
DEFAULT_CSV = Path(__file__).resolve().parent.parent / "data" / "politifact.csv"
STRONG_SIM = 0.75      # near-identical wording: allowed to influence the trust score
SHOW_SIM = 0.35        # related claims shown to the user but NOT used in the score
LIAR_CREDIBILITY = {"pants-fire": 0.0, "pants-on-fire": 0.0, "false": 0.0, "mostly-false": 0.25,
                    "barely-true": 0.25, "half-true": 0.5, "mostly-true": 0.75, "true": 1.0}
_LOCAL = {"path": None, "vec": None, "mat": None, "rows": None}


def _norm_label(x):
    return str(x).strip().lower().replace("_", "-").replace(" ", "-")


def _load_local(csv_path):
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    if _LOCAL["path"] == str(csv_path) and _LOCAL["mat"] is not None:
        return
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["statement", "label"]).drop_duplicates("statement")
    df["cred"] = df["label"].map(lambda x: LIAR_CREDIBILITY.get(_norm_label(x)))
    df = df.dropna(subset=["cred"]).reset_index(drop=True)
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
    mat = vec.fit_transform(df["statement"].astype(str))          # rows are L2-normalised
    _LOCAL.update(path=str(csv_path), vec=vec, mat=mat, rows=df)


def local_lookup(claim, csv_path=None, top_k=3):
    csv_path = Path(csv_path or os.environ.get("POLITIFACT_CSV") or DEFAULT_CSV)
    out = {"available": False, "matched": False, "reviews": [], "source": "local"}
    if not (claim or "").strip():
        out["reason"] = "No text to check."
        return out
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        out["reason"] = (f"{csv_path} is missing or empty. Create it with make_politifact_csv.py "
                         f"to enable the offline PolitiFact comparison.")
        return out
    try:
        _load_local(csv_path)
    except Exception as e:
        out["reason"] = f"Could not load {csv_path}: {e}"
        return out
    q = _LOCAL["vec"].transform([claim])
    sims = (_LOCAL["mat"] @ q.T).toarray().ravel()                # cosine similarity
    reviews = []
    for i in np.argsort(-sims)[:top_k]:
        sim = float(sims[i])
        if sim < SHOW_SIM:
            break
        row = _LOCAL["rows"].iloc[i]
        reviews.append({"site": "politifact.com", "publisher": "PolitiFact (LIAR archive)",
                        "rating": str(row["label"]), "credibility": float(row["cred"]),
                        "similarity": sim, "claim": str(row["statement"]), "url": None,
                        "used_for_score": sim >= STRONG_SIM})
    out.update(available=True, reviews=reviews, matched=any(r["used_for_score"] for r in reviews),
               n_archive=int(len(_LOCAL["rows"])))
    return out


def lookup_claim(claim):
    """Google Fact Check (PolitiFact + BOOM) if GOOGLE_FACTCHECK_API_KEY is set, else the offline archive."""
    if os.environ.get("GOOGLE_FACTCHECK_API_KEY"):
        return fact_check_lookup(claim)
    return local_lookup(claim)


# ---------------------------------------------------------------- 4) digital trust score
WEIGHTS = {"authenticity": 0.50, "fact_check": 0.20, "provenance": 0.20, "agreement": 0.10}


def trust_score(fused_p, agreement, fact_check=None, provenance=None, n_modalities=2):
    """0-100 heuristic. Components that are unavailable are left out and the weights renormalised.
    fact_check : result of fact_check_lookup (or None)
    provenance : dict like {"c2pa": "camera"|"ai"|"invalid"|None, "ai_watermark": 0/1}
    """
    comps = {"authenticity": 1.0 - float(fused_p), "agreement": float(agreement)}
    notes = []

    if fact_check and fact_check.get("matched"):
        vals = []
        for r in fact_check["reviews"]:
            if "rating" not in r or r.get("used_for_score") is False:
                continue                                  # weak/related matches never feed the score
            v = r.get("credibility")
            if v is None:
                v = rating_to_credibility(r.get("rating"))
            if v is not None:
                vals.append(v)
        if vals:
            comps["fact_check"] = float(np.mean(vals))
        else:
            notes.append("Fact-check found but rating not recognised; ignored.")
    else:
        notes.append("No close fact-check match; component left out (absence is not evidence).")

    if provenance:
        p = None
        if provenance.get("ai_watermark") == 1 or provenance.get("c2pa") == "ai":
            p = 0.0
        elif provenance.get("c2pa") == "camera":
            p = 1.0
        elif provenance.get("c2pa") == "invalid":
            p = 0.1
        if p is not None:
            comps["provenance"] = p
    if "provenance" not in comps:
        notes.append("No provenance evidence; component left out.")

    tw = sum(WEIGHTS[k] for k in comps)
    score = 100.0 * sum(WEIGHTS[k] * v for k, v in comps.items()) / tw
    band = "High" if score >= 75 else "Medium" if score >= 40 else "Low"
    coverage = tw / sum(WEIGHTS.values())            # share of the evidence we actually had
    return {"score": round(score, 1), "band": band, "components": comps,
            "evidence_coverage": round(coverage, 2), "n_modalities": n_modalities, "notes": notes}