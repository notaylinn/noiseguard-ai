"""
NoiseGuard AI — Streamlit landing page & multipage app entry point.

This file is the landing page. Additional pages live in frontend/pages/
and are auto-discovered by Streamlit's multipage navigation. All
network calls go through `components.api_client` — no business logic
or direct model access happens in the frontend, consistent with Clean
Architecture: the UI is a thin presentation layer over the FastAPI backend.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from components.api_client import check_health  # noqa: E402

st.set_page_config(page_title="NoiseGuard AI", page_icon="🔊", layout="wide")

st.markdown(
    """
    <style>
    .hero-title { font-size: 2.6rem; font-weight: 800; color: #1a3c5e; margin-bottom: 0; }
    .hero-subtitle { font-size: 1.15rem; color: #4a5a68; margin-top: 0.2rem; }
    .metric-card { background: #f5f8fb; border-radius: 12px; padding: 1.2rem; border: 1px solid #e0e6ec; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="hero-title">🔊 NoiseGuard AI</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">AI-powered environmental noise assessment for Kazakhstan — '
    "objective, explainable sound analysis for citizens, housing services and municipalities.</p>",
    unsafe_allow_html=True,
)

health = check_health()
col1, col2 = st.columns([3, 1])
with col2:
    if health:
        st.success(f"Backend online — {health['app_name']} v{health['version']}")
    else:
        st.error("Backend unreachable. Start the FastAPI server (see README) and refresh.")

st.divider()

st.subheader("The problem")
st.write(
    "Residents of Kazakhstani cities regularly deal with disruptive environmental noise — "
    "construction, traffic, industrial equipment, loud venues — but when they report it to a "
    "housing service (КСК), akimat or business, their complaint is usually just a subjective "
    "description ('it's very loud'). Without objective data, complaints are hard to act on and "
    "easy to dismiss."
)

st.subheader("What NoiseGuard AI does")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("#### 🎙️ Record or upload")
    st.write("Capture a sound directly in the browser or upload an existing recording.")
with c2:
    st.markdown("#### 🧠 Real AI analysis")
    st.write("YAMNet (TensorFlow Hub) classifies the sound; librosa extracts objective DSP features.")
with c3:
    st.markdown("#### 📄 Professional report")
    st.write("Get a structured interpretation, relative loudness estimate and a downloadable PDF report.")

st.divider()
st.subheader("Get started")
st.write("Use the sidebar to navigate: **Upload & Record** to analyze a sound, **History** to review past "
         "analyses, **Statistics** for aggregate dashboards, or **About** to read the full research background.")

st.info(
    "⚠️ **Important limitation:** NoiseGuard AI estimates *relative* acoustic intensity from "
    "uncalibrated recordings. It does **not** produce certified decibel (dB SPL) measurements and "
    "cannot replace an accredited sound-level meter for legal/regulatory purposes.",
    icon="⚠️",
)
