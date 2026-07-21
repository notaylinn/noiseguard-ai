"""
Settings page — lets the user point the frontend at a different backend
URL at runtime (useful for local dev vs. deployed demo) and view the
current backend health/configuration.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from components.api_client import check_health  # noqa: E402

st.set_page_config(page_title="Settings — NoiseGuard AI", page_icon="⚙️", layout="wide")
st.title("⚙️ Settings")

st.subheader("Backend connection")
current_url = os.environ.get("NOISEGUARD_API_URL", "http://localhost:8000/api/v1")
st.text_input(
    "Backend API base URL",
    value=current_url,
    disabled=True,
    help="Set the NOISEGUARD_API_URL environment variable before launching Streamlit to change this.",
)

health = check_health()
if health:
    st.success("Connected")
    st.json(health)
else:
    st.error("Backend unreachable. Start it with: `uvicorn backend.app.main:app --reload`")

st.divider()
st.subheader("About this app")
st.write(
    "NoiseGuard AI is an MVP built for the Tech Vision hackathon. All analysis is performed locally "
    "via a real YAMNet model — no audio is sent to third-party cloud services."
)
st.caption("Version 1.0.0")
