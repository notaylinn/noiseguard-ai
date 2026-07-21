"""
Statistics dashboard — aggregate view across all stored analyses.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from components import charts  # noqa: E402
from components.api_client import ApiError, get_statistics  # noqa: E402

st.set_page_config(page_title="Statistics — NoiseGuard AI", page_icon="📊", layout="wide")
st.title("📊 Statistics Dashboard")

try:
    stats = get_statistics()
except ApiError as e:
    st.error(f"Could not load statistics: {e}")
    stats = None

if not stats or stats["total_analyses"] == 0:
    st.info("No data yet. Run some analyses on the **Upload & Record** page first.")
else:
    c1, c2, c3 = st.columns(3)
    c1.metric("Total analyses", stats["total_analyses"])
    c2.metric("Average relative level", f"{stats['average_dbfs']:.1f} dBFS")
    c3.metric("Average duration", f"{stats['average_duration_sec']:.1f} s")

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(charts.category_distribution_pie(stats["category_distribution"]), use_container_width=True)
    with col_b:
        st.plotly_chart(charts.loudness_distribution_bar(stats["loudness_distribution"]), use_container_width=True)

    st.caption(
        "Category distribution is derived from each recording's top YAMNet prediction, mapped to a "
        "noise-complaint category. Loudness distribution reflects RELATIVE signal level estimates, not "
        "certified dB SPL measurements."
    )
