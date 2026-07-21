"""
History page — browse, inspect, delete past analyses and export CSV.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from components import charts  # noqa: E402
from components.api_client import ApiError, delete_analysis, download_csv_export, download_pdf_report, get_analysis, get_history  # noqa: E402

st.set_page_config(page_title="History — NoiseGuard AI", page_icon="🗂️", layout="wide")
st.title("🗂️ Analysis History")

try:
    history = get_history(limit=200)
except ApiError as e:
    st.error(f"Could not load history: {e}")
    history = []

if not history:
    st.info("No analyses yet. Go to **Upload & Record** to run your first analysis.")
else:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"Showing {len(history)} most recent analyses.")
    with col2:
        try:
            csv_bytes = download_csv_export()
            st.download_button("⬇️ Export all to CSV", data=csv_bytes, file_name="noiseguard_history.csv", mime="text/csv")
        except ApiError as e:
            st.warning(f"CSV export unavailable: {e}")

    st.plotly_chart(charts.history_timeline(history), use_container_width=True)

    for record in history:
        with st.expander(
            f"{record['created_at'][:19].replace('T', ' ')} — {record['top_label']} "
            f"({record['top_confidence']*100:.1f}%) — {record['loudness_category'].replace('_', ' ').title()}"
        ):
            c1, c2, c3 = st.columns([2, 1, 1])
            with c1:
                st.write(f"**File:** {record['filename']}")
                st.write(f"**Duration:** {record['duration_sec']:.2f}s")
                st.write(f"**Estimated level:** {record['estimated_dbfs']:.1f} dBFS")
            with c2:
                try:
                    pdf_bytes = download_pdf_report(record["id"])
                    st.download_button("📄 PDF", data=pdf_bytes, file_name=f"report_{record['id'][:8]}.pdf",
                                        mime="application/pdf", key=f"pdf_{record['id']}")
                except ApiError:
                    st.caption("PDF unavailable")
            with c3:
                if st.button("🗑️ Delete", key=f"del_{record['id']}"):
                    try:
                        delete_analysis(record["id"])
                        st.rerun()
                    except ApiError as e:
                        st.error(f"Could not delete: {e}")

            if st.button("View full details", key=f"detail_{record['id']}"):
                detail = get_analysis(record["id"])
                st.json(detail)
