"""
Upload & Record page — the core analysis workflow.

Lets the user upload an audio file OR record one directly in the
browser, sends it to the FastAPI backend for real YAMNet + librosa
analysis, and renders the full result: waveform, predictions,
confidence, relative loudness, interpretation and recommendations.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from components import charts  # noqa: E402
from components.api_client import ApiError, analyze_audio, download_pdf_report  # noqa: E402

st.set_page_config(page_title="Upload & Record — NoiseGuard AI", page_icon="🎙️", layout="wide")
st.title("🎙️ Upload or Record Audio")

tab_upload, tab_record = st.tabs(["📁 Upload a file", "🎤 Record audio"])

audio_bytes: bytes | None = None
audio_name = "recording.wav"
mime_type = "audio/wav"

with tab_upload:
    uploaded = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a", "ogg", "flac", "webm"])
    if uploaded is not None:
        audio_bytes = uploaded.getvalue()
        audio_name = uploaded.name
        mime_type = uploaded.type or "audio/wav"
        st.audio(audio_bytes)

with tab_record:
    st.write("Record directly using your microphone (requires browser microphone permission).")
    try:
        from audio_recorder_streamlit import audio_recorder

        recorded = audio_recorder(text="Click to record", recording_color="#e53935", neutral_color="#1a73e8")
        if recorded is not None and len(recorded) > 0:
            audio_bytes = recorded
            audio_name = "microphone_recording.wav"
            mime_type = "audio/wav"
            st.audio(audio_bytes)
    except ImportError:
        st.warning(
            "`audio-recorder-streamlit` is not installed. Run `pip install audio-recorder-streamlit` "
            "or use the Upload tab instead."
        )

st.divider()

analyze_clicked = st.button("🧠 Run AI Analysis", type="primary", disabled=audio_bytes is None)

if analyze_clicked and audio_bytes:
    with st.spinner("Running YAMNet inference and extracting acoustic features... (first run loads the model and may take longer)"):
        try:
            result = analyze_audio(audio_bytes, audio_name, mime_type)
            st.session_state["last_result"] = result
        except ApiError as e:
            st.error(f"Analysis failed: {e}")
            result = None
else:
    result = st.session_state.get("last_result")

if result:
    st.success(f"Analysis complete — top prediction: **{result['top_prediction']['label']}** "
               f"({result['top_prediction']['confidence']*100:.1f}% confidence)")

    col_a, col_b = st.columns([2, 1])
    with col_a:
        if result.get("waveform_preview"):
            st.plotly_chart(charts.waveform_figure(result["waveform_preview"]), use_container_width=True)
        st.plotly_chart(charts.prediction_bar_chart(result["predictions"]), use_container_width=True)
    with col_b:
        st.plotly_chart(charts.confidence_gauge(result["top_prediction"]["confidence"]), use_container_width=True)
        loudness = result["loudness"]
        badge_color = {"quiet": "🟢", "moderate": "🟡", "loud": "🟠", "very_loud": "🔴"}.get(loudness["category"], "⚪")
        st.metric("Relative Loudness", f"{badge_color} {loudness['category'].replace('_', ' ').title()}", f"{loudness['dbfs']:.1f} dBFS")

    st.subheader("🌍 Environmental Interpretation")
    st.write(result["environmental_interpretation"])
    st.caption(loudness["explanation"])

    # --- Human-in-the-Loop: User Confirmation (UI-only, session-scoped) ---
    st.subheader("👤 User Confirmation")

    ai_prediction = None
    top_prediction = result.get("top_prediction") if isinstance(result, dict) else None
    if top_prediction:
        ai_prediction = top_prediction.get("label")

    st.write("**AI Prediction:**")
    st.write(ai_prediction if ai_prediction else "_No AI prediction available_")

    category_options = [
        "Construction",
        "Traffic",
        "Music",
        "Industrial",
        "Dog",
        "Conversation",
        "Other",
    ]

    analysis_id = result.get("id") if isinstance(result, dict) else None
    confirmation_key = f"user_confirmation_choice_{analysis_id}" if analysis_id else "user_confirmation_choice"
    category_key = f"user_confirmation_category_{analysis_id}" if analysis_id else "user_confirmation_category"

    confirmation_choice = st.radio(
        "Please confirm or correct the AI prediction:",
        options=["Confirm AI prediction", "Change category manually"],
        key=confirmation_key,
    )

    selected_category = ai_prediction
    user_confirmed_ai = confirmation_choice == "Confirm AI prediction"

    if confirmation_choice == "Change category manually":
        default_index = 0
        if ai_prediction and ai_prediction in category_options:
            default_index = category_options.index(ai_prediction)

        selected_category = st.selectbox(
            "Select the correct category:",
            options=category_options,
            index=default_index,
            key=category_key,
        )

        st.write(f"User selected:\n{selected_category}")
    else:
        st.write("User confirmed AI prediction.")
    # --- End Human-in-the-Loop section ---

    st.subheader("✅ Recommendations")
    for rec in result["recommendations"]:
        st.markdown(f"- {rec}")

    with st.expander("🔬 Full acoustic features (DSP)"):
        st.json(result["features"])

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        try:
            # user_confirmed_ai / selected_category are passed only for this
            # PDF render — never stored in the database or analysis history.
            pdf_bytes = download_pdf_report(
                result["id"],
                user_confirmed_ai=user_confirmed_ai,
                user_selected_category=None if user_confirmed_ai else selected_category,
            )
            st.download_button("📄 Download PDF Report", data=pdf_bytes, file_name=f"noiseguard_report_{result['id'][:8]}.pdf", mime="application/pdf")
        except ApiError as e:
            st.warning(f"PDF report not yet available: {e}")
    with col2:
        st.caption(f"Analysis ID: `{result['id']}`")
