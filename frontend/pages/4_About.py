"""
About page — research background: problem statement, target users,
Kazakhstan use cases, CustDev summary, and why AI improves the solution.
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="About — NoiseGuard AI", page_icon="ℹ️", layout="wide")
st.title("ℹ️ About NoiseGuard AI")

st.header("Problem statement")
st.write(
    "Environmental noise complaints in Kazakhstani cities — from construction sites, traffic, "
    "industrial equipment, and loud venues — are almost always reported subjectively ('it's too "
    "loud', 'it happens all the time'). Housing services (КСК/ОСИ), akimats and businesses have no "
    "easy way to verify or prioritize these complaints without dispatching an inspector with "
    "certified equipment, which is slow and resource-limited. Citizens, in turn, lack any tool to "
    "document a disturbance objectively before escalating it."
)

st.header("Target users")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**🏠 Residents**")
    st.write("Want to document a recurring disturbance (construction, neighbors, traffic) before filing a formal complaint.")
with col2:
    st.markdown("**🏢 Housing services (КСК/ОСИ)**")
    st.write("Need a quick, structured way to triage incoming noise complaints before escalating to authorities.")
with col3:
    st.markdown("**🏛️ Municipal / akimat staff**")
    st.write("Handle high complaint volumes and benefit from pre-structured, categorized reports.")

st.header("Realistic Kazakhstan use cases")
st.markdown(
    """
- A resident of a Almaty apartment complex records recurring late-night construction noise to attach to a complaint filed with the local akimat.
- A КСК administrator uses the platform to triage which resident complaints (traffic vs. neighbor noise vs. HVAC equipment) require escalation vs. direct mediation.
- A small business owner disputes a noise complaint from a neighboring tenant by producing an independent recording and AI-generated report of their own equipment's actual sound profile.
- A citizen documents a pattern of nighttime vehicle idling near a residential courtyard across multiple recordings to support a request for signage or enforcement.
"""
)

st.header("CustDev interview summary")
st.write(
    "We interviewed residents who regularly experience environmental noise, including "
    "traffic, construction, neighbors, and nearby entertainment venues. Across the interviews, "
    "three recurring problems emerged:\n\n"

    "• People often feel uncomfortable submitting a noise complaint because they believe "
    "their evidence is not convincing enough.\n\n"

    "• Most participants did not know how to properly document noise without professional "
    "equipment, and a simple phone recording was perceived as insufficient.\n\n"

    "• Participants said they would be much more confident submitting a complaint if the "
    "recording were automatically transformed into a structured report explaining the likely "
    "noise source and supporting acoustic information."
)

    

st.header("Why existing solutions are insufficient")
st.markdown(
    """
    - *Smartphone recordings* provide raw audio but no structured explanation or evidence.
- *Sound level meter apps* show loudness only and cannot identify the likely sound source.
- *Professional acoustic inspections* are accurate but expensive, slow, and inaccessible for everyday complaints.
- *Text-only complaints* are often considered subjective and difficult for authorities or housing services to evaluate.
""")


st.header("Why AI improves the solution")
st.markdown(
    """
- **Objective classification**: instead of a citizen describing "a loud banging sound," YAMNet identifies the acoustic pattern (e.g. matching construction/jackhammer sounds) from the audio itself.
- **Consistent structure**: every analysis produces the same structured output (classification, DSP features, relative intensity, interpretation, recommendations) — making complaints comparable and easier to triage at scale.
- **Explainability over a black-box number**: the report shows *which* class was detected and *why* (feature values), rather than a single unexplained decibel reading.
- **Local inference**: running YAMNet locally (no cloud dependency) makes the tool usable in low-connectivity contexts and avoids sending citizens' audio to third-party services.
"""
)

st.header("Key limitation (please read)")
st.warning(
    "NoiseGuard AI provides a **relative** acoustic intensity estimate derived from an uncalibrated "
    "recording device. It explicitly does **not** claim to produce certified decibel (dB SPL) "
    "measurements, and cannot replace an accredited sound-level meter for legal or regulatory "
    "proceedings. Its purpose is to give citizens and staff an objective *starting point* — "
    "classification and documentation — not a legally certified measurement."
)

st.header("Technology stack")
st.markdown(
    """
| Layer | Technology |
|---|---|
| Backend API | FastAPI |
| Frontend | Streamlit |
| AI Model | TensorFlow + TensorFlow Hub (YAMNet) |
| DSP / Feature extraction | Librosa |
| Database | SQLite + SQLAlchemy |
| Visualization | Plotly |
| Reports | ReportLab (PDF), pandas (CSV) |
"""
)
