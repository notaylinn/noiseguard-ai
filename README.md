# 🔊 NoiseGuard AI

**AI-powered environmental noise assessment platform for Kazakhstan.**

Built for the **Tech Vision** hackathon. NoiseGuard AI lets citizens
record or upload an environmental sound, classifies it with a real
local machine-learning model (Google's **YAMNet** via TensorFlow Hub),
extracts objective acoustic (DSP) features with **librosa**, and
generates a structured, professional report — turning "it's really
loud outside" into an objective, documented, actionable report for a
housing service, akimat or business.

> ⚠️ **Honesty first:** this app estimates *relative* acoustic
> intensity from an uncalibrated recording device. It does **not**
> produce certified decibel (dB SPL) measurements and cannot replace
> an accredited sound-level meter for legal/regulatory purposes. See
> [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and the in-app
> disclaimers for details.

## Why

Residents of Kazakhstani cities regularly deal with disruptive noise —
construction, traffic, industrial equipment, loud venues — but have no
good way to document it objectively before filing a complaint. See
[the About page](frontend/pages/4_About.py) for the full problem
statement, target users, realistic use cases, and CustDev summary.

## Features

- 🎙️ Upload or record audio directly in the browser
- 🧠 Real local YAMNet inference (521 AudioSet sound classes)
- 📊 Librosa DSP feature extraction: RMS, MFCC, mel-spectrogram,
  spectral centroid/bandwidth/roll-off, zero-crossing rate, duration
- 📈 Waveform, spectrogram, prediction and confidence charts (Plotly)
- 🔉 Relative loudness estimate with explicit calibration caveat
- 🌍 Plain-language environmental interpretation + actionable recommendations
- 🗂️ Analysis history with per-record detail, delete, and PDF download
- 📊 Aggregate statistics dashboard
- 📄 Professional PDF report generation (ReportLab)
- 📤 CSV export of full history (pandas)
- ⚙️ Settings / backend health page

## Tech stack

Python · FastAPI · Streamlit · TensorFlow · TensorFlow Hub (YAMNet) ·
Librosa · SQLite · SQLAlchemy · Plotly · ReportLab · pandas

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Terminal 1
uvicorn backend.app.main:app --reload

# Terminal 2
streamlit run frontend/streamlit_app.py
```

Full setup, troubleshooting, and platform notes: [docs/INSTALLATION.md](docs/INSTALLATION.md).

## Repository structure

```
noiseguard-ai/
├── backend/app/
│   ├── main.py                  # FastAPI app assembly
│   ├── api/                     # routes, schemas, DI wiring
│   ├── core/                    # config, logging
│   ├── domain/                  # entities + interfaces (framework-agnostic)
│   ├── services/                # business logic (analysis, report, export)
│   ├── ml/                      # YAMNet classifier + librosa feature extractor
│   ├── db/                      # SQLAlchemy models + repository pattern
│   └── utils/                   # audio I/O helpers
├── frontend/
│   ├── streamlit_app.py         # landing page
│   ├── pages/                   # Upload&Record, History, Statistics, About, Settings
│   └── components/              # api_client, chart builders
├── tests/                       # pytest suite (unit + API)
├── data/sample_audio/           # synthetic placeholder audio for quick testing
├── reports/examples/            # example generated PDF report
├── docs/                        # architecture, API, installation docs
├── scripts/generate_sample_audio.py
└── requirements.txt
```

Full architecture rationale (Clean Architecture, SOLID, DI, repository
pattern): [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
API reference: [docs/API.md](docs/API.md).

## Testing

```bash
pytest tests/ -v
```

See [docs/INSTALLATION.md#6-run-tests](docs/INSTALLATION.md#6-run-tests)
for what is and isn't mocked and why.

## Known limitations (by design, stated up front)

- Loudness is a **relative**, uncalibrated estimate (dBFS from digital
  signal amplitude) — not certified dB SPL. This is stated in the UI,
  the PDF report, and the API response itself.
- YAMNet's 521 AudioSet classes are general-purpose, not
  Kazakhstan-specific; the app maps relevant classes to noise-complaint
  categories (`backend/app/ml/yamnet_labels.py`) but will label
  out-of-scope sounds as "other_environmental" rather than guessing.
- Sample audio in `data/sample_audio/` is synthetically generated
  (sine tones / noise) for immediate demo purposes — real environmental
  recordings will produce far more meaningful classifications.

## License

MIT — see [LICENSE](LICENSE).
