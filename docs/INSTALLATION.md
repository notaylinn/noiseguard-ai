# Installation Guide

## Prerequisites

- Python 3.10 – 3.11 (TensorFlow 2.16 wheels; Python 3.12 may not have
  prebuilt TensorFlow wheels on all platforms — 3.10/3.11 is safest)
- ~2GB free disk space (TensorFlow + YAMNet model cache)
- Internet access on first run (YAMNet is downloaded from TensorFlow Hub
  and cached locally under `.model_cache/`; every run after that is fully offline)
- (Optional) a working microphone + browser permissions for in-app recording

## 1. Clone and create a virtual environment

```bash
git clone <repo-url> noiseguard-ai
cd noiseguard-ai
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

## 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 3. (Optional) generate placeholder sample audio

The repo ships with a handful of synthetic WAV tones under
`data/sample_audio/` so you can try the app immediately. To regenerate
them (stdlib only, no dependencies required):

```bash
python scripts/generate_sample_audio.py
```

## 4. Start the backend (FastAPI)

```bash
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

- The database (`storage/noiseguard.db`) is created automatically on startup.
- The YAMNet model loads lazily on the **first** `/analysis` request —
  this first call will take longer (downloading + loading the model);
  subsequent calls are fast.
- Swagger docs: http://localhost:8000/docs

## 5. Start the frontend (Streamlit), in a second terminal

```bash
source .venv/bin/activate
streamlit run frontend/streamlit_app.py
```

Open the URL Streamlit prints (default http://localhost:8501).

If your backend runs on a non-default host/port, set:
```bash
export NOISEGUARD_API_URL="http://localhost:8000/api/v1"
```

## 6. Run tests

```bash
pytest tests/ -v
```

`tests/test_feature_extraction.py` and `tests/test_repository.py` run
fast with no ML model download. `tests/test_api.py` substitutes a
lightweight stub classifier via FastAPI dependency overrides so the
full request/response/DB flow is tested without requiring TensorFlow
to download YAMNet during CI — production code always uses the real
`YamnetClassifier` (see `backend/app/api/dependencies.py`).

## Troubleshooting

| Issue | Fix |
|---|---|
| `tensorflow_hub` download fails / times out | Check internet access; the model is cached under `.model_cache/` after the first successful download |
| `ffmpeg`/audio decode errors for mp3/m4a | Install `ffmpeg` on your system (`apt install ffmpeg` / `brew install ffmpeg`) — librosa uses it via `audioread` for non-WAV formats |
| Streamlit shows "Backend unreachable" | Confirm `uvicorn` is running and `NOISEGUARD_API_URL` matches its address |
| Microphone recording tab shows a warning | Run `pip install audio-recorder-streamlit`, or just use the Upload tab |
