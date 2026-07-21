.PHONY: install backend frontend test sample-audio

install:
	pip install -r requirements.txt

sample-audio:
	python scripts/generate_sample_audio.py

backend:
	uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000

frontend:
	streamlit run frontend/streamlit_app.py

test:
	pytest tests/ -v
