"""
Thin HTTP client wrapping the NoiseGuard AI REST API.

This is the ONLY module in the Streamlit frontend that knows about
HTTP/requests details. Pages call these functions and work with plain
dicts, keeping page code focused on presentation, not API plumbing —
i.e. business/integration logic stays out of the UI layer.
"""
from __future__ import annotations

import os

import requests

API_BASE_URL = os.environ.get("NOISEGUARD_API_URL", "http://localhost:8000/api/v1")
DEFAULT_TIMEOUT = 60


class ApiError(Exception):
    pass


def analyze_audio(file_bytes: bytes, filename: str, mime_type: str = "audio/wav") -> dict:
    files = {"file": (filename, file_bytes, mime_type)}
    resp = requests.post(f"{API_BASE_URL}/analysis", files=files, timeout=DEFAULT_TIMEOUT)
    if resp.status_code != 200:
        raise ApiError(_extract_error(resp))
    return resp.json()


def get_history(limit: int = 50, offset: int = 0) -> list[dict]:
    resp = requests.get(f"{API_BASE_URL}/history", params={"limit": limit, "offset": offset}, timeout=DEFAULT_TIMEOUT)
    if resp.status_code != 200:
        raise ApiError(_extract_error(resp))
    return resp.json()


def get_analysis(analysis_id: str) -> dict:
    resp = requests.get(f"{API_BASE_URL}/history/{analysis_id}", timeout=DEFAULT_TIMEOUT)
    if resp.status_code != 200:
        raise ApiError(_extract_error(resp))
    return resp.json()


def delete_analysis(analysis_id: str) -> dict:
    resp = requests.delete(f"{API_BASE_URL}/history/{analysis_id}", timeout=DEFAULT_TIMEOUT)
    if resp.status_code != 200:
        raise ApiError(_extract_error(resp))
    return resp.json()


def get_statistics() -> dict:
    resp = requests.get(f"{API_BASE_URL}/statistics", timeout=DEFAULT_TIMEOUT)
    if resp.status_code != 200:
        raise ApiError(_extract_error(resp))
    return resp.json()


def get_pdf_report_url(analysis_id: str) -> str:
    return f"{API_BASE_URL}/reports/{analysis_id}/pdf"


def download_pdf_report(
    analysis_id: str,
    user_confirmed_ai: bool | None = None,
    user_selected_category: str | None = None,
) -> bytes:
    # user_confirmed_ai / user_selected_category are sent as query params
    # only for this request. They are not stored anywhere and are not
    # part of the analysis result object.
    params = {}
    if user_confirmed_ai is not None:
        params["user_confirmed_ai"] = user_confirmed_ai
        if not user_confirmed_ai and user_selected_category:
            params["user_selected_category"] = user_selected_category

    resp = requests.get(get_pdf_report_url(analysis_id), params=params, timeout=DEFAULT_TIMEOUT)
    if resp.status_code != 200:
        raise ApiError(_extract_error(resp))
    return resp.content


def download_csv_export() -> bytes:
    resp = requests.get(f"{API_BASE_URL}/export/csv", timeout=DEFAULT_TIMEOUT)
    if resp.status_code != 200:
        raise ApiError(_extract_error(resp))
    return resp.content


def check_health() -> dict | None:
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return resp.json() if resp.status_code == 200 else None
    except requests.RequestException:
        return None


def _extract_error(resp: requests.Response) -> str:
    try:
        return resp.json().get("detail", resp.text)
    except Exception:
        return resp.text
