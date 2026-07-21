"""
Reusable Plotly chart builders for the Streamlit frontend.

Isolating chart construction here keeps page files focused on layout
and flow, and makes every chart independently reusable (e.g. the
confidence bar chart is used on both the Analysis and History pages).
"""
from __future__ import annotations

import plotly.graph_objects as go


def waveform_figure(samples: list[float], sample_rate: int = 16000, title: str = "Waveform") -> go.Figure:
    time_axis = [i / sample_rate * (len(samples) / max(len(samples), 1)) for i in range(len(samples))]
    fig = go.Figure(go.Scatter(y=samples, mode="lines", line=dict(color="#1a73e8", width=1)))
    fig.update_layout(title=title, xaxis_title="Sample index", yaxis_title="Amplitude", height=280, margin=dict(l=40, r=20, t=40, b=40))
    return fig


def spectrogram_figure(mel_spec_db, title: str = "Mel Spectrogram") -> go.Figure:
    fig = go.Figure(data=go.Heatmap(z=mel_spec_db, colorscale="Viridis"))
    fig.update_layout(title=title, xaxis_title="Time frame", yaxis_title="Mel band", height=320, margin=dict(l=40, r=20, t=40, b=40))
    return fig


def prediction_bar_chart(predictions: list[dict], title: str = "Top Predictions") -> go.Figure:
    labels = [p["label"] for p in predictions][::-1]
    values = [p["confidence"] * 100 for p in predictions][::-1]
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color="#1a73e8"))
    fig.update_layout(title=title, xaxis_title="Confidence (%)", height=max(250, 40 * len(labels)), margin=dict(l=150, r=20, t=40, b=40))
    return fig


def confidence_gauge(confidence: float, title: str = "Top Prediction Confidence") -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            title={"text": title},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1a73e8"},
                "steps": [
                    {"range": [0, 40], "color": "#fde2e1"},
                    {"range": [40, 70], "color": "#fff3cd"},
                    {"range": [70, 100], "color": "#d4edda"},
                ],
            },
        )
    )
    fig.update_layout(height=280, margin=dict(l=30, r=30, t=50, b=10))
    return fig


def category_distribution_pie(distribution: dict[str, int], title: str = "Detected Category Distribution") -> go.Figure:
    labels = [k.replace("_", " ").title() for k in distribution.keys()]
    values = list(distribution.values())
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.4))
    fig.update_layout(title=title, height=380, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def loudness_distribution_bar(distribution: dict[str, int], title: str = "Loudness Category Distribution") -> go.Figure:
    order = ["quiet", "moderate", "loud", "very_loud"]
    labels = [k.replace("_", " ").title() for k in order if k in distribution]
    values = [distribution[k] for k in order if k in distribution]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=["#4caf50", "#ffc107", "#ff7043", "#e53935"]))
    fig.update_layout(title=title, yaxis_title="Number of analyses", height=350, margin=dict(l=40, r=20, t=40, b=40))
    return fig


def history_timeline(records: list[dict], title: str = "Analyses Over Time") -> go.Figure:
    dates = [r["created_at"][:10] for r in records]
    counts: dict[str, int] = {}
    for d in dates:
        counts[d] = counts.get(d, 0) + 1
    sorted_dates = sorted(counts.keys())
    fig = go.Figure(go.Scatter(x=sorted_dates, y=[counts[d] for d in sorted_dates], mode="lines+markers", line=dict(color="#1a73e8")))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Analyses", height=320, margin=dict(l=40, r=20, t=40, b=40))
    return fig
