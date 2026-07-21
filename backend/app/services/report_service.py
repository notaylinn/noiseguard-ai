"""
PDF report generation using ReportLab.

Produces a professional, self-contained PDF summarizing one
`AnalysisResult` — suitable for attaching to a complaint filed with a
municipality, housing service or business. Kept isolated from the API
layer so report formatting logic can be unit-tested independently.
"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from backend.app.core.config import get_settings
from backend.app.core.logging_config import get_logger
from backend.app.domain.entities import AnalysisResult

logger = get_logger(__name__)


class PdfReportService:
    """Generates a PDF report for a single analysis result."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._styles = getSampleStyleSheet()
        self._styles.add(
            ParagraphStyle(name="NGTitle", fontSize=20, leading=24, spaceAfter=6, textColor=colors.HexColor("#1a3c5e"))
        )
        self._styles.add(
            ParagraphStyle(name="NGSection", fontSize=13, leading=16, spaceBefore=14, spaceAfter=6, textColor=colors.HexColor("#1a3c5e"))
        )
        self._styles.add(ParagraphStyle(name="NGBody", fontSize=10, leading=14))
        self._styles.add(ParagraphStyle(name="NGDisclaimer", fontSize=8.5, leading=11, textColor=colors.HexColor("#7a1f1f")))

    def generate(
        self,
        result: AnalysisResult,
        output_path: str | Path | None = None,
        user_confirmed_ai: bool | None = None,
        user_selected_category: str | None = None,
    ) -> Path:
        output_path = Path(output_path) if output_path else self._settings.report_storage_dir / f"report_{result.id}.pdf"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        doc = SimpleDocTemplate(str(output_path), pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm)
        story = []
        s = self._styles

        story.append(Paragraph("NoiseGuard AI — Environmental Noise Analysis Report", s["NGTitle"]))
        story.append(Paragraph(f"Generated: {result.created_at.strftime('%Y-%m-%d %H:%M UTC')} &nbsp;|&nbsp; Analysis ID: {result.id}", s["NGBody"]))
        story.append(Spacer(1, 10))

        story.append(Paragraph("1. Summary", s["NGSection"]))
        summary_data = [
            ["Source file", result.filename],
            ["Duration", f"{result.features.duration_sec:.2f} s"],
            ["Top classification", f"{result.top_prediction.label} ({result.top_prediction.confidence*100:.1f}% confidence)"],
            ["Relative loudness", f"{result.loudness.category.value.replace('_', ' ').title()} ({result.loudness.dbfs:.1f} dBFS)"],
        ]
        story.append(self._table(summary_data))

        story.append(Paragraph("2. AI Classification (YAMNet)", s["NGSection"]))
        pred_data = [["Class", "Confidence"]] + [[p.label, f"{p.confidence*100:.1f}%"] for p in result.predictions]
        story.append(self._table(pred_data, header=True))

        story.append(Paragraph("3. Acoustic Features (DSP)", s["NGSection"]))
        f = result.features
        feat_data = [
            ["RMS (mean / std)", f"{f.rms_mean:.4f} / {f.rms_std:.4f}"],
            ["Zero crossing rate", f"{f.zero_crossing_rate:.4f}"],
            ["Spectral centroid", f"{f.spectral_centroid:.1f} Hz"],
            ["Spectral bandwidth", f"{f.spectral_bandwidth:.1f} Hz"],
            ["Spectral roll-off", f"{f.spectral_rolloff:.1f} Hz"],
            ["Mel-spectrogram mean", f"{f.mel_spectrogram_db_mean:.1f} dB"],
            ["Estimated relative level", f"{f.estimated_dbfs:.1f} dBFS"],
        ]
        story.append(self._table(feat_data))

        story.append(Paragraph("4. Environmental Interpretation", s["NGSection"]))
        story.append(Paragraph(result.environmental_interpretation, s["NGBody"]))

        if user_confirmed_ai is not None:
            story.append(Paragraph("5. User Confirmation", s["NGSection"]))
            user_confirmation_data = [["AI Prediction", result.top_prediction.label]]
            if user_confirmed_ai:
                user_confirmation_data.append(["User Confirmation", "Confirmed AI prediction"])
            else:
                user_confirmation_data.append(["User Confirmation", user_selected_category or "—"])
            story.append(self._table(user_confirmation_data))

        story.append(Paragraph("6. Recommendations", s["NGSection"]))
        for rec in result.recommendations:
            story.append(Paragraph(f"• {rec}", s["NGBody"]))

        story.append(Paragraph("7. Important Limitation", s["NGSection"]))
        story.append(
            Paragraph(
                "This report is generated from an uncalibrated smartphone/microphone recording. "
                "The loudness value is a RELATIVE estimate derived from digital signal amplitude "
                "(dBFS), NOT a certified decibel (dB SPL) measurement. It must not be treated as "
                "legally certified acoustic evidence. For official measurements, contact an "
                "accredited acoustic measurement authority.",
                s["NGDisclaimer"],
            )
        )

        doc.build(story)
        logger.info("Generated PDF report at %s", output_path)
        return output_path

    def _table(self, data: list[list[str]], header: bool = False) -> Table:
        table = Table(data, colWidths=[6 * cm, 9 * cm] if not header else None, hAlign="LEFT")
        style = [
            ("FONTSIZE", (0, 0), (-1, -1), 9.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ]
        if header:
            style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3c5e")))
            style.append(("TEXTCOLOR", (0, 0), (-1, 0), colors.white))
            style.append(("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"))
        else:
            style.append(("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"))
        table.setStyle(TableStyle(style))
        return table
