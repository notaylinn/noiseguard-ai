"""
Core business logic: orchestrates feature extraction, ML classification,
loudness estimation, environmental interpretation and recommendation
generation into a single `AnalysisResult`.

This is where "AI integration" actually happens at the business-logic
level — it depends only on the `SoundClassifierPort` and
`FeatureExtractorPort` interfaces (constructor-injected), never on
TensorFlow directly, and never on the database. This is textbook
Dependency Inversion + Single Responsibility.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from backend.app.core.config import get_settings
from backend.app.core.logging_config import get_logger
from backend.app.domain.entities import (
    AnalysisResult,
    LoudnessCategory,
    LoudnessEstimate,
)
from backend.app.domain.interfaces import (
    AnalysisRepositoryPort,
    FeatureExtractorPort,
    SoundClassifierPort,
)
from backend.app.ml.yamnet_labels import map_label_to_category

logger = get_logger(__name__)

_CATEGORY_INTERPRETATIONS: dict[str, str] = {
    "road_traffic": (
        "The dominant sound matches road traffic patterns (engines, tires, horns). "
        "This is consistent with noise disturbances near streets, highways or parking areas."
    ),
    "rail_traffic": "The recording matches rail-related sounds (trains, rail cars), typical of noise near railway lines.",
    "air_traffic": "The recording matches aircraft-related sounds, typical of noise near flight paths or airports.",
    "construction": (
        "The dominant sound matches construction activity (drilling, hammering, power tools). "
        "This pattern is common in disputes over construction-site noise regulations."
    ),
    "industrial_mechanical": "The recording matches mechanical/industrial equipment noise (engines, generators, ventilation systems).",
    "human_activity": "The dominant sound matches human voices or crowd activity rather than mechanical noise sources.",
    "music_entertainment": "The recording matches music or amplified entertainment sound, common in disputes over venue or neighbor noise.",
    "animal": "The dominant sound matches animal vocalizations (e.g. barking), common in residential noise complaints.",
    "alarm_siren": "The recording matches alarm or siren sounds, which may indicate an emergency or a malfunctioning device.",
    "natural_ambient": "The dominant sound matches natural ambient noise (wind, rain, water) rather than a human-made source.",
    "quiet_ambient": "The recording is dominated by low-level ambient sound with no strong single noise source detected.",
    "other_environmental": "A specific environmental sound was detected, but it does not map to a predefined complaint category.",
}

_RECOMMENDATIONS: dict[str, list[str]] = {
    "road_traffic": [
        "Document the time of day and frequency of the traffic noise over several days.",
        "Check whether the road falls under municipal or highway authority jurisdiction before filing a complaint.",
        "Consider requesting a formal noise barrier or speed-limit review from the local akimat.",
    ],
    "construction": [
        "Verify the construction site's permitted working hours under local regulations.",
        "Record multiple samples across different days to demonstrate a consistent pattern.",
        "File a complaint with the municipal construction supervision authority, attaching this report.",
    ],
    "industrial_mechanical": [
        "Identify the nearby facility or equipment that may be the source.",
        "Request an official noise-level inspection from sanitary-epidemiological services.",
        "Keep a log of recurrence times to support a formal complaint.",
    ],
    "human_activity": [
        "Consider direct, documented communication with the source (e.g. building management) before escalation.",
        "If recurring, keep a log of dates/times as supporting evidence.",
    ],
    "music_entertainment": [
        "Check local quiet-hour regulations applicable to residential buildings or venues.",
        "Contact building management or the venue's licensing authority if the pattern is recurring.",
    ],
    "alarm_siren": [
        "If this is a malfunctioning alarm, contact the property owner or building management immediately.",
        "If it persists, notify local emergency or municipal services.",
    ],
    "animal": [
        "Consider first speaking with the animal's owner if known.",
        "Recurring cases can be reported to local animal control or housing services.",
    ],
    "rail_traffic": ["Contact the railway operator's public complaints line with the recording and timestamps."],
    "air_traffic": ["Check whether the location falls under an airport noise-mitigation zone before filing a complaint."],
    "natural_ambient": ["This appears to be a natural sound source; formal noise complaints are unlikely to apply."],
    "quiet_ambient": ["No dominant disturbing source was detected in this sample."],
    "other_environmental": ["Consider recording a longer or clearer sample for a more specific classification."],
}


class AnalysisService:
    """Application service coordinating the full analysis pipeline."""

    def __init__(
        self,
        classifier: SoundClassifierPort,
        feature_extractor: FeatureExtractorPort,
        repository: AnalysisRepositoryPort | None = None,
    ) -> None:
        self._classifier = classifier
        self._extractor = feature_extractor
        self._repository = repository
        self._settings = get_settings()

    def analyze(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        filename: str,
        persist: bool = True,
        audio_path: str | None = None,
    ) -> AnalysisResult:
        if sample_rate != self._settings.sample_rate:
            raise ValueError(f"Expected {self._settings.sample_rate}Hz audio, got {sample_rate}Hz")
        if len(waveform) == 0:
            raise ValueError("Empty audio waveform - cannot analyze")

        logger.info("Starting analysis pipeline for '%s' (%.2fs)", filename, len(waveform) / sample_rate)

        features = self._extractor.extract(waveform, sample_rate)
        predictions = self._classifier.classify(waveform, sample_rate)
        top_prediction = predictions[0]

        loudness = self._estimate_loudness(features.estimated_dbfs)
        category = map_label_to_category(top_prediction.label)
        interpretation = self._build_interpretation(top_prediction.label, category, loudness)
        recommendations = self._build_recommendations(category, loudness)

        result = AnalysisResult(
            id=str(uuid.uuid4()),
            filename=filename,
            created_at=datetime.now(timezone.utc),
            predictions=predictions,
            top_prediction=top_prediction,
            features=features,
            loudness=loudness,
            environmental_interpretation=interpretation,
            recommendations=recommendations,
            audio_path=audio_path,
        )

        if persist and self._repository is not None:
            self._repository.save(result)

        logger.info("Analysis complete for '%s': top=%s (%.1f%%) loudness=%s",
                    filename, top_prediction.label, top_prediction.confidence * 100, loudness.category.value)
        return result

    def _estimate_loudness(self, dbfs: float) -> LoudnessEstimate:
        s = self._settings
        if dbfs <= s.loudness_quiet_max:
            category = LoudnessCategory.QUIET
        elif dbfs <= s.loudness_moderate_max:
            category = LoudnessCategory.MODERATE
        elif dbfs <= s.loudness_loud_max:
            category = LoudnessCategory.LOUD
        else:
            category = LoudnessCategory.VERY_LOUD

        explanation = (
            f"Estimated relative signal level: {dbfs:.1f} dBFS (digital full-scale reference). "
            "This is derived from the recording's RMS amplitude and reflects RELATIVE acoustic "
            "intensity only. It is NOT a calibrated dB SPL (sound pressure level) measurement — "
            "smartphone microphones have uncontrolled, device-specific gain, distance to the "
            "source affects the reading, and this value cannot substitute for certified "
            "sound-level-meter equipment in a legal or regulatory context."
        )
        return LoudnessEstimate(dbfs=dbfs, category=category, explanation=explanation)

    @staticmethod
    def _build_interpretation(label: str, category: str, loudness: LoudnessEstimate) -> str:
        base = _CATEGORY_INTERPRETATIONS.get(category, _CATEGORY_INTERPRETATIONS["other_environmental"])
        intensity_note = {
            LoudnessCategory.QUIET: "The estimated relative intensity is low.",
            LoudnessCategory.MODERATE: "The estimated relative intensity is moderate.",
            LoudnessCategory.LOUD: "The estimated relative intensity is high.",
            LoudnessCategory.VERY_LOUD: "The estimated relative intensity is very high.",
        }[loudness.category]
        return f"Detected class: '{label}'. {base} {intensity_note}"

    @staticmethod
    def _build_recommendations(category: str, loudness: LoudnessEstimate) -> list[str]:
        recs = list(_RECOMMENDATIONS.get(category, _RECOMMENDATIONS["other_environmental"]))
        if loudness.category in (LoudnessCategory.LOUD, LoudnessCategory.VERY_LOUD):
            recs.append(
                "For a formally admissible measurement, request an inspection with a certified "
                "sound level meter from the relevant sanitary or municipal authority."
            )
        return recs
