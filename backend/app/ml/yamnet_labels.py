"""
Utilities for loading and mapping YAMNet's 521 AudioSet class names to
human-readable environmental categories relevant to noise complaints.

YAMNet ships its own class map CSV inside the TF-Hub asset
(`model.class_map_path()`), which is the authoritative source used at
inference time. The static groupings below are only used to translate
a raw AudioSet label (e.g. "Jackhammer") into the kind of plain-language
environmental interpretation a citizen or municipal officer can read.
"""
from __future__ import annotations

# Maps AudioSet leaf labels (lowercased) to a broader noise-complaint
# category. This is deliberately partial: any label not found here
# falls back to a generic "Detected environmental sound" interpretation
# so the app never fabricates a category YAMNet did not predict.
NOISE_CATEGORY_MAP: dict[str, str] = {
    # Traffic / transport
    "car": "road_traffic", "car passing by": "road_traffic", "truck": "road_traffic",
    "bus": "road_traffic", "motorcycle": "road_traffic", "traffic noise, roadway noise": "road_traffic",
    "vehicle horn, car horn, honking": "road_traffic", "car alarm": "road_traffic",
    "railroad car, train wagon": "rail_traffic", "train": "rail_traffic", "train whistle": "rail_traffic",
    "aircraft": "air_traffic", "fixed-wing aircraft, airplane": "air_traffic", "helicopter": "air_traffic",

    # Construction
    "jackhammer": "construction", "drill": "construction", "power tool": "construction",
    "hammer": "construction", "sawing": "construction", "construction": "construction",

    # Industrial / mechanical
    "engine": "industrial_mechanical", "engine starting": "industrial_mechanical",
    "idling": "industrial_mechanical", "machine gun": "industrial_mechanical",
    "generator": "industrial_mechanical", "vacuum cleaner": "industrial_mechanical",
    "mechanical fan": "industrial_mechanical", "air conditioning": "industrial_mechanical",

    # Human / social
    "speech": "human_activity", "shout": "human_activity", "yell": "human_activity",
    "conversation": "human_activity", "crowd": "human_activity", "children playing": "human_activity",
    "laughter": "human_activity", "party": "human_activity",

    # Music / entertainment
    "music": "music_entertainment", "musical instrument": "music_entertainment",
    "singing": "music_entertainment", "bass drum": "music_entertainment",

    # Animals
    "dog": "animal", "bark": "animal", "cat": "animal", "bird": "animal", "bird vocalization, bird call, bird song": "animal",

    # Alarms / sirens
    "siren": "alarm_siren", "civil defense siren": "alarm_siren", "alarm": "alarm_siren",
    "fire alarm": "alarm_siren", "smoke detector, smoke alarm": "alarm_siren", "emergency vehicle": "alarm_siren",

    # Nature / ambient
    "wind": "natural_ambient", "rain": "natural_ambient", "thunder": "natural_ambient",
    "water": "natural_ambient", "silence": "quiet_ambient",
}


def map_label_to_category(label: str) -> str:
    return NOISE_CATEGORY_MAP.get(label.strip().lower(), "other_environmental")
