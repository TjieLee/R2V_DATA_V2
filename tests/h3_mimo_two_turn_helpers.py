"""Split existing frozen annotation fixtures into the three model response shapes."""

import json

from r2v_data_v2.h3.audio_backends import AudioFileProbe


class SnippetAudioBackend:
    """Deterministic CPU stub for backend fixtures whose media contain placeholder bytes."""

    def __init__(self):
        self.extractions = []

    def probe_audio_file(self, path):
        return AudioFileProbe(sample_rate_hz=32000, channels=2, frame_count=3200000,
                              duration_seconds=100.0, format_name="flac")

    def extract_voice_reference(self, **kwargs):
        self.extractions.append(kwargs)
        data = {
            key: kwargs[key] for key in ("source_start_sample", "source_end_sample", "sample_rate_hz")
        }
        kwargs["destination"].write_bytes(json.dumps(data, sort_keys=True).encode())
        return kwargs["destination"]


def split_annotation(raw):
    payload = json.loads(raw)
    semantics = payload["h3_semantics"]
    visual = {
        "segment_views": payload["visual_observation"]["segment_views"],
        **{key: semantics[key] for key in (
            "subject_definitions", "visual_retention_analysis", "style_opening",
        )},
        "shot1_visual_description": payload["visual_observation"]["visual_blocks"][0]["text"],
    }
    speech = {
        **({"summary": semantics["summary"]} if "summary" in semantics else {}),
        **{key: payload[key] for key in ("audio_observation", "av_grounding", "warnings")},
        "shot1_caption": semantics.get("shot1_caption", ""),
    }
    finalized = {
        "speaker_voice_profiles": payload["audio_observation"]["speaker_voice_profiles"],
        **{key: semantics[key] for key in (
            "overall_soundscape", "non_diegetic_music",
        ) if key in semantics},
    }
    speech["audio_observation"] = {**speech["audio_observation"], "speaker_voice_profiles": []}
    return json.dumps(visual), json.dumps(speech), json.dumps(finalized)


def assembly_raw(raw):
    try:
        return split_annotation(raw)[2]
    except (ValueError, KeyError, TypeError):
        return raw
