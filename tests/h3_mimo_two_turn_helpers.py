"""Split existing frozen annotation fixtures into the three model response shapes."""

import json


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
        **{key: payload[key] for key in ("audio_observation", "av_grounding", "warnings")},
        "shot1_caption": semantics.get("shot1_caption", ""),
    }
    finalized = {
        **{key: semantics[key] for key in (
            "summary", "shot1_caption", "overall_soundscape", "non_diegetic_music",
        ) if key in semantics},
    }
    return json.dumps(visual), json.dumps(speech), json.dumps(finalized)


def assembly_raw(raw):
    try:
        return split_annotation(raw)[2]
    except (ValueError, KeyError, TypeError):
        return raw
