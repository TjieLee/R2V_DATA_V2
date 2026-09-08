"""Split existing frozen annotation fixtures into the two model response shapes."""

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
    assembly = {
        **{key: payload[key] for key in ("audio_observation", "av_grounding", "warnings")},
        **{key: semantics[key] for key in (
            "summary", "shot1_caption", "overall_soundscape", "non_diegetic_music",
        ) if key in semantics},
    }
    return json.dumps(visual), json.dumps(assembly)


def assembly_raw(raw):
    try:
        return split_annotation(raw)[1]
    except (ValueError, KeyError, TypeError):
        return raw
