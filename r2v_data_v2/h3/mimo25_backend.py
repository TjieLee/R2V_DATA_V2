from __future__ import annotations

import base64
import hashlib
import json
import math
import mimetypes
import random
import re
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol
from urllib.parse import quote, urlsplit

from openai import OpenAI
from pydantic import Field, StrictBool, StrictStr, model_validator

from r2v_data_v2.h3.schemas import SchemaModel
from r2v_data_v2.h3.speech_presentation import SpeechPresentation
from r2v_data_v2.structured_output import (
    ValidationIssue,
    normalize_structured_json_envelope,
    parse_structured_json_issues,
)

MIMO25_MODEL = "mimo-v2.5"
MIMO25_DEFAULT_BASE_URL = "https://api.xiaomimimo.com/v1"
MIMO25_PROMPT_VERSION = "h3_mimo25_unified_av_reconcile_v38"
MIMO25_POLICY_VERSION = "h3_mimo25_av_authority_contract_v17"
MIMO25_SCHEMA_VERSION = "r2v.h3.mimo25_av_annotation.20"
MIMO25_BACKEND_VERSION = "r2v.h3.mimo25_backend.46"
MIMO25_SPEAKER_MARKER_POLISH_PROMPT_VERSION = "h3_mimo25_speaker_marker_polish_v4"
MIMO25_ICL_VERSION = "h3_official_ref2va_detailed_shot1_v4"
MIMO25_MATERIALIZER_VERSION = "h3_mimo25_materializer_v23"
MIMO25_CANONICAL_ABSENT_SOUNDSCAPE = (
    "No distinct environmental, mechanical, physical, or non-verbal human "
    "sounds are clearly discernible."
)
DEFAULT_BASE64_LIMIT_BYTES = 50 * 1024 * 1024
MimoTransport = Literal["xiaomi", "sglang"]

_GROUP = re.compile(r"g([1-9]\d*)")
_PICTURE_OR_SUBJECT = re.compile(r"<(Picture|Subject) (\d+)>")
_PIPELINE_OWNED = re.compile(
    r"<Audio \d+>|<Video \d+>|\(S\d+\)|<d>|</d>|\[Shot \d+\]"
)
_KEYFRAME_ROLE = re.compile(
    r"<Picture\s+[1-9]\d*>[^.\n]{0,100}\b(first frame|last frame|keyframe)\b"
    r"|\b(first frame|last frame|keyframe)\b[^.\n]{0,100}<Picture\s+[1-9]\d*>",
    flags=re.IGNORECASE,
)
_SUBJECT_AUDIO_PROFILE = re.compile(
    r"\b(?:voice|vocal|pitch|timbre|cadence|articulation|accent|dialect)\b"
    r"|\bspeaking(?:[\s-]+)rate\b",
    flags=re.IGNORECASE,
)
_VOICE_IDENTITY_PROFILE = re.compile(
    r"\b(?:doctor|teacher|chef|officer|narrator|actor|actress)\b"
    r"|\b(?:Chinese|American|British|Japanese|Korean|Indian|French|German|Russian)\b"
    r"|\b(?:mother|father|sister|brother|husband|wife|manager|employee)\b"
    r"|\b(?:personality|character|temperament)\b"
    r"|\b(?:voice|speaker)\s+(?:of|named|identified\s+as)\b",
    flags=re.IGNORECASE,
)
_HUMAN_ATTRIBUTE_TYPES = frozenset(
    {"hair", "face", "glasses", "upper_clothing", "accessory"}
)
_STANDALONE_HUMAN_SUBJECT_LEAD = re.compile(
    r"^\s*(?:is\s+)?(?:an?\s+|the\s+)?(?:woman|man|person|girl|boy|child)"
    r"(?:\s+(?:with|wearing|who|standing|shown|depicted)\b|\s*[,.;:]|\s*$)",
    flags=re.IGNORECASE,
)
_SOUNDSCAPE_CONTAMINATION = re.compile(
    r"\b(?:BGM|background music|instrumental music|score|soundtrack|music|song|"
    r"melody|(?<!non-)(?<!non )musical|spoken dialogue|human speech|dialogue|"
    r"narration|narrator(?:\s+speaks?)?|"
    r"voice-over|lyrics|singing)\b",
    flags=re.IGNORECASE,
)
_AUDIO_LAYER_CONTENT_TOKEN = re.compile(r"[a-z0-9]+")
_AUDIO_LAYER_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "by",
        "for",
        "from",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }
)
_AUDIO_LAYER_MIN_CONTENT_TOKEN_COUNT = 5
_AUDIO_LAYER_MIN_CONTAINMENT_RATIO = 0.85
_SOUNDSCAPE_CLAUSE_BOUNDARY = re.compile(
    r"[,;:.!?\n]+|\b(?:but|however|while|whereas|yet)\b",
    flags=re.IGNORECASE,
)
_SOUNDSCAPE_LOCAL_NEGATOR = re.compile(
    r"\b(?:no|not|without|neither|nor|none|lack|lacks|lacked|lacking|absent|"
    r"undetectable|indiscernible)\b",
    flags=re.IGNORECASE,
)
_SOUNDSCAPE_NEGATION_SCOPE_BREAK = re.compile(
    r"\b(?:is|are|was|were|plays?|continues?|remains?|speaks?|sings?|sounds?|"
    r"hears?|heard)\b",
    flags=re.IGNORECASE,
)
_SOUNDSCAPE_POST_NEGATION = re.compile(
    r"^\s*(?:(?:is|are|was|were|seems?|remains?)\s+)?(?:not\s+"
    r"(?:audible|present|detectable|discernible)|absent|undetectable|indiscernible)\b",
    flags=re.IGNORECASE,
)
_SOUNDSCAPE_ABSENCE_CLAUSE_BOUNDARY = re.compile(
    r"[;.!?\n]+|\b(?:but|however|while|whereas|yet)\b",
    flags=re.IGNORECASE,
)
_EXPLICIT_SOUNDSCAPE_ABSENCE = re.compile(
    r"^\s*no\b(?P<body>.+)\b(?:is|are|was|were)\s+"
    r"(?:(?:clearly|currently|readily)\s+)?"
    r"(?:audible|heard|present|detectable|discernible|established)\s*$",
    flags=re.IGNORECASE,
)
_SOUNDSCAPE_POSITIVE_VERB = re.compile(
    r"\b(?:is|are|was|were|remains?|continues?|persists?|plays?)\b",
    flags=re.IGNORECASE,
)
_INTERNAL_ANNOTATION_SYNTAX = re.compile(
    r"(?<![A-Za-z0-9_])(?:ae|v|g|e)[1-9]\d*(?![A-Za-z0-9_])"
    r"|\bsegment[_-][A-Za-z0-9_-]+\b"
    r"|\b(?:visual_observation|audio_observation|av_grounding|h3_projection|"
    r"CO_ANALYSIS|VISUAL_DESCRIPTION|AUDIO_DESCRIPTION)\b",
    flags=re.IGNORECASE,
)
_TRUNCATED_FINISH_REASONS = {
    "length",
    "max_tokens",
    "max_completion_tokens",
    "token_limit",
}
_VISIBLE_PRESENTATION_CONTRADICTIONS = frozenset(
    {"offscreen_audio", "voice_over_context", "device_playback_context"}
)
_SOUNDSCAPE_EVENT_CATEGORIES = frozenset(
    {
        "physical",
        "environmental",
        "mechanical",
        "electronic",
        "human_non_speech",
        "other",
    }
)

VocalComposition = Literal[
    "single_speaker",
    "same_speaker_nonlexical",
    "secondary_non_speech_vocalization",
    "overlapping_secondary_speech",
    "sequential_multi_speaker_speech",
    "uncertain",
]
Resolution = Literal["resolved", "needs_acoustic_refinement", "uncertain"]
BindingStatus = Literal[
    "visible_entity",
    "offscreen",
    "no_reliable_entity",
    "uncertain",
]
EvidenceCode = Literal[
    "visible_lip_motion",
    "speaker_visible_mouth_occluded",
    "av_temporal_alignment",
    "voice_continuity",
    "speaker_turn_change",
    "offscreen_audio",
    "lr_asd_support",
    "lr_asd_conflict",
    "source_cluster_support",
    "source_cluster_conflict",
    "insufficient_evidence",
    "no_visible_lip_motion",
    "message_text_alignment",
    "voice_over_context",
    "device_playback_context",
]
AudioEvidenceCode = Literal[
    "voice_continuity",
    "speaker_turn_change",
    "source_cluster_support",
    "source_cluster_conflict",
    "insufficient_evidence",
]


def _has_grounded_onscreen_evidence(
    decision: MimoAVSegmentGrounding,
    view: MimoVisualSegmentView | None,
) -> bool:
    if decision.entity_id is None or view is None:
        return False
    observation = next(
        (
            item
            for item in view.entity_observations
            if item.entity_id == decision.entity_id
        ),
        None,
    )
    if observation is None:
        return False
    evidence = set(decision.evidence_codes)
    if (
        observation.speech_correlated_articulation == "observed"
        and "visible_lip_motion" in evidence
    ):
        return True
    non_assessable = (
        observation.orientation in {"profile", "back", "mixed"}
        or observation.face_visibility in {"occluded", "out_of_frame", "not_assessable"}
        or observation.mouth_visibility in {"occluded", "out_of_frame", "not_assessable"}
    )
    return (
        non_assessable
        and "speaker_visible_mouth_occluded" in evidence
        and bool(evidence & {"av_temporal_alignment", "voice_continuity"})
    )


def _observed_articulation_entity_ids(
    view: MimoVisualSegmentView | None,
    *,
    allowed_entity_ids: set[str],
) -> set[str]:
    if view is None:
        return set()
    return {
        item.entity_id
        for item in view.entity_observations
        if item.entity_id in allowed_entity_ids
        and item.speech_correlated_articulation == "observed"
    }


def _has_stage_a_c_articulation_contradiction(
    decision: MimoAVSegmentGrounding,
    view: MimoVisualSegmentView | None,
    *,
    allowed_entity_ids: set[str],
) -> bool:
    observed_entity_ids = _observed_articulation_entity_ids(
        view,
        allowed_entity_ids=allowed_entity_ids,
    )
    evidence = set(decision.evidence_codes)
    if "no_visible_lip_motion" in evidence and observed_entity_ids:
        return True
    if "visible_lip_motion" not in evidence:
        return False
    if decision.entity_id is None:
        return not observed_entity_ids
    if decision.entity_id not in allowed_entity_ids:
        return False
    return decision.entity_id not in observed_entity_ids


def _has_incomplete_onscreen_grounding_contradiction(
    decision: MimoAVSegmentGrounding,
    view: MimoVisualSegmentView | None,
    *,
    allowed_entity_ids: set[str],
) -> bool:
    return (
        decision.speech_presentation == "onscreen_spoken"
        and (
            decision.binding_status != "visible_entity"
            or decision.entity_id is None
        )
        and bool(
            _observed_articulation_entity_ids(
                view,
                allowed_entity_ids=allowed_entity_ids,
            )
        )
    )


def _audio_allows_visible_entity_resolution(
    decision: MimoAudioSegmentDecision | None,
) -> bool:
    if decision is None or decision.primary_speaker_group is None:
        return False
    if decision.resolution == "resolved":
        return True
    return (
        decision.resolution == "needs_acoustic_refinement"
        and decision.vocal_composition
        in {"single_speaker", "same_speaker_nonlexical"}
    )


def _compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalized_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def _audio_layer_content_tokens(value: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return [
        token
        for token in _AUDIO_LAYER_CONTENT_TOKEN.findall(normalized)
        if token not in _AUDIO_LAYER_STOPWORDS
    ]


def _substantially_same_audio_layer(first: str, second: str) -> bool:
    first_tokens = _audio_layer_content_tokens(first)
    second_tokens = _audio_layer_content_tokens(second)
    first_set = set(first_tokens)
    second_set = set(second_tokens)
    if min(len(first_set), len(second_set)) < _AUDIO_LAYER_MIN_CONTENT_TOKEN_COUNT:
        return False
    first_text = " ".join(first_tokens)
    second_text = " ".join(second_tokens)
    if first_text in second_text or second_text in first_text:
        return True
    overlap = len(first_set & second_set)
    return overlap / min(len(first_set), len(second_set)) >= (
        _AUDIO_LAYER_MIN_CONTAINMENT_RATIO
    )


def _soundscape_contamination_match_is_negated(
    clause: str,
    match: re.Match[str],
) -> bool:
    if _SOUNDSCAPE_POST_NEGATION.match(clause[match.end() :]):
        return True

    prefix = clause[: match.start()]
    negators = list(_SOUNDSCAPE_LOCAL_NEGATOR.finditer(prefix))
    if not negators:
        return False
    negator = negators[-1]
    between = prefix[negator.end() :]
    if negator.group(0).casefold() == "not" and re.match(
        r"\s+only\b", between, flags=re.IGNORECASE
    ):
        return False
    if _SOUNDSCAPE_NEGATION_SCOPE_BREAK.search(between):
        return False
    return len(re.findall(r"\b[\w'-]+\b", between)) <= 8


def _contains_positive_soundscape_contamination(text: str) -> bool:
    for clause in _SOUNDSCAPE_CLAUSE_BOUNDARY.split(text):
        for match in _SOUNDSCAPE_CONTAMINATION.finditer(clause):
            if not _soundscape_contamination_match_is_negated(clause, match):
                return True
    return False


def _is_explicit_negative_soundscape_description(text: str) -> bool:
    clauses = [
        clause.strip()
        for clause in _SOUNDSCAPE_ABSENCE_CLAUSE_BOUNDARY.split(text)
        if clause.strip()
    ]
    if not clauses:
        return False
    for clause in clauses:
        match = _EXPLICIT_SOUNDSCAPE_ABSENCE.fullmatch(clause)
        if match is None or _SOUNDSCAPE_POSITIVE_VERB.search(match.group("body")):
            return False
    return True


def _deduplicate_exact_strings(values: object) -> tuple[object, int]:
    if not isinstance(values, list):
        return values, 0
    seen: set[str] = set()
    result: list[object] = []
    removed = 0
    for value in values:
        if isinstance(value, str) and value in seen:
            removed += 1
            continue
        if isinstance(value, str):
            seen.add(value)
        result.append(value)
    return result, removed


def _strip_redundant_retention_marker_prefix(
    marker: object,
    description: object,
) -> tuple[object, bool]:
    if not isinstance(marker, str) or not isinstance(description, str):
        return description, False
    marker_pattern = r"[\s_-]+".join(re.escape(part) for part in marker.split("_"))
    match = re.match(
        rf"^\s*{marker_pattern}\s*(?:-|:|\u2013|\u2014)\s*",
        description,
        flags=re.IGNORECASE,
    )
    if match is None:
        return description, False
    remainder = description[match.end() :].strip()
    if not remainder:
        return description, False
    return remainder, True


def _canonicalize_raw_annotation_payload(
    raw: str,
) -> tuple[str, dict[str, int]]:
    try:
        payload = json.loads(normalize_structured_json_envelope(raw))
    except (json.JSONDecodeError, TypeError, ValueError):
        return raw, {}
    if not isinstance(payload, dict):
        return raw, {}

    corrections: Counter[str] = Counter()
    audio_observation = payload.get("audio_observation")
    if isinstance(audio_observation, dict):
        decisions = audio_observation.get("segment_decisions")
        if isinstance(decisions, list):
            for decision in decisions:
                if not isinstance(decision, dict) or "audio_evidence_codes" not in decision:
                    continue
                values, count = _deduplicate_exact_strings(
                    decision["audio_evidence_codes"]
                )
                if count:
                    decision["audio_evidence_codes"] = values
                    corrections["audio_evidence_code_deduplication"] += count

    av_grounding = payload.get("av_grounding")
    if isinstance(av_grounding, dict):
        groundings = av_grounding.get("segment_groundings")
        if isinstance(groundings, list):
            for grounding in groundings:
                if not isinstance(grounding, dict):
                    continue
                if "evidence_codes" in grounding:
                    values, count = _deduplicate_exact_strings(
                        grounding["evidence_codes"]
                    )
                    if count:
                        grounding["evidence_codes"] = values
                        corrections[
                            "av_grounding_evidence_code_deduplication"
                        ] += count
                if (
                    grounding.get("binding_status") == "visible_entity"
                    and grounding.get("speech_presentation") == "offscreen_spoken"
                ):
                    evidence = grounding.get("evidence_codes")
                    offscreen = isinstance(evidence, list) and "offscreen_audio" in evidence
                    grounding["binding_status"] = "offscreen" if offscreen else "no_reliable_entity"
                    grounding["speech_presentation"] = "offscreen_spoken" if offscreen else "uncertain"
                    grounding["entity_id"] = None
                    grounding["confidence"] = "low"
                    if not offscreen and isinstance(evidence, list):
                        _append_insufficient_evidence(grounding)
                    corrections["raw_visible_offscreen_binding_downgrade"] += 1
                if (
                    grounding.get("binding_status") != "visible_entity"
                    and grounding.get("entity_id") is not None
                ):
                    grounding["entity_id"] = None
                    corrections["non_visible_binding_entity_id_removed"] += 1

    h3_semantics = payload.get("h3_semantics")
    if isinstance(h3_semantics, dict):
        retention_rows = h3_semantics.get("visual_retention_analysis")
        if isinstance(retention_rows, list):
            for row in retention_rows:
                if not isinstance(row, dict):
                    continue
                description, changed = _strip_redundant_retention_marker_prefix(
                    row.get("marker"),
                    row.get("description"),
                )
                if changed:
                    row["description"] = description
                    corrections["retention_marker_prefix_normalization"] += 1

    if not corrections:
        return raw, {}
    return _compact_json(payload), dict(sorted(corrections.items()))


def _significant_transcript(value: str) -> bool:
    normalized = _normalized_text(value)
    cjk_count = sum("\u3400" <= character <= "\u9fff" for character in normalized)
    alphanumeric_count = sum(character.isalnum() for character in normalized)
    return cjk_count >= 4 or alphanumeric_count >= 12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


SecondaryVocalKind = Literal[
    "discourse_particle",
    "interjection",
    "laughter",
    "cough",
    "sigh",
    "breath",
    "gasp",
    "crying",
    "non_lyrical_singing",
    "speech",
    "other_nonlexical",
    "unknown",
]


class MimoAbsentSecondaryVocalActivity(SchemaModel):
    present: Literal[False]
    speaker_relation: Literal["none"]
    kind: None


class MimoPresentSecondaryVocalActivity(SchemaModel):
    present: Literal[True]
    speaker_relation: Literal["same_speaker", "different_speaker", "uncertain"]
    kind: SecondaryVocalKind


MimoSecondaryVocalActivity = Annotated[
    MimoAbsentSecondaryVocalActivity | MimoPresentSecondaryVocalActivity,
    Field(discriminator="present"),
]


class _MimoAudioSegmentDecisionBase(SchemaModel):
    segment_id: str = Field(min_length=1)
    vocal_composition: VocalComposition
    resolution: Resolution
    primary_speaker_group: str | None = Field(pattern=r"^g[1-9]\d*$")
    delivery_style: StrictStr | None
    secondary_vocal_activity: MimoSecondaryVocalActivity
    confidence: Literal["high", "medium", "low"]
    audio_evidence_codes: list[AudioEvidenceCode] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def validate_decision(self) -> _MimoAudioSegmentDecisionBase:
        if self.delivery_style is not None and not self.delivery_style.strip():
            raise ValueError("MiMo segment delivery must be non-empty or null")
        if self.resolution == "resolved" and self.primary_speaker_group is None:
            raise ValueError("resolved segment requires one primary speaker group")
        secondary = self.secondary_vocal_activity
        if self.vocal_composition == "single_speaker":
            if secondary.present:
                raise ValueError("single_speaker forbids secondary vocal activity")
        elif self.vocal_composition == "same_speaker_nonlexical":
            if (
                not secondary.present
                or secondary.speaker_relation != "same_speaker"
                or secondary.kind == "speech"
            ):
                raise ValueError("same-speaker nonlexical composition is inconsistent")
        elif self.vocal_composition == "secondary_non_speech_vocalization":
            if (
                not secondary.present
                or secondary.speaker_relation not in {"different_speaker", "uncertain"}
                or secondary.kind == "speech"
            ):
                raise ValueError("secondary non-speech vocalization is inconsistent")
        elif (
            self.vocal_composition
            in {"overlapping_secondary_speech", "sequential_multi_speaker_speech"}
            and (
                not secondary.present
                or secondary.speaker_relation != "different_speaker"
                or secondary.kind != "speech"
                or self.resolution != "needs_acoustic_refinement"
            )
        ):
            raise ValueError("multi-speaker speech requires acoustic refinement")
        if len(self.audio_evidence_codes) != len(set(self.audio_evidence_codes)):
            raise ValueError("Audio segment evidence codes must be unique")
        return self


class MimoResolvedAudioSegmentDecision(_MimoAudioSegmentDecisionBase):
    resolution: Literal["resolved"]
    primary_speaker_group: str = Field(pattern=r"^g[1-9]\d*$")


class MimoAcousticRefinementAudioSegmentDecision(_MimoAudioSegmentDecisionBase):
    resolution: Literal["needs_acoustic_refinement"]
    primary_speaker_group: str | None = Field(pattern=r"^g[1-9]\d*$")


class MimoUncertainAudioSegmentDecision(_MimoAudioSegmentDecisionBase):
    resolution: Literal["uncertain"]
    primary_speaker_group: str | None = Field(pattern=r"^g[1-9]\d*$")


MimoAudioSegmentDecision = Annotated[
    MimoResolvedAudioSegmentDecision
    | MimoAcousticRefinementAudioSegmentDecision
    | MimoUncertainAudioSegmentDecision,
    Field(discriminator="resolution"),
]


class MimoAVSegmentGrounding(SchemaModel):
    segment_id: str = Field(min_length=1)
    primary_speaker_group: str | None = Field(pattern=r"^g[1-9]\d*$")
    binding_status: BindingStatus
    speech_presentation: SpeechPresentation
    entity_id: str | None
    confidence: Literal["high", "medium", "low"]
    evidence_codes: list[EvidenceCode] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def validate_grounding(self) -> MimoAVSegmentGrounding:
        if self.binding_status == "visible_entity":
            if self.entity_id is None:
                raise ValueError("visible_entity requires supplied entity_id")
            if self.speech_presentation != "onscreen_spoken":
                raise ValueError("visible_entity requires onscreen_spoken presentation")
        elif self.entity_id is not None:
            raise ValueError("only visible_entity may publish entity_id")
        if self.speech_presentation == "onscreen_spoken":
            if self.binding_status == "offscreen":
                raise ValueError("onscreen_spoken cannot use offscreen binding status")
        elif self.binding_status == "visible_entity" or self.entity_id is not None:
            raise ValueError("non-onscreen speech cannot claim a visible entity")
        if len(self.evidence_codes) != len(set(self.evidence_codes)):
            raise ValueError("AV grounding evidence codes must be unique")
        return self


MimoSegmentDecision = MimoAVSegmentGrounding


class MimoAudioEvent(SchemaModel):
    event_id: str = Field(pattern=r"^ae[1-9]\d*$")
    approximate_start_time: float = Field(ge=0, allow_inf_nan=False)
    approximate_end_time: float = Field(gt=0, allow_inf_nan=False)
    category: Literal[
        "physical",
        "environmental",
        "mechanical",
        "electronic",
        "human_non_speech",
        "diegetic_music",
        "non_diegetic_music",
        "other",
    ]
    pattern: Literal["single", "repeated", "continuous"]
    description: StrictStr
    source_grounding: Literal[
        "audible_only", "audiovisually_grounded", "uncertain_source"
    ]

    @model_validator(mode="after")
    def validate_event(self) -> MimoAudioEvent:
        if self.approximate_end_time <= self.approximate_start_time:
            raise ValueError("MiMo Audio event must have positive duration")
        if not self.description.strip():
            raise ValueError("MiMo Audio event description must not be empty")
        if (
            re.search(r"\b\d{1,2}:\d{2}(?:\.\d+)?\b", self.description)
            or _PIPELINE_OWNED.search(self.description)
            or _PICTURE_OR_SUBJECT.search(self.description)
            or "[[" in self.description
            or "]]" in self.description
        ):
            raise ValueError("MiMo Audio event description contains forbidden syntax")
        return self


class MimoAudioSemantics(SchemaModel):
    temporal_non_speech_events: list[MimoAudioEvent]
    overall_soundscape_status: Literal["present", "absent", "unknown"]
    overall_soundscape: StrictStr | None = None
    complete_silence_verified: StrictBool = False
    non_diegetic_music_status: Literal["present", "absent", "unknown"]
    non_diegetic_music: StrictStr | None = None
    audiovisual_summary: StrictStr


class MimoVisualBlock(SchemaModel):
    block_id: str = Field(pattern=r"^v[1-9]\d*$")
    text: StrictStr

    @model_validator(mode="after")
    def validate_visual_text(self) -> MimoVisualBlock:
        if (
            not self.text.strip()
            or "[[" in self.text
            or "]]" in self.text
            or _PIPELINE_OWNED.search(self.text)
            or _PICTURE_OR_SUBJECT.search(self.text)
            or "<" in self.text
            or ">" in self.text
        ):
            raise ValueError("MiMo visual block contains non-visual or pipeline syntax")
        return self


class MimoVisibleEntityObservation(SchemaModel):
    entity_id: str = Field(min_length=1)
    visibility: Literal["visible", "partially_visible"]
    orientation: Literal[
        "front", "three_quarter", "profile", "back", "mixed", "uncertain"
    ]
    face_visibility: Literal[
        "clear", "partial", "occluded", "out_of_frame", "not_assessable", "uncertain"
    ]
    mouth_visibility: Literal[
        "clear", "partial", "occluded", "out_of_frame", "not_assessable", "uncertain"
    ]
    speech_correlated_articulation: Literal[
        "observed", "not_observed", "not_assessable", "uncertain"
    ]


class MimoVisualSegmentView(SchemaModel):
    segment_id: str = Field(min_length=1)
    visible_entity_ids: list[str]
    entity_observations: list[MimoVisibleEntityObservation]

    @model_validator(mode="after")
    def validate_entities(self) -> MimoVisualSegmentView:
        observed = [item.entity_id for item in self.entity_observations]
        if (
            len(self.visible_entity_ids) != len(set(self.visible_entity_ids))
            or len(observed) != len(set(observed))
            or self.visible_entity_ids != observed
        ):
            raise ValueError("visual segment entity inventory must agree exactly")
        return self


class MimoVisualObservation(SchemaModel):
    visual_blocks: list[MimoVisualBlock] = Field(min_length=1)
    segment_views: list[MimoVisualSegmentView]


class MimoSubjectDefinitionDraft(SchemaModel):
    subject_label: str = Field(pattern=r"^<Subject [1-9]\d*>$")
    description: StrictStr

    @model_validator(mode="after")
    def validate_definition(self) -> MimoSubjectDefinitionDraft:
        if not self.description.strip():
            raise ValueError("MiMo Subject definition description must not be empty")
        if any(
            match.group(1) == "Picture"
            for match in _PICTURE_OR_SUBJECT.finditer(self.description)
        ):
            raise ValueError(
                "MiMo Subject definition description cannot own Picture provenance"
            )
        return self

    def render(self) -> str:
        return f"{self.subject_label} {self.description.strip()}"


class MimoVisualRetentionDraft(SchemaModel):
    subject_label: str = Field(pattern=r"^<Subject [1-9]\d*>$")
    marker: Literal["fully_preserved", "partially_preserved", "weak_reference"]
    description: StrictStr

    @model_validator(mode="after")
    def validate_retention(self) -> MimoVisualRetentionDraft:
        if not self.description.strip():
            raise ValueError("MiMo retention description must not be empty")
        return self

    def render(self) -> str:
        return f"{self.subject_label}: {self.marker} - {self.description.strip()}"


class MimoH3Semantics(SchemaModel):
    subject_definitions: list[MimoSubjectDefinitionDraft] = Field(min_length=1)
    summary: StrictStr
    style_opening: StrictStr = Field(min_length=1)
    shot1_caption: StrictStr = Field(min_length=1)
    overall_soundscape: StrictStr
    non_diegetic_music: StrictStr
    visual_retention_analysis: list[MimoVisualRetentionDraft] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_semantics(self) -> MimoH3Semantics:
        values = (
            *(item.description for item in self.subject_definitions),
            self.summary,
            self.style_opening,
            self.shot1_caption,
            *(item.description for item in self.visual_retention_analysis),
        )
        if any(not value.strip() for value in values):
            raise ValueError("MiMo H3 semantics text must not be empty")
        if any(re.search(r"\[Shot\s+\d+\]", value) for value in (self.style_opening, self.shot1_caption)):
            raise ValueError("MiMo must not emit pipeline-owned shot markers")
        if "<d>" in self.style_opening or "</d>" in self.style_opening:
            raise ValueError("dialogue belongs only in shot1_caption")
        return self


class MimoAnnotationWarning(SchemaModel):
    code: Literal["possible_asr_conflict"]
    segment_id: str | None = None

    @model_validator(mode="after")
    def validate_warning(self) -> MimoAnnotationWarning:
        if self.code == "possible_asr_conflict" and self.segment_id is None:
            raise ValueError("possible ASR conflict warning requires segment_id")
        return self


class MimoSpeakerVoiceProfile(SchemaModel):
    speaker_group: str = Field(pattern=r"^g[1-9]\d*$")
    voice_characteristics: StrictStr | None

    @model_validator(mode="after")
    def validate_profile(self) -> MimoSpeakerVoiceProfile:
        if self.voice_characteristics is not None and not self.voice_characteristics.strip():
            raise ValueError("MiMo voice profile must be non-empty or null")
        return self


class MimoAudioObservation(SchemaModel):
    segment_decisions: list[MimoAudioSegmentDecision]
    speaker_voice_profiles: list[MimoSpeakerVoiceProfile]


class MimoAVGrounding(SchemaModel):
    segment_groundings: list[MimoAVSegmentGrounding]


MimoH3Draft = MimoH3Semantics


class MimoAVAnnotationDraft(SchemaModel):
    schema_version: Literal["r2v.h3.mimo25_av_annotation.20"] = MIMO25_SCHEMA_VERSION
    visual_observation: MimoVisualObservation
    audio_observation: MimoAudioObservation
    av_grounding: MimoAVGrounding
    h3_semantics: MimoH3Semantics
    warnings: list[MimoAnnotationWarning] = Field(default_factory=list)

    @property
    def segment_decisions(self) -> list[MimoAVSegmentGrounding]:
        return self.av_grounding.segment_groundings

    @property
    def speaker_voice_profiles(self) -> list[MimoSpeakerVoiceProfile]:
        return self.audio_observation.speaker_voice_profiles


class MimoThinkingContract(SchemaModel):
    type: Literal["disabled", "enabled"] = "disabled"


class MimoBackendProvenance(SchemaModel):
    schema_version: Literal["r2v.h3.mimo25_backend.46"] = MIMO25_BACKEND_VERSION
    speaker_marker_polish_prompt_version: Literal["h3_mimo25_speaker_marker_polish_v4"] = (
        MIMO25_SPEAKER_MARKER_POLISH_PROMPT_VERSION
    )
    backend: Literal[
        "xiaomi_openai_compatible", "sglang_openai_compatible"
    ]
    transport: MimoTransport
    model: Literal["mimo-v2.5"] = MIMO25_MODEL
    base_url: str
    video_fps: Literal[4.0] = 4.0
    media_resolution: Literal["default"] = "default"
    thinking: MimoThinkingContract
    icl_version: Literal["h3_official_ref2va_detailed_shot1_v4"] | None
    temperature: float = Field(ge=0, allow_inf_nan=False)
    max_completion_tokens: int = Field(gt=0)
    response_format: Literal["json_object", "json_schema"]
    stream: Literal[False] = False
    media_mode: Literal["base64", "http"]
    media_root: str
    media_base_url: str | None = None
    prompt_version: Literal["h3_mimo25_unified_av_reconcile_v38"] = (
        MIMO25_PROMPT_VERSION
    )
    policy_version: Literal["h3_mimo25_av_authority_contract_v17"] = (
        MIMO25_POLICY_VERSION
    )
    annotation_schema_version: Literal["r2v.h3.mimo25_av_annotation.20"] = (
        MIMO25_SCHEMA_VERSION
    )
    materializer_version: Literal[
        "h3_mimo25_materializer_v6",
        "h3_mimo25_materializer_v7",
        "h3_mimo25_materializer_v8",
        "h3_mimo25_materializer_v9",
        "h3_mimo25_materializer_v10",
        "h3_mimo25_materializer_v11",
        "h3_mimo25_materializer_v12",
        "h3_mimo25_materializer_v13",
        "h3_mimo25_materializer_v14",
        "h3_mimo25_materializer_v15",
        "h3_mimo25_materializer_v16",
        "h3_mimo25_materializer_v17",
        "h3_mimo25_materializer_v23",
    ] = (
        MIMO25_MATERIALIZER_VERSION
    )
    http_max_attempts: Literal[1] = 1
    full_av_recheck_limit: Literal[0] = 0
    configuration_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_provenance(self) -> MimoBackendProvenance:
        if not self.base_url.strip() or not self.media_root.strip():
            raise ValueError("MiMo endpoint and media root are required")
        expected_backend = f"{self.transport}_openai_compatible"
        if self.backend != expected_backend:
            raise ValueError("MiMo backend provenance differs from transport")
        expected_response_format = (
            "json_schema" if self.transport == "sglang" else "json_object"
        )
        if self.response_format != expected_response_format:
            raise ValueError("MiMo response format differs from transport")
        if (self.media_mode == "http") != (self.media_base_url is not None):
            raise ValueError("MiMo HTTP media mode requires one public base URL")
        values = self.model_dump(mode="json", exclude={"configuration_fingerprint"})
        if self.configuration_fingerprint != _sha256_text(_compact_json(values)):
            raise ValueError("MiMo backend configuration fingerprint is invalid")
        return self


class MimoUsage(SchemaModel):
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    image_tokens: int | None = Field(default=None, ge=0)
    video_tokens: int | None = Field(default=None, ge=0)
    audio_tokens: int | None = Field(default=None, ge=0)
    cached_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)


class MimoCompletionDiagnostic(SchemaModel):
    input_modality: Literal[
        "target_video_with_embedded_audio",
        "target_video_plus_canonical_full_audio_fallback",
        "full_av_recheck_embedded_audio",
        "full_av_recheck_with_canonical_audio",
        "target_av_with_auxiliary_raw_audio",
        "auxiliary_audio_only",
        "speaker_marker_text_only",
    ]
    finish_reason: str | None = None
    usage: MimoUsage
    http_attempt_count: int = Field(ge=1)
    warnings: list[str] = Field(default_factory=list)
    request_error: str | None = None


class MimoSpeakerMarkerPolish(SchemaModel):
    shot1_caption: StrictStr
    needs_review: StrictBool = False


class MimoSpeakerMarkerPolishAudit(SchemaModel):
    attempted: bool = False
    applied: bool = False
    needs_review: bool | None = None
    raw_response: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class MimoBackendResult:
    annotation: MimoAVAnnotationDraft
    raw_responses: tuple[str, ...]
    diagnostics: tuple[MimoCompletionDiagnostic, ...]
    model_call_count: int
    http_attempt_count: int
    http_retry_count: int
    recheck_count: int
    input_modality: Literal[
        "target_video_with_embedded_audio",
        "target_video_plus_canonical_full_audio_fallback",
        "target_av_with_auxiliary_raw_audio",
    ]
    deterministic_correction_counts: dict[str, int] = field(default_factory=dict)
    text_model_call_count: int = 0
    speaker_marker_polish: MimoSpeakerMarkerPolishAudit = field(default_factory=MimoSpeakerMarkerPolishAudit)


class MimoAuxAudioDescription(SchemaModel):
    description: StrictStr


@dataclass(frozen=True)
class MimoAuxAudioDescriptionCall:
    diagnostic: MimoCompletionDiagnostic
    description: str | None = None
    raw_response: str | None = None
    error: str | None = None
    model_call_count: int = 1


AUXILIARY_AUDIO_PROMPT = """Listen to the supplied separated audio and describe only POSITIVE audible content and useful perceptual/acoustic qualities in natural English.
Describe sounds that are present. Do NOT summarize what is absent: do not write "no other sounds", "no voices", "no music", "no instruments", or "nothing else is audible".
Do NOT infer a physical source or cause, room/location/environment, recording setup or processing history. Do not claim a vehicle, engine or machinery unless directly unmistakable.
If the physical source is uncertain, describe the acoustic event itself: low continuous hum, low-frequency rumble, rhythmic clicking or reverberant tone.
Perceptual wording such as soft, slow, gentle, muffled, reverberant, melancholic or reflective is allowed. Preserve uncertainty without guessing a cause.
Do not infer content from the file name or expected role. Do not classify the track by its expected category or decide diegetic versus non-diegetic. Do not add visual context.
Do not mention that this is a source-separation output in the returned description."""


class MimoBackendFailure(ValueError):
    def __init__(
        self,
        *,
        code: str,
        reason: str,
        raw_responses: tuple[str, ...] = (),
        diagnostics: tuple[MimoCompletionDiagnostic, ...] = (),
        issues: tuple[ValidationIssue, ...] = (),
        model_call_count: int = 0,
        http_attempt_count: int = 0,
        http_retry_count: int = 0,
        recheck_count: int = 0,
        annotation: MimoAVAnnotationDraft | None = None,
        text_model_call_count: int = 0,
        speaker_marker_polish: MimoSpeakerMarkerPolishAudit | None = None,
    ) -> None:
        super().__init__(reason)
        self.code = code
        self.reason = reason
        self.raw_responses = raw_responses
        self.diagnostics = diagnostics
        self.issues = issues
        self.model_call_count = model_call_count
        self.http_attempt_count = http_attempt_count
        self.http_retry_count = http_retry_count
        self.recheck_count = recheck_count
        self.annotation = annotation
        self.text_model_call_count = text_model_call_count
        self.speaker_marker_polish = speaker_marker_polish or MimoSpeakerMarkerPolishAudit()


class MimoBackendJob(Protocol):
    clip_uid: str
    r2v_instruction: str
    target_video_path: str
    target_full_audio_path: str
    target_duration_seconds: float
    reference_selection: Any
    reference_subjects: list[Any]
    reference_images: list[Any]
    segments: list[Any]

    def model_dump(self, *, mode: str, exclude: set[str] | None = None) -> dict[str, Any]: ...


class MimoMediaResolver:
    def __init__(
        self,
        *,
        mode: Literal["base64", "http"],
        media_root: Path,
        media_base_url: str | None = None,
        maximum_base64_bytes: int = DEFAULT_BASE64_LIMIT_BYTES,
    ) -> None:
        self.mode = mode
        self.media_root = media_root.expanduser().resolve(strict=True)
        self.media_base_url = media_base_url
        self.maximum_base64_bytes = maximum_base64_bytes
        if not self.media_root.is_dir() or maximum_base64_bytes <= 0:
            raise ValueError("MiMo media root and Base64 limit must be valid")
        if mode == "http":
            if media_base_url is None:
                raise ValueError("MiMo HTTP media mode requires a public base URL")
            parsed = urlsplit(media_base_url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("MiMo media base URL must be public HTTP(S)")
            self.media_base_url = media_base_url.rstrip("/") + "/"
        elif media_base_url is not None:
            raise ValueError("MiMo Base64 media mode cannot define an HTTP URL")

    def resolve(self, path: Path) -> str:
        source = path.expanduser().resolve(strict=True)
        if not source.is_file():
            raise ValueError("MiMo media input must be a readable file")
        try:
            relative = source.relative_to(self.media_root)
        except ValueError as exc:
            raise ValueError("MiMo media is outside the configured media root") from exc
        if self.mode == "http":
            assert self.media_base_url is not None
            return self.media_base_url + quote(relative.as_posix(), safe="/")
        mime = mimetypes.guess_type(source.name)[0] or "application/octet-stream"
        encoded_size = 4 * ((source.stat().st_size + 2) // 3)
        if encoded_size > self.maximum_base64_bytes:
            raise MimoBackendFailure(
                code="media_too_large",
                reason=f"Base64 media exceeds {self.maximum_base64_bytes} bytes",
            )
        return f"data:{mime};base64," + base64.b64encode(source.read_bytes()).decode("ascii")


@dataclass(frozen=True)
class MimoBackendConfig:
    media_resolver: MimoMediaResolver
    api_key: str
    transport: MimoTransport = "xiaomi"
    base_url: str = MIMO25_DEFAULT_BASE_URL
    model: str = MIMO25_MODEL
    video_fps: float = 4.0
    media_resolution: str = "default"
    temperature: float = 0.0
    max_completion_tokens: int = 16384
    timeout_seconds: float = 900.0
    http_max_attempts: int = 1
    thinking: Literal["disabled", "enabled"] = "disabled"
    icl: Literal["none", "official_ref2va_v1"] = "official_ref2va_v1"

    def __post_init__(self) -> None:
        if (
            not self.api_key.strip()
            or self.transport not in {"xiaomi", "sglang"}
            or self.thinking not in {"disabled", "enabled"}
            or self.icl not in {"none", "official_ref2va_v1"}
            or not self.base_url.strip()
            or self.model != MIMO25_MODEL
            or self.video_fps != 4.0
            or self.media_resolution != "default"
            or not math.isfinite(self.temperature)
            or self.temperature < 0
            or self.max_completion_tokens <= 0
            or self.timeout_seconds <= 0
            or self.http_max_attempts != 1
        ):
            raise ValueError("MiMo backend configuration violates the single-attempt contract")

    def provenance(self) -> MimoBackendProvenance:
        values = {
            "schema_version": MIMO25_BACKEND_VERSION,
            "backend": f"{self.transport}_openai_compatible",
            "transport": self.transport,
            "model": self.model,
            "base_url": self.base_url,
            "video_fps": self.video_fps,
            "media_resolution": self.media_resolution,
            "thinking": {"type": self.thinking},
            "icl_version": MIMO25_ICL_VERSION if self.icl == "official_ref2va_v1" else None,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_completion_tokens,
            "response_format": (
                "json_schema" if self.transport == "sglang" else "json_object"
            ),
            "stream": False,
            "media_mode": self.media_resolver.mode,
            "media_root": str(self.media_resolver.media_root),
            "media_base_url": self.media_resolver.media_base_url,
            "prompt_version": MIMO25_PROMPT_VERSION,
            "speaker_marker_polish_prompt_version": MIMO25_SPEAKER_MARKER_POLISH_PROMPT_VERSION,
            "policy_version": MIMO25_POLICY_VERSION,
            "annotation_schema_version": MIMO25_SCHEMA_VERSION,
            "materializer_version": MIMO25_MATERIALIZER_VERSION,
            "http_max_attempts": self.http_max_attempts,
            "full_av_recheck_limit": 0,
        }
        return MimoBackendProvenance(
            **values,
            configuration_fingerprint=_sha256_text(_compact_json(values)),
        )


SYSTEM_PROMPT = """ROLE / OUTPUT
Return one compact JSON object in the supplied schema for this H3 shadow pipeline. Populate visual/audio observations, AV grounding, and h3_semantics in ONE response; do not emit hidden reasoning.

AUTHORITY
- DiariZen owns exact segment/sample boundaries. All decision inventories follow allowed_segment_ids, including LR-ASD=0, unbound, and zero-anchor segments. Never split, merge, filter, or invent segments.
- If allowed_segment_ids is empty, output empty visual_observation.segment_views, audio_observation.segment_decisions, and av_grounding.segment_groundings. Never invent a synthetic segment.
- Preserve supplied Qwen3-ASR dialogue content and language without additions or omissions. Consecutive speech by the same speaker may share one natural <d> block; never alter upstream segment boundaries.
- Frozen entities, Subjects, Pictures, order, and ownership are immutable. Use H3 reference labels ONLY from allowed_h3_reference_labels. If no <Audio N> appears in allowed_h3_reference_labels, NEVER emit <Audio N>. Do not mention a referenced voice timbre, Audio reference, or "using the voice from <Audio N>" unless that exact Audio label is explicitly allowed. Target video is observation-only, never <Video N>. Current LR-ASD bindings and source clusters are proposals, not truth.

VISUAL OBSERVATION
- visual_blocks are concise English visual evidence, not a second caption draft: use one coherent block by default, more only for meaningful progression. No timestamps, transcript, audio inference, pipeline labels, or invented psychology, intent, causality, relationships, or unseen details.
- segment_views follow allowed_segment_ids. visible_entity_ids and entity_observations agree exactly on entities visible in that interval. Back/profile views, occluded faces, and hidden/cropped mouths still count as visible presence. speech_correlated_articulation records exact-window observation, not speaker identity.

AUDIO + AV GROUNDING
- Stage B identifies acoustic speakers as contiguous gN by first appearance, not turns. Pauses, language, sentences, ASR, or segment boundaries alone never create groups. Record vocal_composition, delivery, secondary_vocal_activity, and non-speech evidence, without entity identity or spatial presentation.
- Multiple vocal sounds are valid observations; use needs_acoustic_refinement if primary identity is unsafe. Each transcribed segment needs nonempty delivery_style; non-transcribed segments use null. Voice profiles cover resolved transcribed groups in first-appearance order with supported acoustic traits, not transcript or identity claims.
- Original AV is the sound authority.
- Stage C preserves each Stage B primary group. Stage B acoustic grouping remains evidence, but resolution == resolved is NOT a prerequisite for final visible binding. Single, likely-single, or ordinary uncertain acoustic evidence may receive a direct final AV visible binding; confirmed transcribed overlapping/sequential multi-speaker speech remains excluded from identity publication.
- Offscreen is spatial: hidden lips/face or partial occlusion of a visible person is not offscreen. offscreen_spoken requires offscreen/null entity/offscreen_audio; voice_over requires null/voice_over_context; device_playback requires null/device_playback_context; message_voice_over requires null/message_text_alignment/voice_over_context. Do not infer offscreen from missing supporting evidence.
- direct_anchor_present without explicit LR-ASD conflict is a strong prior, overridden only by current-segment AV contradiction. LR-ASD support is not required to recover a true visible speaker. The same reliably resolved visible entity reuses one group; split/merge source clusters only with AV support.
- Transcribed overlapping_secondary_speech or sequential_multi_speaker_speech blocks final publication pending authoritative turn refinement; preserve the raw caption for QA, never heuristically split ASR.

VISIBLE SPEAKER BINDING
- The final target AV is allowed to identify a visible speaker directly. If the full AV reasonably indicates that a known visible entity is speaking, bind that entity directly.
- LR-ASD, source clusters, lip motion, mouth visibility, temporal alignment, and voice continuity are supporting clues, NOT mandatory prerequisites. Absence of supporting evidence is NOT a contradiction.
- Do not downgrade a plausible visible speaker merely because lip articulation is subtle, occluded, profile-view, cropped, or not independently confirmed by LR-ASD.
- Use no_reliable_entity / uncertain only when the speaker is genuinely ambiguous, multiple speakers prevent safe attribution, or the AV contains concrete contradictory evidence.
- A visible listener must still not inherit speech when the AV clearly indicates another source. Reinspect conflicting Stage A articulation and Stage C lip-motion evidence. Never invent an entity.

EXISTING REFERENCE FIELDS
- Write natural visual Subject definitions, concise summary, and retention rows. Pipeline code appends exact Picture provenance; omit Picture labels from definition descriptions. Entity Subjects describe reusable entities. Attribute Subjects describe only their attribute, not a second person/object or their owner: hair shape/color/texture, facial features, eyewear, garment, or accessory as applicable.
- Attribute retention is judged on its owning entity, not independent existence. Allowed markers: fully_preserved, partially_preserved, weak_reference; attribute_transfer is forbidden.
- In both subject_definitions and visual_retention_analysis, subject_label already owns the label; description should not repeat <Subject N> or bare Subject numbering.

AUXILIARY AUDIO EVIDENCE
- Auxiliary descriptions are positive acoustic observations from source-separated tracks of the SAME target clip. Treat clearly reported audible events as valid recall evidence: separation exposes sounds that may be weak or masked in the original mixture. SOURCE_UNAVAILABLE means no candidate evidence, never silence.
- FINAL AV IS A FACTUAL CONTRADICTION FILTER, NOT A REQUIREMENT TO RE-PROVE EVERY AUXILIARY DETAIL. Preserve positive observations by default. Do NOT require every auxiliary detail to be independently re-proven from the original mixture before keeping it.
- Use the ORIGINAL TARGET AV to correct CONCRETE FACTUAL ERRORS: a clearly wrong source or physical cause, incorrect scene/context claim, speech/music leakage misclassified as another sound, incorrect diegetic versus non-diegetic interpretation, or clear separator artifact. Otherwise preserve useful auxiliary detail.
- Do not discard a candidate merely because it is faint, masked, not independently clear in the original mix, subjectively worded, or lacks an exact source. Harmless acoustic/perceptual descriptions such as slow, gentle, simple, soft, continuous, melancholic, reflective, tense, light, slight hiss, muffled or reverberant may remain unless they introduce a concrete factual contradiction.
- Correct only the mistaken source/context interpretation, not the underlying audible event. For "a low rumble, possibly from an engine", if engine is unsupported but the rumble is not contradicted, retain "a low rumble"; do not turn it into "N/A". Remove the entire event only for a concrete factual falsehood or clear leakage/artifact.
- For music_separator_candidate, preserve clearly reported music and useful acoustic wording by default. Original AV determines H3 placement: audience-facing score -> non_diegetic_music; diegetic/in-scene music -> chronological shot1_caption and "N/A" for non_diegetic_music; concrete separator leakage/artifact -> discard. Never output "N/A" merely because music is weak or masked, or strip harmless adjectives merely because AV cannot independently verify them.
- An unseen source does not prove non-diegetic music. If the distinction is uncertain, preserve conservative audible-music prose in shot1_caption and use "N/A" for non_diegetic_music. Music never belongs in overall_soundscape.
- For sfx_separator_candidate, preserve positive non-musical, non-dialogue observations by default, including room tone, footsteps, handling, mechanical/environmental sounds and human non-speech. An unsupported engine guess may become "A low continuous rumble or hum is audible", not "N/A".
- Auxiliary negative/absence claims never establish absence.
- Negative/absence statements from an auxiliary track are TRACK-LOCAL and MUST NOT be copied into final H3. From "A piano melody is audible. There are no voices or other sounds", use only the positive melody observation. Never propagate "no other instruments", "no background noise", "no music" or similar absence prose into shot1_caption, overall_soundscape or non_diegetic_music.
- Keep observed acoustic content; remove unsupported causal/source/environment inference. Do not inherit "possibly an engine", "moving vehicle", "machinery", "large empty room", "recorded from a distance", "recorded through a barrier", "recording of a recording", or claims about synthesis/processing effects unless ORIGINAL TARGET AV itself establishes that specific fact. Preserve hum, rumble, rhythmic clicking, piano/violin, hiss, muffled/reverberant qualities and harmless perceptual wording without requiring independent re-proof.
- overall_soundscape contains ZERO music. An event classified as music must NOT also appear in overall_soundscape: non-diegetic music -> non_diegetic_music; diegetic music -> shot1_caption; never overall_soundscape. Do not write "no voices/music" there; write eligible ambience/SFX or "N/A".

PRIMARY H3 WRITING TASK
- This pilot is exactly one shot. Write style_opening and shot1_caption; the pipeline inserts [Shot 1]. Never output [Shot N], shot timing, or placeholders.
- style_opening: one concise sentence about global visual/cinematographic style, camera language, and lighting only. Do not summarize people, clothing, scene contents, Subjects, actions, chronology, dialogue, or audio.

SHOT1 CAPTION / DETAILED DESCRIPTION
- shot1_caption is the full detailed_description body for this single-shot clip, not a short caption or summary.
- Observe the ENTIRE target video from beginning to end before writing.
- Describe the shot in natural playback order using natural English audiovisual prose, with enough visual detail to reconstruct what is actually seen and how it changes.
For the shot, cover the observable dimensions that are present:
1. shot scale, framing, viewpoint, and composition;
2. each important visible subject's appearance, position, orientation, and spatial relationship to other subjects/objects;
3. environment, background, foreground, major props, and scene layout;
4. lighting, color, and clearly visible visual atmosphere;
5. camera behavior: static, pan, push, tracking, handheld motion, etc.;
6. visible actions, gestures, gaze changes, facial-expression changes, posture changes, object interactions, and other state changes in chronological order;
7. dialogue and speaker markers at the moment they occur; do not append all dialogue at the end;
8. Only localized diegetic or shot-synchronized sound events that need to appear at a specific point in playback order belong in shot1_caption;
9. where referenced Subjects/Pictures actually appear or affect the shot.
Important:
- A single shot is NOT a reason to make the description short.
- Do not stop after describing the opening composition and dialogue.
- Continue through the end of the clip and describe meaningful visible developments and reactions.
- Prefer concrete observable visual detail over plot summary or interpretation.
- Do NOT pad the description with invented details just to make it longer.
- Do NOT invent psychology, causality, relationships, unseen objects, camera motion, lighting changes, or actions that are not observable.
- No hard word-count requirement. Detail should scale with actual information visible in the clip.
- Avoid repeating the exact same fact merely to increase length.
- Continuous ambience / room tone / hum / rumble belongs only in overall_soundscape.
- Audience-only score/background music belongs only in non_diegetic_music.
- Do not summarize overall_soundscape or non_diegetic_music at the end of shot1_caption.
- Diegetic/in-scene music may remain in shot1_caption at its actual chronological position.
- shot1_caption states observable audiovisual events, not AV-grounding reasoning or diagnostic explanation.
- Never explain speaker/source inference with prose such as "indicating", "suggesting", "therefore", "because this means", "may be offscreen", or "may be a voice-over".
- Binding/presentation rationale belongs only in av_grounding.
- If a source is determined as offscreen/voice-over, express the final presentation naturally rather than explaining why it was classified that way.
- Observable lip/action facts may be described when visually relevant, but do not turn them into analytical conclusions.

VISIBLE SUBJECT PRESERVATION
- Every defined ENTITY <Subject N> that is visibly present in the target shot MUST appear with its exact <Subject N> label at its first clear visual appearance in shot1_caption.
- Do not replace a defined visible entity only with generic prose such as "a woman", "a man", "the doctor", "the church", etc. After first establishment, natural pronouns/descriptions may be used.
- A visible entity remains visually present even when profile-view, back-facing, partially occluded, face/mouth occluded, or silent. Do not drop a visible Subject merely because it is not the speaker.
- Attribute-only Subjects do not need mechanical repetition unless that referenced attribute is specifically being cited.
- Visual presence does NOT by itself assign (Sx).
- If the AV reasonably establishes that a visible defined Subject is the speaker, write <Subject N> (Sx) at the corresponding vocal event.
- If a defined Subject is visible but the vocal source is genuinely offscreen/other, preserve the visible <Subject N> in the visual prose and describe the vocal source separately.
- Back/profile/occluded mouth is still visible presence and is not equivalent to offscreen.
- Do not downgrade a clearly onscreen speaker merely because mouth motion is subtle/not assessable when the full AV supports that binding. Concrete offscreen evidence can still override.
- Do NOT create <Audio N> merely because target audio/dialogue/music exists. Only use an allowed <Audio N> when the input/reference contract actually contains that Audio reference asset.

SPEAKER MARKERS
- Number stable (Sx) by final groups' first transcribed appearance. Referenced visible speakers use <Subject N> (Sx); unbound sources use a natural semantic source plus (Sx). No fixed says clause or immediate adjacency is required.
- (Sx) IDs identify actual vocal sources/events. They are NOT Subject numbers. Do NOT attach (Sx) merely because a person is visible, introduced, listening or present. A silent Subject must not consume a speaker ID. Number by first ACTUAL vocal appearance: first actual vocal source -> S1; second distinct vocal source -> S2.
- Example: <Subject 1> is visible but silent; <Subject 2> speaks first. Correct: "<Subject 1> watches quietly. <Subject 2> (S1) says, <d>...</d>". Wrong: "<Subject 1> (S1) ... <Subject 2> (S2) says ...".
- The first dialogue event from a vocal source must establish its (Sx). When the speaker changes, the new vocal source MUST be explicitly introduced with its own valid (Sx) before that <d>.
- Consecutive dialogue blocks from the SAME continuing speaker may inherit the already established (Sx); mechanical repetition of the same marker is not required. If speaker continuity is uncertain, repeat the explicit (Sx).
- Allowed same-speaker continuity: "A man (S1) says, <d>[English] First.</d> He continues, <d>[English] Second.</d>".
- Required transition: "A man (S1) says, <d>[English] First.</d> A woman (S2) replies, <d>[English] Second.</d>".
- overall_soundscape: Write only non-musical, non-dialogue audible ambience/SFX. Preserve positive auxiliary observations and useful acoustic detail unless original AV provides a concrete factual contradiction; correct a mistaken source without deleting the sound. Exclude spoken dialogue, singing, and all music. Use "N/A" only when no eligible content remains, not merely because a positive candidate is weak or masked.
- non_diegetic_music: Write audience-facing background music/score, preserving clearly reported auxiliary music and harmless acoustic wording by default. Use original AV to correct factual errors and determine placement, not to re-prove every detail. If music is diegetic/in-scene, describe it naturally at its observed chronological position in shot1_caption instead and output "N/A" here. Discard music only for concrete factual falsehood or clear leakage/artifact, not merely weak original-mix audibility.
- The official ICL is an official H3 writing/field-boundary demonstration, not the response-schema definition. Follow the actual supplied schema and official six-section Ref2VA semantics."""


def _official_detailed_description_icl_messages() -> list[dict[str, str]]:
    """Extract official single-shot writing examples without rewriting their prose."""
    path = Path(__file__).resolve().parents[2] / "docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md"
    guide = path.read_text(encoding="utf-8")
    section = guide.split("## 7. Complete Example", 1)[1]
    example = section.split("```text\n", 1)[1].split("\n```", 1)[0]
    detailed, audio_sections = example.split("detailed_description:\n", 1)[1].split(
        "\n\noverall_soundscape:\n", 1,
    )
    soundscape, music = audio_sections.split("\n\nnon_diegetic_music:\n", 1)
    opening, shot_body = detailed.split("[Shot 1] ", 1)
    shot_body = shot_body.split("\n[Shot 2]", 1)[0]
    messages = [
        {"role": "user", "content": (
            "This is an official H3 writing/field-boundary demonstration from the "
            "MiniMax-H3 Ref2VA Complete Example, not the response-schema definition. "
            "It shows the global style opening, only the first shot body, and the "
            "separate overall_soundscape and non_diegetic_music sections."
        )},
        {"role": "assistant", "content": _compact_json({
            "style_opening": opening.strip(),
            "shot1_caption": shot_body,
            "overall_soundscape": soundscape.strip(),
            "non_diegetic_music": music.strip(),
        })},
    ]
    base_path = path.with_name("VIDEO_PROMPT_WRITING_GUIDE_base_en.md")
    base_guide = base_path.read_text(encoding="utf-8")
    for case in ("Case 2: I2VA", "Case 3: FL2VA", "Case 4: L2VA"):
        section = base_guide.split(f"### {case}\n", 1)[1].split("\n### ", 1)[0]
        example = section.split("```text\n", 1)[1].split("\n```", 1)[0]
        body, audio_sections = example.split(
            "integrated_multimodal_description: [Shot 1] ", 1,
        )[1].split("\n\noverall_soundscape: ", 1)
        soundscape, music = audio_sections.split("\n\nnon_diegetic_music: ", 1)
        messages.extend([
            {"role": "user", "content": (
                f"Official MiniMax-H3 {case} writing/field-boundary demonstration, "
                "not the response-schema definition. Observe the single-shot visual "
                "progression and separate soundscape/music fields. Use only references "
                "allowed by the actual task, not these example assets."
            )},
            {"role": "assistant", "content": _compact_json({
                "shot1_caption": body,
                "overall_soundscape": soundscape,
                "non_diegetic_music": music,
            })},
        ])
    return messages


def _value(value: object, name: str) -> object | None:
    return value.get(name) if isinstance(value, dict) else getattr(value, name, None)


def _token(value: object, *names: str) -> int | None:
    current: object | None = value
    for name in names:
        if current is None:
            return None
        current = _value(current, name)
    return current if isinstance(current, int) and current >= 0 else None


def _completion_diagnostic(
    completion: object,
    choice: object,
    *,
    modality: Literal[
        "target_video_with_embedded_audio",
        "target_video_plus_canonical_full_audio_fallback",
        "full_av_recheck_embedded_audio",
        "full_av_recheck_with_canonical_audio",
        "target_av_with_auxiliary_raw_audio",
        "auxiliary_audio_only",
        "speaker_marker_text_only",
    ],
    http_attempt_count: int,
    thinking: Literal["disabled", "enabled"] = "disabled",
) -> MimoCompletionDiagnostic:
    usage = _value(completion, "usage")
    prompt_details = _value(usage, "prompt_tokens_details")
    completion_details = _value(usage, "completion_tokens_details")
    warnings = []
    if prompt_details is None:
        warnings.append("prompt_tokens_details_unavailable")
    if completion_details is None:
        warnings.append("completion_tokens_details_unavailable")
    cached_tokens = _token(prompt_details, "cached_tokens")
    if cached_tokens is None:
        cached_tokens = _token(usage, "prompt_tokens_details", "cached_tokens")
    reasoning_tokens = _token(completion_details, "reasoning_tokens")
    if reasoning_tokens is None:
        reasoning_tokens = _token(
            usage, "completion_tokens_details", "reasoning_tokens"
        )
    if thinking == "disabled" and reasoning_tokens is not None and reasoning_tokens > 0:
        warnings.append("reasoning_tokens_nonzero_under_disabled_thinking")
    return MimoCompletionDiagnostic(
        input_modality=modality,
        finish_reason=_value(choice, "finish_reason"),
        usage=MimoUsage(
            prompt_tokens=_token(usage, "prompt_tokens"),
            completion_tokens=_token(usage, "completion_tokens"),
            total_tokens=_token(usage, "total_tokens"),
            image_tokens=_token(prompt_details, "image_tokens"),
            video_tokens=_token(prompt_details, "video_tokens"),
            audio_tokens=_token(prompt_details, "audio_tokens"),
            cached_tokens=cached_tokens,
            reasoning_tokens=reasoning_tokens,
        ),
        http_attempt_count=http_attempt_count,
        warnings=warnings,
    )


def _validate_finish_reason(diagnostic: MimoCompletionDiagnostic) -> None:
    value = diagnostic.finish_reason
    if value is None:
        diagnostic.warnings.append("finish_reason_unavailable")
        return
    normalized = value.strip().lower().replace("-", "_")
    if normalized == "stop":
        return
    if normalized in _TRUNCATED_FINISH_REASONS or "token_limit" in normalized:
        raise MimoBackendFailure(
            code="mimo_output_truncated",
            reason="MiMo AV response ended at the token limit",
        )
    raise MimoBackendFailure(
        code="mimo_incomplete_finish_reason",
        reason=f"MiMo AV response did not finish normally: {value}",
    )


def _validate_av_observation_usage(
    diagnostic: MimoCompletionDiagnostic,
    *,
    require_explicit_audio: bool,
) -> None:
    if diagnostic.usage.image_tokens == 0:
        raise MimoBackendFailure(
            code="mimo_reference_images_not_observed",
            reason="MiMo reported zero frozen-reference image tokens",
        )
    if diagnostic.usage.image_tokens is None:
        diagnostic.warnings.append("image_tokens_unavailable")
    if diagnostic.usage.video_tokens == 0:
        raise MimoBackendFailure(
            code="mimo_target_video_not_observed",
            reason="MiMo reported zero target-video tokens",
        )
    if diagnostic.usage.video_tokens is None:
        diagnostic.warnings.append("video_tokens_unavailable")
    if require_explicit_audio:
        if diagnostic.usage.audio_tokens == 0:
            raise MimoBackendFailure(
                code="mimo_target_audio_not_observed",
                reason="MiMo reported zero explicit canonical-audio tokens",
            )
        if diagnostic.usage.audio_tokens is None:
            diagnostic.warnings.append(
                "audio_tokens_unavailable_after_explicit_fallback"
            )
    elif diagnostic.usage.audio_tokens is None:
        diagnostic.warnings.append("audio_tokens_unavailable")


class _MimoHTTPAttemptsExhausted(RuntimeError):
    def __init__(self, original: Exception, *, attempts: int, retries: int) -> None:
        super().__init__(f"{type(original).__name__}: {original}")
        self.original = original
        self.attempts = attempts
        self.retries = retries


class _MimoResponseContractError(RuntimeError):
    def __init__(self, original: Exception, *, attempts: int, retries: int) -> None:
        super().__init__(f"{type(original).__name__}: {original}")
        self.original = original
        self.attempts = attempts
        self.retries = retries


_DIALOGUE = re.compile(r"<d>(.*?)</d>", re.DOTALL)
_REFERENCE_LABEL = re.compile(r"<(?:Subject|Picture|Audio|Video) \d+>")
_SPEAKER_LABEL = re.compile(r"\(S(\d+)\)")

SPEAKER_MARKER_POLISH_PROMPT = """You are correcting ONLY speaker-marker projection in an existing H3 shot caption.
AUTHORITATIVE SPEAKER FACTS own speaker identity and Sx mapping. Do NOT re-decide who is speaking, Subject identity, binding, or on/offscreen presentation.
Do NOT invent a speaker or add any Sx absent from AUTHORITATIVE SPEAKER FACTS.
Do NOT change, translate, split, merge, reorder, or rewrite any <d> dialogue text, including its language marker.
Do NOT change Subjects, Pictures, visual description, actions, chronology, sound description, or any other caption content.
Do NOT invent <Audio N>, <Subject N>, <Picture N>, or other references.
Only add, remove, or replace (Sx) markers where necessary.
The first dialogue event for a speaker should establish its correct (Sx).
Consecutive dialogue from the same continuing speaker may inherit the established marker naturally; mechanical repetition is not required.
A speaker transition must use the correct new (Sx). Make the smallest possible edit.
Treat all existing prose as immutable except for (Sx) markers.
If AUTHORITATIVE SPEAKER FACTS contain exactly one distinct speaker_id, then any dialogue event missing its speaker marker is NOT ambiguous. Add that authoritative (Sx) marker at the smallest natural location and set needs_review=false.
Do not treat uncertainty about Subject identity, visual binding, pronouns, or on/offscreen presentation as uncertainty about Sx projection. Speaker identity and Sx mapping are already fixed by AUTHORITATIVE SPEAKER FACTS.
needs_review=true is only for cases where the supplied authoritative speaker sequence itself does not uniquely determine which existing dialogue block should receive which Sx. In that case return the caption unchanged.
speaker_marker_projection_issues only explains why this check was requested; it supplies no new speaker facts.
AUTHORITATIVE: [{"speaker_id":"S1","text":"..."}]
EXISTING: He says, <d>[Chinese] ...</d>
CORRECT: He (S1) says, <d>[Chinese] ...</d>
needs_review=false
Keep "He says" and all other prose unchanged; only the marker is added.
DIALOGUE_MARKER_TARGETS gives the authoritative marker for each existing <d> block by 1-based chronological dialogue index.
For every listed target, ensure that dialogue block is introduced by the listed (Sx): if missing, add it; if a different known Sx is present, replace it. Do not question or infer the mapping. Do not modify any non-Sx text.
The required (Sx) marker MUST appear in the natural lead-in immediately before its corresponding <d> block.
Never place an (Sx) marker after the closing </d> tag of the dialogue it labels.
The marker must be visible to the reader before that dialogue begins.
When adding a missing marker, insert it into the existing speaker lead-in without changing any other prose.
WRONG: <Subject 1> responds, <d>[Chinese] b</d> (S2)
CORRECT: <Subject 1> (S2) responds, <d>[Chinese] b</d>
WRONG: She responds, <d>[Chinese] b</d> (S2)
CORRECT: She (S2) responds, <d>[Chinese] b</d>
If DIALOGUE_MARKER_TARGETS is supplied, it is already uniquely aligned. Set needs_review=false after applying the required marker-only edits.
DIALOGUE_MARKER_TARGETS: [{"dialogue_index":1,"speaker_id":"S1"},{"dialogue_index":2,"speaker_id":"S2"}]
EXISTING: An offscreen male voice (S1) says, <d>[Chinese] a</d> <Subject 1> replies, <d>[Chinese] b</d>
CORRECT: An offscreen male voice (S1) says, <d>[Chinese] a</d> <Subject 1> (S2) replies, <d>[Chinese] b</d>
needs_review=false
Return JSON only with exactly shot1_caption (string) and needs_review (boolean)."""

_SPEAKER_MARKER_POLISH_ISSUES = frozenset({
    "direct_single_speaker_marker_missing",
    "direct_dialogue_speaker_marker_missing",
    "direct_dialogue_speaker_marker_mismatch",
})


def _speaker_marker_only_edit(original: str, candidate: str) -> bool:
    if _DIALOGUE.findall(original) != _DIALOGUE.findall(candidate):
        return False
    return (
        " ".join(_SPEAKER_LABEL.sub("", original).split())
        == " ".join(_SPEAKER_LABEL.sub("", candidate).split())
    )


def protect_direct_dialogue(
    text: str,
    speech: list[dict[str, Any]],
    *,
    allowed_labels: set[str],
) -> tuple[str, list[ValidationIssue], list[str]]:
    """Check generated vocal-event syntax only; never align or rewrite ASR text."""
    issues = []
    unknown = set(_REFERENCE_LABEL.findall(text)) - allowed_labels
    if unknown:
        issues.append(
            ValidationIssue(
                "direct_unknown_reference", "shot1_caption", str(sorted(unknown))
            )
        )
    known = {item["speaker_id"] for item in speech}
    actual = {"S" + number for number in _SPEAKER_LABEL.findall(text)}
    if actual - known:
        issues.append(
            ValidationIssue(
                "direct_unknown_speaker", "shot1_caption", str(sorted(actual - known))
            )
        )
    blocks = list(_DIALOGUE.finditer(text))
    if (
        text.count("<d>") != len(blocks)
        or text.count("</d>") != len(blocks)
        or any("<d>" in block.group(1) for block in blocks)
    ):
        issues.append(
            ValidationIssue(
                "direct_dialogue_format",
                "shot1_caption",
                "dialogue tags must be paired and non-nested",
            )
        )
    previous_end = 0
    speech_sequence = [item["speaker_id"] for item in speech]
    aligned_counts = len(blocks) == len(speech_sequence)
    for index, block in enumerate(blocks):
        lead_in = text[previous_end : block.start()]
        lead_markers = {speaker for speaker in known if f"({speaker})" in lead_in}
        # Equal counts permit speaker checks; they are never an inventory gate.
        same_speaker_continuation = (
            aligned_counts
            and index > 0
            and speech_sequence[index] == speech_sequence[index - 1]
        )
        if lead_markers:
            if aligned_counts and speech_sequence[index] not in lead_markers:
                issues.append(
                    ValidationIssue(
                        "direct_dialogue_speaker_marker_mismatch",
                        f"dialogue_{index + 1}",
                        "explicit speaker marker disagrees with authoritative chronological speaker",
                    )
                )
        elif not same_speaker_continuation:
            issues.append(
                ValidationIssue(
                    (
                        "direct_single_speaker_marker_missing" if len(known) == 1
                        else "direct_dialogue_speaker_marker_missing"
                    ),
                    f"dialogue_{index + 1}",
                    "generated vocal event needs a valid speaker lead-in",
                )
            )
        if re.match(r"^\[[^\]\r\n]+\]", block.group(1)) is None:
            issues.append(
                ValidationIssue(
                    "direct_dialogue_language_missing",
                    f"dialogue_{index + 1}",
                    "dialogue requires a language marker",
                )
            )
        previous_end = block.end()
    if "[[" in text or "]]" in text:
        issues.append(
            ValidationIssue(
                "direct_internal_syntax",
                "shot1_caption",
                "internal placeholders are not H3",
            )
        )
    if re.search(r"\[Shot\s+\d+\]", text):
        issues.append(
            ValidationIssue(
                "direct_shot_marker", "shot1_caption", "shot markers are pipeline-owned"
            )
        )
    return (
        text,
        [issue for issue in issues if issue.code not in _REVIEW_ONLY_ISSUES],
        [issue.code for issue in issues if issue.code in _REVIEW_ONLY_ISSUES],
    )


def direct_speech_facts(annotation: MimoAVAnnotationDraft, segments: list[Any]) -> list[dict[str, Any]]:
    audio = {item.segment_id: item for item in annotation.audio_observation.segment_decisions}
    grounding = {item.segment_id: item for item in annotation.segment_decisions}
    speaker_ids: dict[str, str] = {}
    facts = []
    for segment in segments:
        if segment.asr_status != "transcribed":
            continue
        decision, bound = audio.get(segment.segment_id), grounding.get(segment.segment_id)
        if decision is None or bound is None:
            continue  # the authoritative inventory validator reports this mismatch
        resolved = decision.resolution == "resolved" or bound.binding_status == "visible_entity"
        group = decision.primary_speaker_group if resolved else f"fallback__{segment.source_speaker_cluster_id}"
        if group not in speaker_ids:
            speaker_ids[group] = f"S{len(speaker_ids) + 1}"
        facts.append({
            "segment_id": segment.segment_id, "language": segment.asr_language,
            "text": segment.asr_text or "", "speaker_id": speaker_ids[group],
        })
    return facts


_REVIEW_ONLY_ISSUES = frozenset({
    "direct_single_speaker_marker_missing", "stage_a_av_articulation_contradiction",
    "visible_entity_requires_resolved_audio", "onscreen_grounding_incomplete",
    "visible_entity_requires_confirmed_onscreen_speech",
    "onscreen_speech_requires_reliable_visible_speaker_evidence",
    "missing_transcribed_segment_delivery", "non_transcribed_segment_delivery",
    "speaker_voice_profile_inventory_mismatch", "speaker_voice_profile_contains_identity_claim",
    "audio_event_exceeds_target", "draft_contains_internal_annotation_syntax",
    "draft_contains_pipeline_owned_syntax", "unassigned_picture_keyframe_role",
    "subject_definition_contains_audio_profile", "attribute_subject_redefines_owner_entity",
    "draft_contains_authoritative_transcript", "audio_semantics_contains_authoritative_transcript",
    "non_diegetic_music_leaked_into_soundscape", "non_diegetic_music_misclassified_as_soundscape_event",
    "overall_soundscape_contains_music_or_speech", "soundscape_event_category_contamination",
})


def validate_annotation(
    annotation: MimoAVAnnotationDraft,
    *,
    segment_ids: list[str],
    segment_intervals: dict[str, tuple[float, float]],
    transcribed_segment_ids: list[str],
    authoritative_transcripts: list[str],
    allowed_entity_ids: set[str],
    allowed_reference_labels: set[str],
    reference_subjects: list[Any],
    target_duration_seconds: float,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    audio_decisions = annotation.audio_observation.segment_decisions
    groundings = annotation.av_grounding.segment_groundings
    visual_views = annotation.visual_observation.segment_views
    if [item.segment_id for item in audio_decisions] != segment_ids:
        issues.append(
            ValidationIssue(
                "segment_inventory_mismatch",
                "audio_observation.segment_decisions",
                "Audio decisions must exactly follow supplied chronological IDs",
            )
        )
    if [item.segment_id for item in groundings] != segment_ids:
        issues.append(
            ValidationIssue(
                "segment_inventory_mismatch",
                "av_grounding.segment_groundings",
                "AV groundings must exactly follow supplied chronological IDs",
            )
        )
    if [item.segment_id for item in visual_views] != segment_ids:
        issues.append(
            ValidationIssue(
                "visual_segment_inventory_mismatch",
                "visual_observation.segment_views",
                "visual segment views must exactly follow supplied chronological IDs",
            )
        )
    audio_by_id = {item.segment_id: item for item in audio_decisions}
    view_by_id = {item.segment_id: item for item in visual_views}
    for view in visual_views:
        unknown_visual_entities = sorted(
            set(view.visible_entity_ids) - allowed_entity_ids
        )
        if unknown_visual_entities:
            issues.append(
                ValidationIssue(
                    "unknown_visual_entity",
                    view.segment_id,
                    f"unknown supplied visual entities: {unknown_visual_entities}",
                )
            )
    for decision in groundings:
        audio = audio_by_id.get(decision.segment_id)
        view = view_by_id.get(decision.segment_id)
        if audio is None or decision.primary_speaker_group != audio.primary_speaker_group:
            issues.append(
                ValidationIssue(
                    "av_audio_speaker_group_mismatch",
                    decision.segment_id,
                    "AV grounding must preserve the Stage B primary speaker group",
                )
            )
        if (
            decision.binding_status == "visible_entity"
            and not _audio_allows_visible_entity_resolution(audio)
        ):
            issues.append(
                ValidationIssue(
                    "visible_entity_requires_resolved_audio",
                    decision.segment_id,
                    "visible entity grounding requires a resolved Stage B speaker or "
                    "one AV-resolvable single-speaker refinement group",
                )
            )
        if _has_stage_a_c_articulation_contradiction(
            decision,
            view,
            allowed_entity_ids=allowed_entity_ids,
        ):
            issues.append(
                ValidationIssue(
                    "stage_a_av_articulation_contradiction",
                    decision.segment_id,
                    "Stage A articulation and Stage C lip-motion evidence disagree",
                )
            )
        if _has_incomplete_onscreen_grounding_contradiction(
            decision,
            view,
            allowed_entity_ids=allowed_entity_ids,
        ):
            issues.append(
                ValidationIssue(
                    "onscreen_grounding_incomplete",
                    decision.segment_id,
                    "onscreen speech with plausible Stage A evidence requires a "
                    "complete visible entity grounding",
                )
            )
        if (
            decision.binding_status == "visible_entity"
            and decision.entity_id is not None
            and (view is None or decision.entity_id not in view.visible_entity_ids)
        ):
            issues.append(
                ValidationIssue(
                    "visible_entity_absent_from_visual_segment",
                    decision.segment_id,
                    "visible_entity must be present in the exact Stage A segment view",
                )
            )
        if decision.binding_status == "visible_entity" and (
            decision.speech_presentation != "onscreen_spoken"
            or decision.entity_id is None
            or not _has_grounded_onscreen_evidence(decision, view)
        ):
            issues.append(
                ValidationIssue(
                    "visible_entity_requires_confirmed_onscreen_speech",
                    decision.segment_id,
                    "visible_entity requires onscreen_spoken, entity_id, and reliable onscreen speaker evidence",
                )
            )
        if decision.speech_presentation == "onscreen_spoken" and not (
            _has_grounded_onscreen_evidence(decision, view)
        ):
            issues.append(
                ValidationIssue(
                    "onscreen_speech_requires_reliable_visible_speaker_evidence",
                    decision.segment_id,
                    "onscreen_spoken requires visible articulation or mouth-occluded visible-speaker continuity",
                )
            )
        presentation_contradictions = sorted(
            set(decision.evidence_codes) & _VISIBLE_PRESENTATION_CONTRADICTIONS
        )
        if (
            decision.binding_status == "visible_entity"
            or decision.speech_presentation == "onscreen_spoken"
        ) and presentation_contradictions:
            issues.append(
                ValidationIssue(
                    "visible_speaker_evidence_presentation_contradiction",
                    decision.segment_id,
                    "visible/onscreen speech conflicts with explicit non-onscreen "
                    f"evidence: {presentation_contradictions}",
                )
            )
        if (
            decision.speech_presentation == "onscreen_spoken"
            and decision.binding_status == "offscreen"
        ):
            issues.append(
                ValidationIssue(
                    "onscreen_speech_cannot_be_offscreen",
                    decision.segment_id,
                    "onscreen_spoken requires visible_entity, no_reliable_entity, or uncertain",
                )
            )
        if decision.speech_presentation != "onscreen_spoken" and (
            decision.binding_status == "visible_entity" or decision.entity_id is not None
        ):
            issues.append(
                ValidationIssue(
                    "non_onscreen_speech_claims_visible_entity",
                    decision.segment_id,
                    "non-onscreen speech cannot publish a visible entity",
                )
            )
        if decision.speech_presentation == "offscreen_spoken" and not (
            decision.binding_status == "offscreen"
            and decision.entity_id is None
            and "offscreen_audio" in decision.evidence_codes
        ):
            issues.append(
                ValidationIssue(
                    "offscreen_presentation_evidence_mismatch",
                    decision.segment_id,
                    "offscreen_spoken requires offscreen, null entity, and offscreen_audio",
                )
            )
        if decision.speech_presentation == "voice_over" and not (
            decision.entity_id is None
            and "voice_over_context" in decision.evidence_codes
        ):
            issues.append(
                ValidationIssue(
                    "voice_over_evidence_mismatch",
                    decision.segment_id,
                    "voice_over requires null entity and voice_over_context",
                )
            )
        if decision.speech_presentation == "device_playback" and not (
            decision.entity_id is None
            and "device_playback_context" in decision.evidence_codes
        ):
            issues.append(
                ValidationIssue(
                    "device_playback_evidence_mismatch",
                    decision.segment_id,
                    "device_playback requires null entity and device_playback_context",
                )
            )
        if decision.speech_presentation == "message_voice_over" and not (
            decision.entity_id is None
            and "message_text_alignment" in decision.evidence_codes
            and "voice_over_context" in decision.evidence_codes
        ):
            issues.append(
                ValidationIssue(
                    "message_voice_over_evidence_mismatch",
                    decision.segment_id,
                    "message_voice_over requires explicit message and voice-over evidence",
                )
            )
        if decision.entity_id is not None and decision.entity_id not in allowed_entity_ids:
            issues.append(
                ValidationIssue(
                    "unknown_entity",
                    decision.segment_id,
                    f"unknown supplied entity: {decision.entity_id}",
                )
            )
    warning_segment_ids = [item.segment_id for item in annotation.warnings]
    for segment_id in warning_segment_ids:
        if segment_id not in segment_ids:
            issues.append(
                ValidationIssue(
                    "warning_unknown_segment",
                    "warnings",
                    f"unknown warning segment: {segment_id}",
                )
            )
    first_groups: list[str] = []
    for decision in audio_decisions:
        group = decision.primary_speaker_group
        if group is not None and group not in first_groups:
            first_groups.append(group)
    expected_groups = [f"g{index}" for index in range(1, len(first_groups) + 1)]
    if first_groups != expected_groups:
        issues.append(
            ValidationIssue(
                "non_contiguous_speaker_groups",
                "audio_observation.segment_decisions",
                "speaker groups must be contiguous by first chronological appearance",
            )
        )
    entities_by_group: dict[str, set[str]] = {}
    groups_by_entity: dict[str, set[str]] = {}
    for decision in groundings:
        if (
            decision.binding_status == "visible_entity"
            and decision.primary_speaker_group is not None
            and decision.entity_id is not None
        ):
            entities_by_group.setdefault(decision.primary_speaker_group, set()).add(
                decision.entity_id
            )
            groups_by_entity.setdefault(decision.entity_id, set()).add(
                decision.primary_speaker_group
            )
    for group, entity_ids in entities_by_group.items():
        if len(entity_ids) > 1:
            issues.append(
                ValidationIssue(
                    "speaker_group_entity_contradiction",
                    "av_grounding.segment_groundings",
                    f"{group} maps to multiple visible entities: {sorted(entity_ids)}",
                )
            )
    for entity_id, groups in groups_by_entity.items():
        if len(groups) > 1:
            issues.append(
                ValidationIssue(
                    "visible_entity_speaker_group_contradiction",
                    "av_grounding.segment_groundings",
                    f"{entity_id} maps to multiple resolved groups: {sorted(groups)}",
                )
            )
    transcribed_segment_set = set(transcribed_segment_ids)
    for decision in audio_decisions:
        if decision.segment_id in transcribed_segment_set:
            if decision.delivery_style is None:
                issues.append(
                    ValidationIssue(
                        "missing_transcribed_segment_delivery",
                        decision.segment_id,
                        "transcribed segment requires audible delivery_style",
                    )
                )
        elif decision.delivery_style is not None:
            issues.append(
                ValidationIssue(
                    "non_transcribed_segment_delivery",
                    decision.segment_id,
                    "non-transcribed segment requires delivery_style=null",
                )
            )
    required_profile_groups = _required_voice_profile_groups(
        audio_decisions,
        transcribed_segment_ids=transcribed_segment_set,
    )
    actual_profile_groups = [
        profile.speaker_group for profile in annotation.speaker_voice_profiles
    ]
    if actual_profile_groups != required_profile_groups or len(
        actual_profile_groups
    ) != len(set(actual_profile_groups)):
        issues.append(
            ValidationIssue(
                "speaker_voice_profile_inventory_mismatch",
                "speaker_voice_profiles",
                "voice profiles must exactly follow resolved transcribed speaker groups",
            )
        )
    for profile in annotation.speaker_voice_profiles:
        if (
            profile.voice_characteristics is not None
            and _VOICE_IDENTITY_PROFILE.search(profile.voice_characteristics)
        ):
            issues.append(
                ValidationIssue(
                    "speaker_voice_profile_contains_identity_claim",
                    "speaker_voice_profiles",
                    f"{profile.speaker_group} voice profile contains identity wording",
                )
            )
    semantics_draft = annotation.h3_semantics
    prose_parts = [
        block.text
        for block in annotation.visual_observation.visual_blocks
    ]
    non_shot_text = "\n".join(
        (
            *(item.render() for item in semantics_draft.subject_definitions),
            semantics_draft.summary,
            *(item.render() for item in semantics_draft.visual_retention_analysis),
        )
    )
    if "[[" in non_shot_text or "]]" in non_shot_text:
        issues.append(
            ValidationIssue(
                "direct_internal_syntax",
                "h3_semantics",
                "free-form internal timeline syntax is forbidden",
            )
        )
    draft_text = "\n".join((*prose_parts, non_shot_text))
    if _INTERNAL_ANNOTATION_SYNTAX.search(draft_text):
        issues.append(
            ValidationIssue(
                "draft_contains_internal_annotation_syntax",
                "h3_semantics_or_visual_observation",
                "internal staged field names or IDs cannot enter final H3 prose",
            )
        )
    if _PIPELINE_OWNED.search(draft_text):
        issues.append(
            ValidationIssue(
                "draft_contains_pipeline_owned_syntax",
                "h3_semantics_or_visual_observation",
                "MiMo draft contains pipeline-owned H3 syntax",
            )
        )
    if _KEYFRAME_ROLE.search(draft_text):
        issues.append(
            ValidationIssue(
                "unassigned_picture_keyframe_role",
                "h3_semantics_or_visual_observation",
                "Pictures are content references, not frame anchors",
            )
        )
    unknown_labels = {
        match.group(0)
        for match in _PICTURE_OR_SUBJECT.finditer(draft_text)
        if match.group(0) not in allowed_reference_labels
    }
    if unknown_labels:
        issues.append(
            ValidationIssue(
                "draft_contains_unknown_reference",
                "h3_semantics_or_visual_observation",
                str(sorted(unknown_labels)),
            )
        )
    supplied_subject_labels = {str(item.subject_label) for item in reference_subjects}
    retention_subject_labels = [
        item.subject_label for item in semantics_draft.visual_retention_analysis
    ]
    unknown_retention_subjects = sorted(
        set(retention_subject_labels) - supplied_subject_labels
    )
    if unknown_retention_subjects:
        issues.append(
            ValidationIssue(
                "unknown_subject_retention",
                "h3_semantics.visual_retention_analysis",
                str(unknown_retention_subjects),
            )
        )
    if (
        len(semantics_draft.visual_retention_analysis) != len(reference_subjects)
        or len(retention_subject_labels) != len(set(retention_subject_labels))
        or set(retention_subject_labels) != supplied_subject_labels
    ):
        issues.append(
            ValidationIssue(
                "subject_retention_contract_mismatch",
                "h3_semantics.visual_retention_analysis",
                "retention rows must exactly cover supplied Subjects",
            )
        )
    if len(semantics_draft.subject_definitions) != len(reference_subjects):
        issues.append(
            ValidationIssue(
                "subject_definition_contract_mismatch",
                "h3_semantics.subject_definitions",
                "definition rows must exactly cover supplied Subjects",
            )
        )
    expected_subject_labels = [str(item.subject_label) for item in reference_subjects]
    actual_subject_labels = [
        item.subject_label for item in semantics_draft.subject_definitions
    ]
    if actual_subject_labels != expected_subject_labels:
        issues.append(
            ValidationIssue(
                "subject_definition_contract_mismatch",
                "h3_semantics.subject_definitions",
                "definition rows must follow the exact supplied Subject order",
            )
        )
    for subject in reference_subjects:
        subject_label = str(subject.subject_label)
        definitions = [
            item
            for item in semantics_draft.subject_definitions
            if item.subject_label == subject_label
        ]
        if len(definitions) != 1:
            issues.append(
                ValidationIssue(
                    "subject_definition_contract_mismatch",
                    "h3_semantics.subject_definitions",
                    f"{subject_label} requires exactly one visual description row",
                )
            )
        if len(definitions) == 1 and _SUBJECT_AUDIO_PROFILE.search(
            definitions[0].description
        ):
            issues.append(
                ValidationIssue(
                    "subject_definition_contains_audio_profile",
                    "h3_semantics.subject_definitions",
                    f"{subject_label} definition must remain visual-only",
                )
            )
        if (
            len(definitions) == 1
            and subject.kind == "attribute"
            and subject.attribute_type in _HUMAN_ATTRIBUTE_TYPES
            and _STANDALONE_HUMAN_SUBJECT_LEAD.search(definitions[0].description)
        ):
            issues.append(
                ValidationIssue(
                    "attribute_subject_redefines_owner_entity",
                    subject_label,
                    "attribute Subject must describe only the referenced attribute, "
                    "not a standalone person",
                )
            )
    model_owned_text = _normalized_text(
        "\n".join(
            (
                *(item.render() for item in semantics_draft.subject_definitions),
                semantics_draft.summary,
                *(item.render() for item in semantics_draft.visual_retention_analysis),
                *prose_parts,
            )
        )
    )
    for transcript in authoritative_transcripts:
        normalized_transcript = _normalized_text(transcript)
        if (
            _significant_transcript(transcript)
            and normalized_transcript
            and normalized_transcript in model_owned_text
        ):
            issues.append(
                ValidationIssue(
                    "draft_contains_authoritative_transcript",
                    "h3_semantics_or_visual_observation",
                    "model-authored draft copies exact authoritative ASR text",
                )
            )
            break
    model_owned_audio_fields = [
        *(item.delivery_style for item in audio_decisions if item.delivery_style is not None),
        *(item.voice_characteristics for item in annotation.speaker_voice_profiles
          if item.voice_characteristics is not None),
    ]
    normalized_audio_fields = [
        _normalized_text(value) for value in model_owned_audio_fields
    ]
    for transcript in authoritative_transcripts:
        normalized_transcript = _normalized_text(transcript)
        if (
            _significant_transcript(transcript)
            and normalized_transcript
            and any(
                normalized_transcript in field for field in normalized_audio_fields
            )
        ):
            issues.append(
                ValidationIssue(
                    "audio_semantics_contains_authoritative_transcript",
                    "audio_semantics",
                    "model-authored Audio semantics copies exact authoritative ASR text",
                )
            )
            break
    return issues


_CONSERVATIVE_VISIBLE_SPEAKER_ISSUES = {
    "visible_entity_requires_confirmed_onscreen_speech",
    "onscreen_speech_requires_reliable_visible_speaker_evidence",
}


def _required_voice_profile_groups(
    decisions: list[MimoAudioSegmentDecision],
    *,
    transcribed_segment_ids: set[str],
) -> list[str]:
    required_groups: list[str] = []
    for decision in decisions:
        group = decision.primary_speaker_group
        if (
            decision.resolution == "resolved"
            and decision.segment_id in transcribed_segment_ids
            and group is not None
            and group not in required_groups
        ):
            required_groups.append(group)
    return required_groups


def _normalize_speaker_voice_profiles(
    annotation: MimoAVAnnotationDraft,
    *,
    transcribed_segment_ids: set[str],
    source_groups_by_final_group: dict[str, list[str]] | None = None,
) -> tuple[MimoAVAnnotationDraft, int]:
    required_groups = _required_voice_profile_groups(
        annotation.audio_observation.segment_decisions,
        transcribed_segment_ids=transcribed_segment_ids,
    )
    profiles_by_group: dict[str, MimoSpeakerVoiceProfile] = {}
    for profile in annotation.speaker_voice_profiles:
        if profile.speaker_group in profiles_by_group:
            return annotation, 0
        profiles_by_group[profile.speaker_group] = profile
    normalized_profiles: list[MimoSpeakerVoiceProfile] = []
    for group in required_groups:
        source_groups = (
            source_groups_by_final_group.get(group, [group])
            if source_groups_by_final_group is not None
            else [group]
        )
        canonical_profile = profiles_by_group.get(source_groups[0])
        if canonical_profile is not None:
            voice_characteristics = canonical_profile.voice_characteristics
        else:
            voice_characteristics = next(
                (
                    profile.voice_characteristics
                    for source_group in source_groups[1:]
                    if (profile := profiles_by_group.get(source_group)) is not None
                    and profile.voice_characteristics is not None
                ),
                None,
            )
        normalized_profiles.append(
            MimoSpeakerVoiceProfile(
                speaker_group=group,
                voice_characteristics=voice_characteristics,
            )
        )
    if normalized_profiles == annotation.speaker_voice_profiles:
        return annotation, 0
    payload = annotation.model_dump(mode="python")
    payload["audio_observation"]["speaker_voice_profiles"] = [
        profile.model_dump(mode="python") for profile in normalized_profiles
    ]
    return MimoAVAnnotationDraft.model_validate(payload), 1


def _append_insufficient_evidence(decision: dict[str, Any]) -> None:
    evidence_codes = list(decision["evidence_codes"])
    if "insufficient_evidence" not in evidence_codes and len(evidence_codes) < 8:
        evidence_codes.append("insufficient_evidence")
    decision["evidence_codes"] = evidence_codes


def _remove_unknown_visual_entities(
    annotation: MimoAVAnnotationDraft,
    *,
    allowed_entity_ids: set[str],
) -> tuple[MimoAVAnnotationDraft, int]:
    payload = annotation.model_dump(mode="python")
    correction_count = 0
    for view in payload["visual_observation"]["segment_views"]:
        visible_entity_ids = list(view["visible_entity_ids"])
        known_entity_ids = [
            entity_id
            for entity_id in visible_entity_ids
            if entity_id in allowed_entity_ids
        ]
        correction_count += len(visible_entity_ids) - len(known_entity_ids)
        view["visible_entity_ids"] = known_entity_ids
        view["entity_observations"] = [
            observation
            for observation in view["entity_observations"]
            if observation["entity_id"] in allowed_entity_ids
        ]
    if correction_count == 0:
        return annotation, 0
    return MimoAVAnnotationDraft.model_validate(payload), correction_count


def _supported_non_onscreen_presentation(
    decision: dict[str, Any],
) -> tuple[BindingStatus, SpeechPresentation] | None:
    evidence = set(decision["evidence_codes"])
    presentation = decision["speech_presentation"]
    if presentation == "offscreen_spoken" and "offscreen_audio" in evidence:
        return "offscreen", "offscreen_spoken"
    if presentation == "voice_over" and "voice_over_context" in evidence:
        return "no_reliable_entity", "voice_over"
    if (
        presentation == "message_voice_over"
        and {"message_text_alignment", "voice_over_context"} <= evidence
    ):
        return "no_reliable_entity", "message_voice_over"
    if presentation == "device_playback" and "device_playback_context" in evidence:
        return "no_reliable_entity", "device_playback"
    return None


def _downgrade_unknown_grounding_entities(
    annotation: MimoAVAnnotationDraft,
    *,
    allowed_entity_ids: set[str],
) -> tuple[MimoAVAnnotationDraft, int]:
    payload = annotation.model_dump(mode="python")
    correction_count = 0
    for decision in payload["av_grounding"]["segment_groundings"]:
        entity_id = decision["entity_id"]
        if entity_id is None or entity_id in allowed_entity_ids:
            continue
        supported = _supported_non_onscreen_presentation(decision)
        if supported is None:
            decision["binding_status"] = "no_reliable_entity"
            decision["speech_presentation"] = "uncertain"
            _append_insufficient_evidence(decision)
        else:
            decision["binding_status"], decision["speech_presentation"] = supported
        decision["entity_id"] = None
        decision["confidence"] = "low"
        correction_count += 1
    if correction_count == 0:
        return annotation, 0
    return MimoAVAnnotationDraft.model_validate(payload), correction_count


def _drop_voice_profile_identity_claims(
    annotation: MimoAVAnnotationDraft,
) -> tuple[MimoAVAnnotationDraft, int]:
    payload = annotation.model_dump(mode="python")
    correction_count = 0
    for profile in payload["audio_observation"]["speaker_voice_profiles"]:
        value = profile["voice_characteristics"]
        if isinstance(value, str) and _VOICE_IDENTITY_PROFILE.search(value):
            profile["voice_characteristics"] = None
            correction_count += 1
    if correction_count == 0:
        return annotation, 0
    return MimoAVAnnotationDraft.model_validate(payload), correction_count


def _conservative_visible_speaker_downgrade(
    annotation: MimoAVAnnotationDraft,
    *,
    allowed_entity_ids: set[str],
) -> tuple[MimoAVAnnotationDraft, int]:
    view_by_segment = {
        view.segment_id: view for view in annotation.visual_observation.segment_views
    }
    payload = annotation.model_dump(mode="python")
    correction_count = 0
    for grounding, decision in zip(
        annotation.av_grounding.segment_groundings,
        payload["av_grounding"]["segment_groundings"],
        strict=True,
    ):
        view = view_by_segment.get(grounding.segment_id)
        if _has_stage_a_c_articulation_contradiction(
            grounding,
            view,
            allowed_entity_ids=allowed_entity_ids,
        ) or _has_incomplete_onscreen_grounding_contradiction(
            grounding,
            view,
            allowed_entity_ids=allowed_entity_ids,
        ):
            continue
        claims_visible_speech = (
            grounding.binding_status == "visible_entity"
            or grounding.speech_presentation == "onscreen_spoken"
        )
        if not claims_visible_speech or _has_grounded_onscreen_evidence(
            grounding,
            view,
        ):
            continue
        evidence_codes = list(decision["evidence_codes"])
        if "offscreen_audio" in evidence_codes:
            decision["binding_status"] = "offscreen"
            decision["speech_presentation"] = "offscreen_spoken"
        else:
            decision["binding_status"] = "no_reliable_entity"
            decision["speech_presentation"] = "uncertain"
        decision["entity_id"] = None
        decision["confidence"] = "low"
        decision["evidence_codes"] = evidence_codes
        _append_insufficient_evidence(decision)
        correction_count += 1
    if correction_count == 0:
        return annotation, 0
    return MimoAVAnnotationDraft.model_validate(payload), correction_count


def _conservative_offscreen_presentation_downgrade(
    annotation: MimoAVAnnotationDraft,
) -> tuple[MimoAVAnnotationDraft, int]:
    payload = annotation.model_dump(mode="python")
    correction_count = 0
    for decision in payload["av_grounding"]["segment_groundings"]:
        if decision["speech_presentation"] != "offscreen_spoken":
            continue
        evidence_codes = list(decision["evidence_codes"])
        valid = (
            decision["binding_status"] == "offscreen"
            and decision["entity_id"] is None
            and "offscreen_audio" in evidence_codes
        )
        if valid:
            continue
        decision["entity_id"] = None
        decision["confidence"] = "low"
        if "offscreen_audio" in evidence_codes:
            decision["binding_status"] = "offscreen"
            decision["speech_presentation"] = "offscreen_spoken"
        else:
            decision["binding_status"] = "no_reliable_entity"
            decision["speech_presentation"] = "uncertain"
            _append_insufficient_evidence(decision)
        correction_count += 1
    if correction_count == 0:
        return annotation, 0
    return MimoAVAnnotationDraft.model_validate(payload), correction_count


def _canonicalize_same_visible_entity_speaker_groups(
    annotation: MimoAVAnnotationDraft,
    *,
    segment_ids: list[str],
    allowed_entity_ids: set[str],
) -> tuple[MimoAVAnnotationDraft, dict[str, int], dict[str, list[str]] | None]:
    decisions = annotation.av_grounding.segment_groundings
    if [decision.segment_id for decision in decisions] != segment_ids:
        return annotation, {}, None
    audio_by_id = {
        item.segment_id: item for item in annotation.audio_observation.segment_decisions
    }
    view_by_id = {
        item.segment_id: item for item in annotation.visual_observation.segment_views
    }

    groups_by_entity: dict[str, list[str]] = {}
    visible_entities_by_group: dict[str, set[str]] = {}
    groups_with_uncertain_visible_binding: set[str] = set()
    entities_with_uncertain_visible_binding: set[str] = set()
    reliable_by_entity: dict[str, bool] = {}
    for decision in decisions:
        group = decision.primary_speaker_group
        if (
            group is not None
            and decision.binding_status == "visible_entity"
            and decision.entity_id is not None
        ):
            visible_entities_by_group.setdefault(group, set()).add(decision.entity_id)
            if (
                decision.segment_id not in audio_by_id
                or audio_by_id[decision.segment_id].resolution != "resolved"
            ):
                groups_with_uncertain_visible_binding.add(group)
                entities_with_uncertain_visible_binding.add(decision.entity_id)
        if (
            decision.segment_id not in audio_by_id
            or audio_by_id[decision.segment_id].resolution != "resolved"
            or decision.binding_status != "visible_entity"
            or decision.entity_id is None
            or group is None
        ):
            continue
        entity_id = decision.entity_id
        groups = groups_by_entity.setdefault(entity_id, [])
        if group not in groups:
            groups.append(group)
        reliable_by_entity[entity_id] = reliable_by_entity.get(
            entity_id, True
        ) and _has_grounded_onscreen_evidence(
            decision,
            view_by_id.get(decision.segment_id),
        )

    merged_group_by_group: dict[str, str] = {}
    for entity_id, groups in groups_by_entity.items():
        if (
            entity_id not in allowed_entity_ids
            or entity_id in entities_with_uncertain_visible_binding
            or len(groups) < 2
            or not reliable_by_entity[entity_id]
            or any(group in groups_with_uncertain_visible_binding for group in groups)
            or any(visible_entities_by_group[group] != {entity_id} for group in groups)
        ):
            continue
        canonical_group = groups[0]
        for group in groups[1:]:
            merged_group_by_group[group] = canonical_group
    if not merged_group_by_group:
        return annotation, {}, None

    original_group_order: list[str] = []
    merged_group_order: list[str] = []
    for decision in decisions:
        group = decision.primary_speaker_group
        if group is None:
            continue
        if group not in original_group_order:
            original_group_order.append(group)
        merged_group = merged_group_by_group.get(group, group)
        if merged_group not in merged_group_order:
            merged_group_order.append(merged_group)
    final_group_by_merged_group = {
        group: f"g{index}"
        for index, group in enumerate(merged_group_order, start=1)
    }
    final_group_by_original_group = {
        group: final_group_by_merged_group[merged_group_by_group.get(group, group)]
        for group in original_group_order
    }
    source_groups_by_final_group: dict[str, list[str]] = {}
    for group in original_group_order:
        final_group = final_group_by_original_group[group]
        source_groups_by_final_group.setdefault(final_group, []).append(group)

    payload = annotation.model_dump(mode="python")
    for decision in payload["av_grounding"]["segment_groundings"]:
        group = decision["primary_speaker_group"]
        if group is not None:
            decision["primary_speaker_group"] = final_group_by_original_group[group]
    for decision in payload["audio_observation"]["segment_decisions"]:
        group = decision["primary_speaker_group"]
        if group is not None:
            decision["primary_speaker_group"] = final_group_by_original_group[group]
    correction_counts = {
        "same_visible_entity_speaker_group_merge": len(merged_group_by_group),
    }
    recanonicalized_group_count = sum(
        final_group_by_merged_group[group] != group for group in merged_group_order
    )
    if recanonicalized_group_count:
        correction_counts["speaker_group_id_recanonicalization"] = (
            recanonicalized_group_count
        )
    return (
        MimoAVAnnotationDraft.model_validate(payload),
        correction_counts,
        source_groups_by_final_group,
    )


def _normalize_speaker_annotation(
    annotation: MimoAVAnnotationDraft,
    *,
    segment_ids: list[str],
    transcribed_segment_ids: set[str],
    allowed_entity_ids: set[str],
) -> tuple[MimoAVAnnotationDraft, dict[str, int]]:
    corrections: Counter[str] = Counter()

    annotation, count = _remove_unknown_visual_entities(
        annotation,
        allowed_entity_ids=allowed_entity_ids,
    )
    corrections["unknown_visual_entity_removed"] += count

    annotation, count = _downgrade_unknown_grounding_entities(
        annotation,
        allowed_entity_ids=allowed_entity_ids,
    )
    corrections["unknown_grounding_entity_downgrade"] += count

    annotation, count = _conservative_offscreen_presentation_downgrade(annotation)
    corrections["conservative_offscreen_presentation_downgrade"] += count

    annotation, count = _drop_voice_profile_identity_claims(annotation)
    corrections["speaker_voice_profile_identity_claim_dropped"] += count

    annotation, group_corrections, source_groups = (
        _canonicalize_same_visible_entity_speaker_groups(
            annotation,
            segment_ids=segment_ids,
            allowed_entity_ids=allowed_entity_ids,
        )
    )
    corrections.update(group_corrections)

    annotation, count = _normalize_speaker_voice_profiles(
        annotation,
        transcribed_segment_ids=transcribed_segment_ids,
        source_groups_by_final_group=source_groups,
    )
    corrections["speaker_voice_profile_inventory_normalization"] += count

    return annotation, dict(
        sorted((key, value) for key, value in corrections.items() if value)
    )


class OpenAIMimo25Backend:
    def __init__(
        self,
        config: MimoBackendConfig,
        *,
        client: Any | None = None,
        sleep: Any = time.sleep,
        jitter: Any = random.random,
    ) -> None:
        self.config = config
        self.client = (client.with_options(max_retries=0) if isinstance(client, OpenAI) else client) or OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout_seconds,
            max_retries=0,
        )
        self._sleep = sleep
        self._jitter = jitter
        self._av_request_started = False
        self._av_raw_response = None

    @property
    def provenance(self) -> MimoBackendProvenance:
        return self.config.provenance()

    def _call(self, payload: dict[str, object]) -> tuple[object, int, int]:
        try:
            return self.client.chat.completions.create(**payload), 1, 0
        except Exception as exc:
            raise _MimoHTTPAttemptsExhausted(exc, attempts=1, retries=0) from exc

    @property
    def _input_modality(self) -> str:
        return "target_video_with_embedded_audio"

    def _media_content(
        self,
        job: MimoBackendJob,
    ) -> list[dict[str, object]]:
        job_payload = job.model_dump(mode="json")
        references = job_payload.get("reference_images")
        if not isinstance(references, list):
            raise TypeError("MiMo job reference inventory is invalid")
        content: list[dict[str, object]] = []
        for reference in references:
            if not isinstance(reference, dict):
                raise TypeError("MiMo reference metadata is invalid")
            path = Path(str(reference["image_artifact_path"]))
            metadata = {"picture_label": reference["picture_label"]}
            content.extend(
                [
                    {"type": "text", "text": _compact_json(metadata)},
                    {
                        "type": "image_url",
                        "image_url": {"url": self.config.media_resolver.resolve(path)},
                    },
                ]
            )
        content.append(
            {
                "type": "video_url",
                "video_url": {
                    "url": self.config.media_resolver.resolve(
                        Path(job.target_video_path)
                    ),
                },
                "fps": self.config.video_fps,
                "media_resolution": self.config.media_resolution,
            }
        )
        return content

    @staticmethod
    def build_compact_task_contract(job: MimoBackendJob) -> dict[str, object]:
        payload = job.model_dump(mode="json")
        segments = payload.get("segments", [])
        if not isinstance(segments, list):
            raise TypeError("MiMo segment inventory is invalid")
        return {
            "clip_uid": job.clip_uid,
            "target_duration_seconds": job.target_duration_seconds,
            "reference_selection": job.reference_selection.model_dump(mode="json"),
            "reference_image_mapping": [
                {
                    key: value
                    for key, value in item.model_dump(mode="json").items()
                    if key
                    not in {
                        "image_artifact_path",
                        "image_sha256",
                    }
                }
                for item in job.reference_images
            ],
            "subject_definition_requirements": [
                {
                    "subject_label": subject.subject_label,
                    "kind": subject.kind,
                    "entity_id": subject.entity_id,
                    "attribute_id": subject.attribute_id,
                    "owner_entity_id": subject.owner_entity_id,
                    "attribute_type": subject.attribute_type,
                    "required_source_picture_labels": list(
                        subject.source_picture_labels
                    ),
                }
                for subject in job.reference_subjects
            ],
            "segments": segments,
            "allowed_segment_ids": [item.segment_id for item in job.segments],
            "transcribed_segment_ids": [
                item.segment_id
                for item in job.segments
                if item.asr_status == "transcribed"
            ],
            "allowed_speaker_bindable_entity_ids": sorted(
                {
                    item.entity_id
                    for item in job.reference_images
                    if item.kind == "subject" and item.entity_id is not None
                }
            ),
        }

    @staticmethod
    def build_mandatory_h3_draft_contract(
        job: MimoBackendJob,
    ) -> dict[str, object]:
        contract = OpenAIMimo25Backend.build_compact_task_contract(job)
        return {
            key: contract[key]
            for key in (
                "reference_selection",
                "reference_image_mapping",
                "subject_definition_requirements",
                "allowed_segment_ids",
                "transcribed_segment_ids",
                "allowed_speaker_bindable_entity_ids",
            )
        }

    def _prompt(
        self, job: MimoBackendJob, *,
        allowed_reference_labels: set[str],
        auxiliary_audio_evidence: dict[str, str] | None = None,
    ) -> str:
        schema = (
            "Return one JSON object matching this schema, with no markdown or extra fields:\n"
            + _compact_json(MimoAVAnnotationDraft.model_json_schema())
            + "\n"
            if self.config.transport == "xiaomi"
            else "Return one JSON object constrained by the supplied response_format.\n"
        )
        contract = self.build_compact_task_contract(job)
        contract["allowed_h3_reference_labels"] = sorted(allowed_reference_labels)
        prompt = schema + "AUTHORITATIVE INPUT:\n" + _compact_json(contract)
        if auxiliary_audio_evidence is not None:
            prompt += "\nAUXILIARY AUDIO CANDIDATES (NOT FACTUAL TRUTH):\n" + _compact_json(
                auxiliary_audio_evidence
            )
        return prompt

    def _request(
        self, job: MimoBackendJob, *,
        allowed_reference_labels: set[str],
        auxiliary_audio_evidence: dict[str, str] | None = None,
    ) -> tuple[str, MimoCompletionDiagnostic, int]:
        content = self._media_content(job)
        content.append({
            "type": "text",
            "text": self._prompt(
                job, allowed_reference_labels=allowed_reference_labels,
                auxiliary_audio_evidence=auxiliary_audio_evidence,
            ),
        })
        modality = self._input_modality
        payload: dict[str, object] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                *(_official_detailed_description_icl_messages() if self.config.icl == "official_ref2va_v1" else []),
                {"role": "user", "content": content},
            ],
            "temperature": self.config.temperature,
            "max_completion_tokens": self.config.max_completion_tokens,
            "stream": False,
        }
        if self.config.transport == "sglang":
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "MimoAVAnnotationDraft",
                    "schema": MimoAVAnnotationDraft.model_json_schema(),
                    "strict": True,
                },
            }
            if self.config.thinking == "disabled":
                payload["reasoning_effort"] = "none"
            payload["extra_body"] = {
                "use_audio_in_video": True,
                "chat_template_kwargs": {
                    "thinking": self.config.thinking == "enabled",
                    "enable_thinking": self.config.thinking == "enabled",
                },
            }
        else:
            payload["response_format"] = {"type": "json_object"}
            payload["extra_body"] = {"thinking": {"type": self.config.thinking}}
        self._av_request_started = True
        completion, attempts, retries = self._call(payload)
        try:
            choices = _value(completion, "choices")
            if not isinstance(choices, list) or not choices:
                raise TypeError("MiMo response has no choices")
            choice = choices[0]
            message = _value(choice, "message")
            raw = _value(message, "content")
            if not isinstance(raw, str):
                raise TypeError("MiMo response content must be text")
            self._av_raw_response = raw
        except Exception as exc:
            raise _MimoResponseContractError(
                exc,
                attempts=attempts,
                retries=retries,
            ) from exc
        return raw, _completion_diagnostic(
            completion,
            choice,
            modality=modality,
            http_attempt_count=attempts,
            thinking=self.config.thinking,
        ), retries

    def describe_auxiliary_audio(self, path: Path) -> MimoAuxAudioDescriptionCall:
        raw = None
        calls = 0
        diagnostic = MimoCompletionDiagnostic(
            input_modality="auxiliary_audio_only",
            usage=MimoUsage(),
            http_attempt_count=1,
        )
        try:
            payload: dict[str, object] = {
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": AUXILIARY_AUDIO_PROMPT},
                    {"role": "user", "content": [{
                        "type": "audio_url",
                        "audio_url": {"url": self.config.media_resolver.resolve(path)},
                    }]},
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "MimoAuxAudioDescription",
                        "schema": MimoAuxAudioDescription.model_json_schema(),
                        "strict": True,
                    },
                },
                "temperature": 0.0,
                "max_completion_tokens": 1024,
                "stream": False,
            }
            if self.config.transport == "sglang":
                payload["reasoning_effort"] = "none"
                payload["extra_body"] = {
                    "chat_template_kwargs": {"thinking": False, "enable_thinking": False}
                }
            else:
                payload["extra_body"] = {"thinking": {"type": "disabled"}}
            calls = 1
            completion, _, _ = self._call(payload)
            choice = _value(completion, "choices")[0]
            raw = _value(_value(choice, "message"), "content")
            diagnostic = _completion_diagnostic(
                completion, choice, modality="auxiliary_audio_only",
                http_attempt_count=1, thinking="disabled",
            )
            description, issues = parse_structured_json_issues(raw, MimoAuxAudioDescription)
            if issues:
                raise ValueError("; ".join(item.message for item in issues))
            return MimoAuxAudioDescriptionCall(
                description=description.description, raw_response=raw, diagnostic=diagnostic,
            )
        except (ValueError, TypeError, IndexError, KeyError, OSError, _MimoHTTPAttemptsExhausted) as exc:
            diagnostic.request_error = f"{type(exc).__name__}: {exc}"
            return MimoAuxAudioDescriptionCall(
                raw_response=raw if isinstance(raw, str) else None,
                diagnostic=diagnostic, error=diagnostic.request_error, model_call_count=calls,
            )

    def _maybe_polish_speaker_markers(
        self,
        annotation: MimoAVAnnotationDraft,
        *,
        speech: list[dict[str, Any]],
        allowed_labels: set[str],
        issues: list[ValidationIssue],
        warnings: list[str],
    ) -> tuple[MimoAVAnnotationDraft, MimoSpeakerMarkerPolishAudit, MimoCompletionDiagnostic | None]:
        audit = MimoSpeakerMarkerPolishAudit()
        codes = {issue.code for issue in issues}
        if codes - _SPEAKER_MARKER_POLISH_ISSUES or not (
            (codes | set(warnings)) & _SPEAKER_MARKER_POLISH_ISSUES
        ):
            return annotation, audit, None
        original = annotation.h3_semantics.shot1_caption
        dialogue_blocks = _DIALOGUE.findall(original)
        if not dialogue_blocks or len(dialogue_blocks) != len(speech):
            return annotation, audit, None
        diagnostic = MimoCompletionDiagnostic(
            input_modality="speaker_marker_text_only", usage=MimoUsage(), http_attempt_count=1,
        )
        payload: dict[str, object] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": SPEAKER_MARKER_POLISH_PROMPT},
                {"role": "user", "content": _compact_json({
                    "authoritative_speaker_facts": speech,
                    "dialogue_marker_targets": [
                        {"dialogue_index": index, "speaker_id": fact["speaker_id"]}
                        for index, fact in enumerate(speech, 1)
                    ],
                    "allowed_h3_reference_labels": sorted(allowed_labels),
                    "existing_shot_caption": original,
                    "speaker_marker_projection_issues": sorted(
                        (codes | set(warnings)) & _SPEAKER_MARKER_POLISH_ISSUES
                    ),
                })},
            ],
            "temperature": 0.0, "max_completion_tokens": 2048, "stream": False,
        }
        if self.config.transport == "sglang":
            payload["response_format"] = {
                "type": "json_schema", "json_schema": {
                    "name": "MimoSpeakerMarkerPolish",
                    "schema": MimoSpeakerMarkerPolish.model_json_schema(), "strict": True,
                },
            }
            payload["reasoning_effort"] = "none"
            payload["extra_body"] = {
                "chat_template_kwargs": {"thinking": False, "enable_thinking": False},
            }
        else:
            payload["response_format"] = {"type": "json_object"}
            payload["extra_body"] = {"thinking": {"type": "disabled"}}
        audit.attempted = True
        try:
            completion, _, _ = self._call(payload)
            choice = _value(completion, "choices")[0]
            raw = _value(_value(choice, "message"), "content")
            audit.raw_response = raw if isinstance(raw, str) else None
            diagnostic = _completion_diagnostic(
                completion, choice, modality="speaker_marker_text_only",
                http_attempt_count=1, thinking="disabled",
            )
            # This request contains no media, regardless of provider usage metadata.
            diagnostic.usage.image_tokens = None
            diagnostic.usage.video_tokens = None
            diagnostic.usage.audio_tokens = None
            _validate_finish_reason(diagnostic)
            response, parse_issues = parse_structured_json_issues(raw, MimoSpeakerMarkerPolish)
            if response is None or parse_issues:
                raise ValueError("; ".join(issue.message for issue in parse_issues))
            audit.needs_review = response.needs_review
            if response.needs_review:
                diagnostic.warnings.append("speaker_marker_polish_needs_review")
            elif not _speaker_marker_only_edit(original, response.shot1_caption):
                audit.error = "speaker_marker_polish_non_marker_edit_rejected"
                diagnostic.warnings.append(audit.error)
            else:
                _, remaining, remaining_warnings = protect_direct_dialogue(
                    response.shot1_caption, speech, allowed_labels=allowed_labels,
                )
                if remaining or set(remaining_warnings) & _SPEAKER_MARKER_POLISH_ISSUES:
                    audit.error = "speaker_marker_polish_unresolved"
                    diagnostic.warnings.append(audit.error)
                else:
                    audit.applied = response.shot1_caption != original
                    annotation = annotation.model_copy(update={
                        "h3_semantics": annotation.h3_semantics.model_copy(
                            update={"shot1_caption": response.shot1_caption},
                        ),
                    })
        except (ValueError, TypeError, IndexError, KeyError, AttributeError, OSError, _MimoHTTPAttemptsExhausted) as exc:
            audit.error = f"{type(exc).__name__}: {exc}"
            diagnostic.request_error = audit.error
            diagnostic.warnings.append("speaker_marker_polish_failed")
        return annotation, audit, diagnostic

    def reconcile(
        self,
        job: MimoBackendJob,
        *,
        segment_ids: list[str],
        transcribed_segment_ids: list[str],
        allowed_entity_ids: set[str],
        allowed_reference_labels: set[str],
        auxiliary_audio_evidence: dict[str, str] | None = None,
    ) -> MimoBackendResult:
        self._av_request_started = False
        try:
            raw, diagnostic, _ = self._request(
                job, allowed_reference_labels=allowed_reference_labels,
                auxiliary_audio_evidence=auxiliary_audio_evidence,
            )
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            diagnostics = (
                (
                    MimoCompletionDiagnostic(
                        input_modality=self._input_modality,
                        usage=MimoUsage(),
                        http_attempt_count=1,
                        request_error=reason,
                    ),
                )
                if self._av_request_started
                else ()
            )
            raise MimoBackendFailure(
                code="mimo_request_failed",
                reason=reason,
                diagnostics=diagnostics,
                raw_responses=(self._av_raw_response,)
                if self._av_raw_response is not None
                else (),
                model_call_count=int(self._av_request_started),
                http_attempt_count=int(self._av_request_started),
            ) from exc
        try:
            _validate_finish_reason(diagnostic)
            _validate_av_observation_usage(diagnostic, require_explicit_audio=False)
            if diagnostic.usage.audio_tokens == 0:
                diagnostic.warnings.append("embedded_audio_tokens_zero")
        except MimoBackendFailure as exc:
            raise MimoBackendFailure(
                code=exc.code,
                reason=exc.reason,
                raw_responses=(raw,),
                diagnostics=(diagnostic,),
                model_call_count=1,
                http_attempt_count=1,
            ) from exc
        canonical_raw, raw_corrections = _canonicalize_raw_annotation_payload(raw)
        corrections = Counter(raw_corrections)
        annotation, issues = parse_structured_json_issues(
            canonical_raw, MimoAVAnnotationDraft
        )
        polish = MimoSpeakerMarkerPolishAudit()
        diagnostics = [diagnostic]
        if annotation is not None:
            annotation, grounding_corrections = _normalize_speaker_annotation(
                annotation,
                segment_ids=segment_ids,
                transcribed_segment_ids=set(transcribed_segment_ids),
                allowed_entity_ids=allowed_entity_ids,
            )
            corrections.update(grounding_corrections)
            diagnostic.warnings.extend(
                f"deterministic_correction:{key}" for key in sorted(corrections)
            )
            issues = validate_annotation(
                annotation,
                segment_ids=segment_ids,
                segment_intervals={
                    item.segment_id: (item.start_time, item.end_time)
                    for item in job.segments
                },
                transcribed_segment_ids=transcribed_segment_ids,
                authoritative_transcripts=[
                    item.asr_text
                    for item in job.segments
                    if item.asr_status == "transcribed" and item.asr_text is not None
                ],
                allowed_entity_ids=allowed_entity_ids,
                allowed_reference_labels=allowed_reference_labels,
                reference_subjects=job.reference_subjects,
                target_duration_seconds=job.target_duration_seconds,
            )
            review_only = _REVIEW_ONLY_ISSUES
            if not transcribed_segment_ids:
                review_only = review_only | {
                    "segment_inventory_mismatch", "visual_segment_inventory_mismatch",
                }
            diagnostic.warnings.extend(
                item.code for item in issues if item.code in review_only
            )
            issues = [item for item in issues if item.code not in review_only]
            speech = direct_speech_facts(annotation, list(job.segments))
            _, direct_issues, direct_warnings = protect_direct_dialogue(
                annotation.h3_semantics.shot1_caption,
                speech,
                allowed_labels=allowed_reference_labels,
            )
            for name in ("summary", "style_opening"):
                unknown = (
                    set(
                        _REFERENCE_LABEL.findall(getattr(annotation.h3_semantics, name))
                    )
                    - allowed_reference_labels
                )
                if unknown:
                    issues.append(
                        ValidationIssue(
                            "direct_unknown_reference", name, str(sorted(unknown))
                        )
                    )
            annotation, polish, polish_diagnostic = self._maybe_polish_speaker_markers(
                annotation, speech=speech, allowed_labels=allowed_reference_labels,
                issues=[*issues, *direct_issues], warnings=direct_warnings,
            )
            if polish_diagnostic is not None:
                diagnostics.append(polish_diagnostic)
            if polish.applied:
                _, direct_issues, direct_warnings = protect_direct_dialogue(
                    annotation.h3_semantics.shot1_caption, speech,
                    allowed_labels=allowed_reference_labels,
                )
            issues.extend(direct_issues)
            diagnostic.warnings.extend(direct_warnings)
        text_calls = int(polish.attempted)
        if annotation is None or issues:
            raise MimoBackendFailure(
                code="mimo_structured_output_failed",
                reason="MiMo single AV observation failed validation",
                raw_responses=(raw,),
                diagnostics=tuple(diagnostics),
                issues=tuple(issues),
                model_call_count=1 + text_calls,
                http_attempt_count=1 + text_calls,
                annotation=annotation,
                text_model_call_count=text_calls,
                speaker_marker_polish=polish,
            )
        return MimoBackendResult(
            annotation=annotation,
            raw_responses=(raw,),
            diagnostics=tuple(diagnostics),
            model_call_count=1 + text_calls,
            http_attempt_count=1 + text_calls,
            http_retry_count=0,
            recheck_count=0,
            input_modality=self._input_modality,
            deterministic_correction_counts=dict(sorted(corrections.items())),
            text_model_call_count=text_calls,
            speaker_marker_polish=polish,
        )


__all__ = [
    "DEFAULT_BASE64_LIMIT_BYTES",
    "MIMO25_MATERIALIZER_VERSION",
    "MIMO25_MODEL",
    "MIMO25_POLICY_VERSION",
    "MIMO25_PROMPT_VERSION",
    "MIMO25_SCHEMA_VERSION",
    "SYSTEM_PROMPT",
    "MimoAVAnnotationDraft",
    "MimoAVGrounding",
    "MimoAVSegmentGrounding",
    "MimoAudioEvent",
    "MimoAudioObservation",
    "MimoAudioSegmentDecision",
    "MimoAudioSemantics",
    "MimoAuxAudioDescription",
    "MimoAuxAudioDescriptionCall",
    "MimoBackendConfig",
    "MimoBackendFailure",
    "MimoBackendProvenance",
    "MimoBackendResult",
    "MimoCompletionDiagnostic",
    "MimoH3Draft",
    "MimoH3Semantics",
    "MimoMediaResolver",
    "MimoSegmentDecision",
    "MimoSpeakerVoiceProfile",
    "MimoSubjectDefinitionDraft",
    "MimoTransport",
    "MimoVisibleEntityObservation",
    "MimoVisualBlock",
    "MimoVisualObservation",
    "MimoVisualRetentionDraft",
    "MimoVisualSegmentView",
    "OpenAIMimo25Backend",
    "SpeechPresentation",
    "sha256_file",
    "validate_annotation",
]
