"""One-request RA2VA shadow adapter over the frozen final annotation contract."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, StrictStr, model_validator

from r2v_data_v2.h3.mimo25_backend import (
    _DIALOGUE,
    _REFERENCE_LABEL,
    MIMO25_SCHEMA_VERSION,
    MimoAVAnnotationDraft,
    MimoBackendConfig,
    MimoBackendFailure,
    MimoBackendProvenance,
    MimoBackendResult,
    MimoCompletionDiagnostic,
    MimoH3Semantics,
    MimoSpeakerMarkerPolishAudit,
    MimoSpeakerVoiceProfile,
    MimoUsage,
    MimoVisualBlock,
    _compact_json,
    _completion_diagnostic,
    _official_detailed_description_icl_messages,
    _sha256_text,
    _validate_av_observation_usage,
    _validate_direct_transcribed_dialogue,
    _validate_finish_reason,
    _value,
    direct_speech_facts,
    protect_direct_dialogue,
)
from r2v_data_v2.h3.mimo25_stem_shadow import StemAwareOpenAIMimo25Backend
from r2v_data_v2.h3.schemas import SchemaModel
from r2v_data_v2.structured_output import (
    ValidationIssue,
    normalize_structured_json_envelope,
    parse_structured_json_issues,
)

SINGLE_PROMPT_VERSION = "h3_mimo26_ra2va_single_v9_closed_subjects"
SINGLE_BACKEND_VERSION = "r2v.h3.mimo25_backend.75"
SINGLE_COMPACT_SCHEMA_VERSION = "r2v.h3.mimo26_single_compact.2"

_COMPACT_PICTURE_LABEL = r"<Picture [1-9]\d*>"
_COMPACT_PICTURE_PROVENANCE = re.compile(
    rf"\s*(?:as shown in|shown in|depicted in|with its visual detail sourced from)\s*"
    rf"{_COMPACT_PICTURE_LABEL}(?:\s*(?:,|and)\s*{_COMPACT_PICTURE_LABEL})*",
    re.IGNORECASE,
)
_COMPACT_ORPHAN_PICTURE = re.compile(rf"\s*{_COMPACT_PICTURE_LABEL}")

SINGLE_SYSTEM_PROMPT = """Return one MimoSingleCompactAnnotationDraftV2 JSON object after observing the complete original target AV, frozen Pictures, and resolved stems. Original AV is the final audiovisual authority; stems aid acoustic recall. Official H3 demonstrations may contain final speaker markers or Picture provenance; those are final-H3 examples, not fields you own in this compact response.

STEP 1 - VISUAL AND H3 CONTENT
Watch the entire video and describe composition, foreground/background, Subject appearance, lighting, camera, actions, expression, and chronological progression through the end. Write visual_blocks, concise ordered Subject definitions and retention rows, one style_opening sentence, and a detailed shot1_caption. Cite visibly preserved entity/background <Subject N> naturally at first appearance. Do not put <Picture N> or Picture provenance in subject_definitions[*].description. Do not output a per-segment visible-entity inventory or internal entity IDs. A sampled mouth-open or mouth-closed frame alone does not establish speaker identity.
Write a concise but content-complete summary with the setting, principal visible Subjects, main visible action or state change, and the role of speech/audio when relevant. Do not reduce it to only the spoken line or a generic event label.
Write a dense chronological shot1_caption covering the clip from the opening composition through the ending visual state. When visibly relevant, include framing and camera movement, Subject posture, gaze, expression and movement, object interactions, foreground/background spatial relationships, visible progression before, during, and after speech, exact dialogue at its chronological position, and the ending visual state. Prefer complete observable coverage over brevity, but never invent details merely to make the caption longer.
Use a background <Subject N> only when it helps describe composition, spatial layout, scene continuity, or disambiguation. Do not mention background Subjects mechanically.
Retention descriptions must be self-contained and unambiguous. Do not begin a retention description with He, She, They, It, His, Her, Their, or Its to refer to the retained Subject. Describe preserved visual traits directly.
The supplied Subject inventory is closed. Use only Subject labels listed in required_output_inventory.subject_labels_in_order. Never create a new <Subject N> for another visible person, object, attribute, or background; describe unreferenced visible content with ordinary prose instead of assigning it a Subject label.
Each supplied <Subject N> refers only to the visual referent represented by its frozen source Picture(s) and frozen Subject contract. Never repurpose a Subject label for a different visible referent. In subject_definitions, describe that Subject's source Picture referent, not another salient person or object in the target video.

STEP 2 - SEGMENT SPEAKER DECISIONS
Output exactly one ordered segments row for EVERY supplied DiariZen segment, including non-transcribed ones. Assign stable g1, g2, ... acoustic groups by first vocal-source appearance; pauses, language changes, and segment boundaries alone do not create groups. Use primary_speaker_group=null when grouping is unreliable. Give transcribed speech a concise audible delivery_style and use null for non-transcribed segments. For a supported visible speaker choose visible_subject and speaker_subject_label from required_output_inventory.allowed_speaker_subject_labels. For genuine offscreen speech choose offscreen and null; if several people remain plausible, choose no_reliable_subject or uncertain and null rather than guessing. Do not output confidence, evidence codes, mouth/lip fields, or entity_id. Profile only distinct non-null groups in transcribed segments in first transcribed appearance order, using their speech-stem windows; do not re-decide grouping or binding.

STEP 3 - EXACT DIALOGUE
Place every immutable required_dialogue_block exactly once in shot1_caption, in chronological playback order, with a natural visible or offscreen speaker lead-in. Do not translate, paraphrase, normalize, merge, or omit dialogue. Non-transcribed segments have no dialogue block. Do NOT emit any (S1), (S2), or other (Sx) speaker markers: the pipeline owns speaker IDs and projects them deterministically.
Before returning, verify that every required_dialogue_block appears exactly once and in chronological order in shot1_caption. Dialogue preservation takes priority over caption elaboration: visual detail must never omit, change, translate, merge, or duplicate a required dialogue block.

STEP 4 - AUDIO AND H3 FIELD BOUNDARIES
Keep grounding rationale out of the caption. Write a concise summary without task prefix. Put continuous ambience and physical/environmental sound in overall_soundscape and audience-only score in non_diegetic_music, not in shot1_caption; the caption owns visual prose, speaker presentation, and exact dialogue. The materializer owns <Audio N>, Picture provenance, relationships, retention, and final six-section H3 formatting. Treat required_output_inventory as exact and ordered: subject_definitions and visual_retention_analysis each cover every supplied Subject in order; segments covers every supplied segment in order; only transcribed_dialogue_blocks_in_order supplies dialogue. If it is empty, output no <d> blocks and no speaker_voice_profiles.

Return only compact JSON, not intermediate reasoning."""


class MimoSingleCompactSegmentDecision(SchemaModel):
    segment_id: str = Field(min_length=1)
    primary_speaker_group: str | None = Field(pattern=r"^g[1-9]\d*$")
    delivery_style: StrictStr | None
    binding_status: Literal["visible_subject", "offscreen", "no_reliable_subject", "uncertain"]
    speaker_subject_label: str | None = Field(pattern=r"^<Subject [1-9]\d*>$")

    @model_validator(mode="after")
    def validate_decision(self) -> MimoSingleCompactSegmentDecision:
        if self.delivery_style is not None and not self.delivery_style.strip():
            raise ValueError("compact delivery style must be non-empty or null")
        if (self.binding_status == "visible_subject") != (self.speaker_subject_label is not None):
            raise ValueError("visible_subject requires a speaker Subject; other bindings require null")
        return self


class MimoSingleCompactH3Semantics(MimoH3Semantics):
    pass


class MimoSingleCompactAnnotationDraftV2(SchemaModel):
    schema_version: Literal["r2v.h3.mimo26_single_compact.2"]
    visual_blocks: list[MimoVisualBlock] = Field(min_length=1)
    segments: list[MimoSingleCompactSegmentDecision]
    speaker_voice_profiles: list[MimoSpeakerVoiceProfile]
    h3_semantics: MimoSingleCompactH3Semantics


def _canonicalize_single_compact_raw(raw: str) -> tuple[str, dict[str, int]]:
    try:
        payload = json.loads(normalize_structured_json_envelope(raw))
    except (json.JSONDecodeError, TypeError, ValueError):
        return raw, {}
    if not isinstance(payload, dict):
        return raw, {}
    semantics = payload.get("h3_semantics")
    definitions = semantics.get("subject_definitions") if isinstance(semantics, dict) else None
    if not isinstance(definitions, list):
        return raw, {}
    corrected = 0
    for definition in definitions:
        if not isinstance(definition, dict) or not isinstance(definition.get("description"), str):
            continue
        description = definition["description"]
        cleaned = _COMPACT_PICTURE_PROVENANCE.sub("", description)
        cleaned = _COMPACT_ORPHAN_PICTURE.sub("", cleaned)
        if cleaned != description:
            cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
            cleaned = cleaned.strip(" ,;:\t\r\n")
            definition["description"] = cleaned if any(char.isalnum() for char in cleaned) else ""
            corrected += 1
    if not corrected:
        return raw, {}
    return json.dumps(payload, ensure_ascii=False), {
        "compact_subject_picture_provenance_removed": corrected,
    }


def _normalize_single_compact_speaker_subjects(
    draft: MimoSingleCompactAnnotationDraftV2, job: Any,
) -> tuple[MimoSingleCompactAnnotationDraftV2, dict[str, int]]:
    subjects = {subject.subject_label: subject for subject in job.reference_subjects}
    segments = []
    corrections: dict[str, int] = {}
    for decision in draft.segments:
        if decision.binding_status != "visible_subject":
            segments.append(decision)
            continue
        subject = subjects.get(decision.speaker_subject_label)
        if subject is not None and subject.kind == "entity" and subject.entity_id is not None:
            segments.append(decision)
            continue
        owner = None
        owner_entity_id = getattr(subject, "owner_entity_id", None)
        if subject is not None and subject.kind == "attribute" and owner_entity_id is not None:
            candidates = [
                item for item in job.reference_subjects
                if item.kind == "entity" and item.entity_id == owner_entity_id
            ]
            if len(candidates) == 1:
                owner = candidates[0]
        if owner is not None:
            segments.append(decision.model_copy(update={"speaker_subject_label": owner.subject_label}))
            code = "compact_speaker_attribute_promoted_to_owner"
        else:
            segments.append(decision.model_copy(update={
                "binding_status": "no_reliable_subject", "speaker_subject_label": None,
            }))
            code = "compact_invalid_speaker_subject_downgraded"
        corrections[code] = corrections.get(code, 0) + 1
    if not corrections:
        return draft, {}
    return draft.model_copy(update={"segments": segments}), corrections


def _single_semantic_icl_messages() -> list[dict[str, str]]:
    example = {
        "schema_version": SINGLE_COMPACT_SCHEMA_VERSION,
        "visual_blocks": [{
            "block_id": "v1",
            "text": "<Subject 1> waits beside the doorway while <Subject 2> turns and answers.",
        }],
        "segments": [
            {
                "segment_id": "segment_0001", "primary_speaker_group": "g1",
                "delivery_style": "calm conversational delivery",
                "binding_status": "visible_subject", "speaker_subject_label": "<Subject 2>",
            },
            {
                "segment_id": "segment_0002", "primary_speaker_group": "g2",
                "delivery_style": "brief questioning delivery",
                "binding_status": "offscreen", "speaker_subject_label": None,
            },
        ],
        "speaker_voice_profiles": [
            {"speaker_group": "g1", "voice_characteristics": "a mid-register voice with a measured cadence"},
            {"speaker_group": "g2", "voice_characteristics": "a lower voice with questioning delivery"},
        ],
        "h3_semantics": {
            "subject_definitions": [
                {"subject_label": "<Subject 1>", "description": "an adult woman with dark shoulder-length hair and a dark green coat"},
                {"subject_label": "<Subject 2>", "description": "an adult man with short dark hair and a gray jacket"},
            ],
            "summary": "Two people pause near an entrance; the visible man answers first, followed by an offscreen speaker.",
            "style_opening": "Naturalistic live-action cinematography with a steady camera and soft practical lighting.",
            "shot1_caption": "<Subject 1> watches quietly beside the doorway while <Subject 2> turns toward the camera side. <Subject 2> replies, <d>[Chinese] 好，我知道了。</d> He pauses as an offscreen voice asks, <d>[Chinese] 你去哪？</d>",
            "overall_soundscape": "A quiet interior ambience with faint footsteps and subtle doorway movement.",
            "non_diegetic_music": "N/A",
            "visual_retention_analysis": [
                {"subject_label": "<Subject 1>", "marker": "fully_preserved", "description": "Her hairstyle, coat, and visible proportions remain consistent."},
                {"subject_label": "<Subject 2>", "marker": "fully_preserved", "description": "His short hair, jacket, and visible proportions remain consistent."},
            ],
        },
    }
    return [
        {"role": "user", "content": (
            "This is a compact synthetic demonstration of Subject, acoustic group, "
            "and exact-dialogue field boundaries. Do not copy its people, dialogue, "
            "entities, visual style, or factual content into the real task."
        )},
        {"role": "assistant", "content": _compact_json(example)},
    ]


def _validate_single_compact_annotation(
    draft: MimoSingleCompactAnnotationDraftV2,
    job: Any,
    *,
    allowed_entity_ids: set[str],
    allowed_reference_labels: set[str],
) -> tuple[list[ValidationIssue], list[str]]:
    issues: list[ValidationIssue] = []
    warnings: list[str] = []
    segment_ids = [segment.segment_id for segment in job.segments]
    if [row.segment_id for row in draft.segments] != segment_ids:
        issues.append(ValidationIssue(
            "segment_inventory_mismatch", "segments",
            "compact rows must exactly follow supplied chronological segment IDs",
        ))
    expected_subjects = [subject.subject_label for subject in job.reference_subjects]
    for field, rows in (
        ("subject_definitions", draft.h3_semantics.subject_definitions),
        ("visual_retention_analysis", draft.h3_semantics.visual_retention_analysis),
    ):
        if [row.subject_label for row in rows] != expected_subjects:
            issues.append(ValidationIssue(
                "subject_inventory_mismatch", field,
                "compact Subject rows must exactly follow supplied Subject order",
            ))
    entity_by_subject = {
        subject.subject_label: subject.entity_id
        for subject in job.reference_subjects
        if subject.kind == "entity" and subject.entity_id is not None
    }
    subjects_by_group: dict[str, set[str]] = {}
    groups_by_subject: dict[str, set[str]] = {}
    decisions = {row.segment_id: row for row in draft.segments}
    for decision in draft.segments:
        if decision.binding_status != "visible_subject":
            continue
        label = decision.speaker_subject_label
        if label not in entity_by_subject:
            issues.append(ValidationIssue(
                "unknown_speaker_subject", decision.segment_id, str(label),
            ))
        if decision.primary_speaker_group is not None:
            group = decision.primary_speaker_group
            assert label is not None
            subjects_by_group.setdefault(group, set()).add(label)
            groups_by_subject.setdefault(label, set()).add(group)
    warnings.extend(
        "compact_group_maps_multiple_subjects"
        for subjects in subjects_by_group.values() if len(subjects) > 1
    )
    warnings.extend(
        "compact_subject_maps_multiple_groups"
        for groups in groups_by_subject.values() if len(groups) > 1
    )
    first_groups = list(dict.fromkeys(
        row.primary_speaker_group for row in draft.segments
        if row.primary_speaker_group is not None
    ))
    if first_groups != [f"g{index}" for index in range(1, len(first_groups) + 1)]:
        issues.append(ValidationIssue(
            "non_contiguous_speaker_groups", "segments",
            "speaker groups must be contiguous by first chronological appearance",
        ))
    transcribed = [segment for segment in job.segments if segment.asr_status == "transcribed"]
    expected_profiles = list(dict.fromkeys(
        decisions[segment.segment_id].primary_speaker_group
        for segment in transcribed if segment.segment_id in decisions
        and decisions[segment.segment_id].primary_speaker_group is not None
    ))
    if [profile.speaker_group for profile in draft.speaker_voice_profiles] != expected_profiles:
        warnings.append("compact_speaker_voice_profile_inventory_mismatch")
    expected_dialogue = [
        f"<d>[{segment.asr_language or 'Unknown'}] {segment.asr_text}</d>"
        for segment in transcribed
    ]
    caption = draft.h3_semantics.shot1_caption
    actual_dialogue = [match.group(0) for match in _DIALOGUE.finditer(caption)]
    if (
        actual_dialogue != expected_dialogue
        or caption.count("<d>") != len(actual_dialogue)
        or caption.count("</d>") != len(actual_dialogue)
    ):
        issues.append(ValidationIssue(
            "direct_transcribed_dialogue_inventory_mismatch", "h3_semantics.shot1_caption",
            "dialogue blocks must exactly match chronological authoritative ASR",
        ))
    for field, value in (
        ("summary", draft.h3_semantics.summary),
        ("style_opening", draft.h3_semantics.style_opening),
        ("overall_soundscape", draft.h3_semantics.overall_soundscape),
        ("non_diegetic_music", draft.h3_semantics.non_diegetic_music),
        *(("subject_definitions", row.description) for row in draft.h3_semantics.subject_definitions),
        *(("visual_retention_analysis", row.description) for row in draft.h3_semantics.visual_retention_analysis),
    ):
        if "<d>" in value or "</d>" in value:
            issues.append(ValidationIssue(
                "dialogue_outside_shot_caption", field, "dialogue belongs only in shot1_caption",
            ))
        if (
            re.search(r"\(S[1-9]\d*\)|\[Shot\s+\d+\]", value)
            or "[[" in value or "]]" in value
        ):
            issues.append(ValidationIssue(
                "pipeline_owned_syntax_outside_shot_caption", field,
                "speaker IDs, shot markers, and internal placeholders are pipeline-owned",
            ))
    prose = [
        *(block.text for block in draft.visual_blocks),
        draft.h3_semantics.summary, draft.h3_semantics.style_opening, caption,
        draft.h3_semantics.overall_soundscape, draft.h3_semantics.non_diegetic_music,
        *(row.description for row in draft.h3_semantics.subject_definitions),
        *(row.description for row in draft.h3_semantics.visual_retention_analysis),
    ]
    unknown_labels = sorted({
        label for value in prose for label in _REFERENCE_LABEL.findall(value)
        if label not in allowed_reference_labels
    })
    if unknown_labels:
        issues.append(ValidationIssue("direct_unknown_reference", "h3_semantics", str(unknown_labels)))
    return issues, sorted(set(warnings))


def _project_compact_speaker_markers(caption: str, speech: list[dict[str, Any]]) -> str:
    # The model may imitate final-H3 ICL markers; only this projection assigns Sx.
    cleaned = re.sub(r"\(S[1-9]\d*\)", "", caption)
    blocks = list(_DIALOGUE.finditer(cleaned))
    if len(blocks) != len(speech):
        raise ValueError("compact dialogue inventory differs before Sx projection")
    for block, fact in reversed(list(zip(blocks, speech, strict=True))):
        cleaned = cleaned[:block.start()] + f"({fact['speaker_id']}) " + cleaned[block.start():]
    return cleaned


def _project_compact_to_legacy_annotation(
    draft: MimoSingleCompactAnnotationDraftV2, job: Any,
) -> MimoAVAnnotationDraft:
    # Legacy-only fields are conservative compatibility placeholders, not MiMo evidence.
    entity_by_subject = {
        subject.subject_label: subject.entity_id
        for subject in job.reference_subjects
        if subject.kind == "entity" and subject.entity_id is not None
    }

    def bound_entity(decision: MimoSingleCompactSegmentDecision) -> str | None:
        if decision.binding_status != "visible_subject":
            return None
        entity_id = entity_by_subject.get(decision.speaker_subject_label)
        if entity_id is None:
            raise ValueError("compact speaker Subject is absent from frozen entity Subjects")
        return entity_id

    bound_entities = [bound_entity(item) for item in draft.segments]
    values = {
        "schema_version": MIMO25_SCHEMA_VERSION,
        "visual_observation": {
            "visual_blocks": [item.model_dump(mode="json") for item in draft.visual_blocks],
            # This bound-speaker projection is not an exhaustive visibility inventory.
            "segment_views": [
                {
                    "segment_id": item.segment_id,
                    "visible_entity_ids": [entity_id] if entity_id is not None else [],
                    "entity_observations": [],
                }
                for item, entity_id in zip(draft.segments, bound_entities, strict=True)
            ],
        },
        "audio_observation": {
            "segment_decisions": [
                {
                    "segment_id": item.segment_id,
                    "primary_speaker_group": item.primary_speaker_group,
                    "delivery_style": item.delivery_style,
                    "resolution": "resolved" if item.primary_speaker_group is not None else "uncertain",
                    "vocal_composition": "uncertain",
                    "secondary_vocal_activity": {
                        "present": False, "speaker_relation": "none", "kind": None,
                    },
                    "confidence": "low", "audio_evidence_codes": ["insufficient_evidence"],
                }
                for item in draft.segments
            ],
            "speaker_voice_profiles": [
                item.model_dump(mode="json") for item in draft.speaker_voice_profiles
            ],
        },
        "av_grounding": {"segment_groundings": [
            {
                "segment_id": item.segment_id,
                "primary_speaker_group": item.primary_speaker_group,
                "binding_status": {
                    "visible_subject": "visible_entity", "offscreen": "offscreen",
                    "no_reliable_subject": "no_reliable_entity", "uncertain": "uncertain",
                }[item.binding_status],
                "entity_id": entity_id,
                "speech_presentation": {
                    "visible_subject": "onscreen_spoken", "offscreen": "offscreen_spoken",
                    "no_reliable_subject": "uncertain", "uncertain": "uncertain",
                }[item.binding_status],
                "confidence": "low", "evidence_codes": ["insufficient_evidence"],
            }
            for item, entity_id in zip(draft.segments, bound_entities, strict=True)
        ]},
        "h3_semantics": draft.h3_semantics.model_dump(mode="json"),
        "warnings": [],
    }
    preliminary = MimoAVAnnotationDraft.model_validate(values)
    speech = direct_speech_facts(preliminary, list(job.segments))
    values["h3_semantics"]["shot1_caption"] = _project_compact_speaker_markers(
        draft.h3_semantics.shot1_caption, speech,
    )
    return MimoAVAnnotationDraft.model_validate(values)


class SingleCallOpenAIMimo25Backend(StemAwareOpenAIMimo25Backend):
    def __init__(self, config: MimoBackendConfig, **kwargs: Any) -> None:
        if config.transport != "sglang":
            raise ValueError("single RA2VA requires SGLang structured output")
        super().__init__(config, **kwargs)

    @property
    def provenance(self) -> MimoBackendProvenance:
        values = self.config.provenance().model_dump(
            mode="json", exclude={"configuration_fingerprint"},
        )
        values.update(schema_version=SINGLE_BACKEND_VERSION, prompt_version=SINGLE_PROMPT_VERSION)
        return MimoBackendProvenance(
            **values, configuration_fingerprint=_sha256_text(_compact_json(values)),
        )

    @property
    def _input_modality(self) -> str:
        return "target_av_with_auxiliary_raw_audio"

    def _request(
        self, job: Any, *, allowed_reference_labels: set[str],
        auxiliary_audio_paths: dict[str, Path] | None = None,
    ) -> tuple[str, tuple[str | None, ...], list[MimoCompletionDiagnostic], dict[str, int]]:
        diagnostic = MimoCompletionDiagnostic(
            input_modality="target_av_with_auxiliary_raw_audio",
            usage=MimoUsage(), http_attempt_count=1,
        )
        raw: str | None = None
        attempted = False
        try:
            if auxiliary_audio_paths is None or set(auxiliary_audio_paths) != {"speech", "music", "sfx"}:
                raise ValueError("single RA2VA requires all three resolved stems")
            contract = self.build_compact_task_contract(job)
            segments = []
            for source in contract["segments"]:
                segment = {
                    key: value for key, value in source.items()
                    if key in {
                        "segment_id", "start_time", "end_time", "source_speaker_cluster_id",
                        "asr_status", "asr_text", "asr_language",
                    }
                }
                if segment["asr_status"] == "transcribed":
                    segment["required_dialogue_block"] = (
                        f"<d>[{segment['asr_language'] or 'Unknown'}] "
                        f"{segment['asr_text']}</d>"
                    )
                segments.append(segment)
            contract["segments"] = segments
            contract["required_output_inventory"] = {
                "subject_labels_in_order": [subject.subject_label for subject in job.reference_subjects],
                "allowed_speaker_subject_labels": [
                    subject.subject_label for subject in job.reference_subjects
                    if subject.kind == "entity" and subject.entity_id is not None
                ],
                "segment_ids_in_order": [segment.segment_id for segment in job.segments],
                "transcribed_dialogue_blocks_in_order": [
                    {"segment_id": segment["segment_id"],
                     "required_dialogue_block": segment["required_dialogue_block"]}
                    for segment in segments if segment["asr_status"] == "transcribed"
                ],
            }
            contract.pop("allowed_speaker_bindable_entity_ids", None)
            contract["allowed_h3_reference_labels"] = sorted(allowed_reference_labels)
            content = self._media_content(job)
            for kind in ("speech", "music", "sfx"):
                content.extend([
                    {"type": "text", "text": f"resolved {kind} stem"},
                    {"type": "audio_url", "audio_url": {
                        "url": self.config.media_resolver.resolve(auxiliary_audio_paths[kind]),
                    }},
                ])
            content.append({"type": "text", "text": "AUTHORITATIVE INPUT:\n" + _compact_json(contract)})
            payload: dict[str, object] = {
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": SINGLE_SYSTEM_PROMPT},
                    *(_official_detailed_description_icl_messages() if self.config.icl == "official_ref2va_v1" else []),
                    *(_single_semantic_icl_messages() if self.config.icl == "official_ref2va_v1" else []),
                    {"role": "user", "content": content},
                ],
                "temperature": self.config.temperature,
                "max_completion_tokens": self.config.max_completion_tokens,
                "stream": False,
                "response_format": {"type": "json_schema", "json_schema": {
                    "name": "MimoSingleCompactAnnotationDraftV2",
                    "schema": MimoSingleCompactAnnotationDraftV2.model_json_schema(), "strict": True,
                }},
                "extra_body": {
                    "use_audio_in_video": True,
                    "chat_template_kwargs": {
                        "thinking": self.config.thinking == "enabled",
                        "enable_thinking": self.config.thinking == "enabled",
                    },
                },
            }
            if self.config.thinking == "disabled":
                payload["reasoning_effort"] = "none"
            attempted = True
            completion, _, _ = self._call(payload)
            choices = _value(completion, "choices")
            if not isinstance(choices, list) or not choices:
                raise TypeError("MiMo response has no choices")
            choice = choices[0]
            raw = _value(_value(choice, "message"), "content")
            if not isinstance(raw, str):
                raise TypeError("MiMo response content must be text")
            diagnostic = _completion_diagnostic(
                completion, choice, modality="target_av_with_auxiliary_raw_audio",
                http_attempt_count=1, thinking=self.config.thinking,
            )
            _validate_finish_reason(diagnostic)
            _validate_av_observation_usage(diagnostic, require_explicit_audio=False)
            if diagnostic.usage.audio_tokens == 0:
                diagnostic.warnings.append("embedded_audio_tokens_zero")
            return raw, (None, raw, None, None), [diagnostic], {}
        except Exception as exc:
            diagnostic.request_error = f"{type(exc).__name__}: {exc}"
            raise MimoBackendFailure(
                code=exc.code if isinstance(exc, MimoBackendFailure) else "mimo_request_failed",
                reason=exc.reason if isinstance(exc, MimoBackendFailure) else diagnostic.request_error,
                issues=exc.issues if isinstance(exc, MimoBackendFailure) else (),
                raw_responses=(raw,) if raw is not None else (),
                diagnostics=(diagnostic,) if attempted else (),
                model_call_count=int(attempted), http_attempt_count=int(attempted),
                speech_av_raw_response=raw,
            ) from exc

    def reconcile(
        self,
        job: Any,
        *,
        segment_ids: list[str],
        transcribed_segment_ids: list[str],
        allowed_entity_ids: set[str],
        allowed_reference_labels: set[str],
        auxiliary_audio_paths: dict[str, Path] | None = None,
    ) -> MimoBackendResult:
        raw, _, diagnostics, _ = self._request(
            job, allowed_reference_labels=allowed_reference_labels,
            auxiliary_audio_paths=auxiliary_audio_paths,
        )
        canonical_raw, corrections = _canonicalize_single_compact_raw(raw)
        draft, issues = parse_structured_json_issues(canonical_raw, MimoSingleCompactAnnotationDraftV2)
        annotation = None
        if draft is not None:
            draft, speaker_corrections = _normalize_single_compact_speaker_subjects(draft, job)
            for code, count in speaker_corrections.items():
                corrections[code] = corrections.get(code, 0) + count
            compact_issues, warnings = _validate_single_compact_annotation(
                draft, job, allowed_entity_ids=allowed_entity_ids,
                allowed_reference_labels=allowed_reference_labels,
            )
            issues.extend(compact_issues)
            diagnostics[0].warnings.extend(warnings)
            if not issues:
                try:
                    annotation = _project_compact_to_legacy_annotation(draft, job)
                except ValueError as exc:
                    issues.append(ValidationIssue(
                        "compact_projection_failed", "annotation", str(exc),
                    ))
        if annotation is not None:
            speech = direct_speech_facts(annotation, list(job.segments))
            _, direct_issues, direct_warnings = protect_direct_dialogue(
                annotation.h3_semantics.shot1_caption,
                speech, allowed_labels=allowed_reference_labels,
            )
            issues.extend(direct_issues)
            issues.extend(_validate_direct_transcribed_dialogue(
                annotation.h3_semantics.shot1_caption, speech,
            ))
            diagnostics[0].warnings.extend(direct_warnings)
        for code, count in sorted(corrections.items()):
            diagnostics[0].warnings.extend([code] * count)
        if issues:
            raise MimoBackendFailure(
                code="mimo_structured_output_failed",
                reason="MiMo single compact annotation failed validation",
                raw_responses=(raw,), diagnostics=tuple(diagnostics), issues=tuple(issues),
                model_call_count=1, http_attempt_count=1,
                speech_av_raw_response=raw, annotation=annotation,
            )
        assert annotation is not None
        return MimoBackendResult(
            annotation=annotation, raw_responses=(raw,), diagnostics=tuple(diagnostics),
            model_call_count=1, http_attempt_count=1, http_retry_count=0,
            recheck_count=0, input_modality=self._input_modality,
            deterministic_correction_counts=dict(sorted(corrections.items())),
            speech_av_raw_response=raw,
        )

    def _maybe_polish_speaker_markers(self, annotation: MimoAVAnnotationDraft, **_: Any) -> tuple[
        MimoAVAnnotationDraft, MimoSpeakerMarkerPolishAudit, None,
    ]:
        return annotation, MimoSpeakerMarkerPolishAudit(), None
