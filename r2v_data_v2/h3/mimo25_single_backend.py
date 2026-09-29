"""One-request RA2VA shadow adapter over the frozen final annotation contract."""

from __future__ import annotations

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
from r2v_data_v2.structured_output import ValidationIssue, parse_structured_json_issues

SINGLE_PROMPT_VERSION = "h3_mimo26_ra2va_single_v5_compact"
SINGLE_BACKEND_VERSION = "r2v.h3.mimo25_backend.71"
SINGLE_COMPACT_SCHEMA_VERSION = "r2v.h3.mimo26_single_compact.1"

SINGLE_SYSTEM_PROMPT = """Return one MimoSingleCompactAnnotationDraft JSON object after observing the complete original target AV, frozen Pictures, and resolved stems. Original AV is the final audiovisual authority; stems aid acoustic recall. Official H3 demonstrations may contain final speaker markers or Picture provenance; those are final-H3 examples, not fields you own in this compact annotation.

STEP 1 — VISUAL AND H3 CONTENT
Observe the entire video and frozen references. Describe composition, foreground/background, appearance, lighting, camera behavior, actions, gaze/expression, and progression through the end. Give every required Subject one concise stable appearance description and one concise retention row in supplied order; do not put <Picture N> or Picture provenance in subject_definitions[*].description. Write one style_opening sentence and detailed chronological shot1_caption. Cite visibly preserved entity/background <Subject N> naturally at first appearance. For each supplied segment output only its segment_id and visible_entity_ids in visual_observation.segment_views. A sampled mouth-open or mouth-closed frame alone does not establish speaker identity; video sampling is temporally sparse.

STEP 2 — ACOUSTIC SPEAKER GROUPING
Use original audio and the speech stem. Fill audio_observation.segment_decisions for EVERY supplied DiariZen segment, including non-transcribed ones, in order. Assign stable g1, g2, ... acoustic groups by first actual vocal-source appearance; pauses, language changes, and segment boundaries alone do not create groups. Use primary_speaker_group=null when the source cannot be reliably grouped. Give transcribed speech a concise audible delivery_style and use null for non-transcribed segments. Profile each distinct non-null group appearing in transcribed segments, ordered by first transcribed appearance, using its corresponding speech-stem windows; do not re-decide grouping or binding in the profile.

STEP 3 — AV BINDING
For each segment compare its acoustic source with the exact-segment visible_entity_ids and the original AV. visible_entity requires entity_id from that same segment's visible_entity_ids and a supported visible speaker, not mere co-presence. offscreen, no_reliable_entity, and uncertain require entity_id=null. When several visible people remain plausible, leave binding unresolved instead of guessing or automatically declaring offscreen. No LR-ASD or upstream entity proposal is supplied. Do not output mouth/articulation evidence, confidence, or evidence codes.

STEP 4 — H3 PROSE AND AUDIO
Keep the detailed visual progression in shot1_caption and place each immutable required_dialogue_block exactly once in chronological playback order, with natural visible or offscreen speaker lead-in. Do not translate, paraphrase, normalize, merge, or omit dialogue. Non-transcribed segments have no dialogue block. Do NOT emit any (S1), (S2), or other (Sx) speaker markers: speaker IDs are pipeline-owned and projected deterministically after the response. Keep grounding rationale out of the caption. Write a concise summary without task prefix. Put continuous ambience and physical/environmental sound in overall_soundscape and audience-only score in non_diegetic_music, not in shot1_caption; the caption owns visual prose, speaker presentation, and exact dialogue. The materializer owns <Audio N>, Picture provenance, relationships, retention, and final H3 formatting.

OUTPUT INVENTORY
Treat required_output_inventory as exact and ordered. subject_definitions and visual_retention_analysis each have one row per supplied Subject in supplied order. visual_observation.segment_views, audio_observation.segment_decisions, and av_grounding.segment_groundings each have one row per supplied segment in chronological order. Only required_output_inventory.transcribed_dialogue_blocks_in_order supplies dialogue; if it is empty, output no <d> blocks and no speaker_voice_profiles. Use only supplied labels and IDs.

Return only compact JSON, not intermediate reasoning."""


class MimoSingleCompactSegmentView(SchemaModel):
    segment_id: str = Field(min_length=1)
    visible_entity_ids: list[str]

    @model_validator(mode="after")
    def validate_visible_entities(self) -> MimoSingleCompactSegmentView:
        if len(self.visible_entity_ids) != len(set(self.visible_entity_ids)):
            raise ValueError("compact visible entity IDs must be unique")
        return self


class MimoSingleCompactVisualObservation(SchemaModel):
    visual_blocks: list[MimoVisualBlock] = Field(min_length=1)
    segment_views: list[MimoSingleCompactSegmentView]


class MimoSingleCompactAudioDecision(SchemaModel):
    segment_id: str = Field(min_length=1)
    primary_speaker_group: str | None = Field(pattern=r"^g[1-9]\d*$")
    delivery_style: StrictStr | None

    @model_validator(mode="after")
    def validate_delivery(self) -> MimoSingleCompactAudioDecision:
        if self.delivery_style is not None and not self.delivery_style.strip():
            raise ValueError("compact delivery style must be non-empty or null")
        return self


class MimoSingleCompactAudioObservation(SchemaModel):
    segment_decisions: list[MimoSingleCompactAudioDecision]
    speaker_voice_profiles: list[MimoSpeakerVoiceProfile]


class MimoSingleCompactGrounding(SchemaModel):
    segment_id: str = Field(min_length=1)
    binding_status: Literal["visible_entity", "offscreen", "no_reliable_entity", "uncertain"]
    entity_id: str | None

    @model_validator(mode="after")
    def validate_binding(self) -> MimoSingleCompactGrounding:
        if (self.binding_status == "visible_entity") != (self.entity_id is not None):
            raise ValueError("compact visible binding requires entity_id; other bindings require null")
        return self


class MimoSingleCompactAVGrounding(SchemaModel):
    segment_groundings: list[MimoSingleCompactGrounding]


class MimoSingleCompactH3Semantics(MimoH3Semantics):
    pass


class MimoSingleCompactAnnotationDraft(SchemaModel):
    schema_version: Literal["r2v.h3.mimo26_single_compact.1"]
    visual_observation: MimoSingleCompactVisualObservation
    audio_observation: MimoSingleCompactAudioObservation
    av_grounding: MimoSingleCompactAVGrounding
    h3_semantics: MimoSingleCompactH3Semantics


def _single_semantic_icl_messages() -> list[dict[str, str]]:
    example = {
        "schema_version": SINGLE_COMPACT_SCHEMA_VERSION,
        "visual_observation": {
            "visual_blocks": [{
                "block_id": "v1",
                "text": "<Subject 1> watches quietly beside the doorway while <Subject 2> turns toward the camera side. Their positions remain separate in the steady composition.",
            }],
            "segment_views": [
                {
                    "segment_id": "segment_0001",
                    "visible_entity_ids": ["e1", "e2"],
                },
                {
                    "segment_id": "segment_0002",
                    "visible_entity_ids": ["e1", "e2"],
                },
            ],
        },
        "audio_observation": {
            "segment_decisions": [
                {
                    "segment_id": "segment_0001", "primary_speaker_group": "g1",
                    "delivery_style": "calm conversational delivery",
                },
                {
                    "segment_id": "segment_0002", "primary_speaker_group": "g2",
                    "delivery_style": "brief questioning delivery",
                },
            ],
            "speaker_voice_profiles": [
                {"speaker_group": "g1", "voice_characteristics": "a mid-register voice with a clear timbre and measured cadence"},
                {"speaker_group": "g2", "voice_characteristics": "a lower voice with restrained questioning delivery"},
            ],
        },
        "av_grounding": {
            "segment_groundings": [
                {
                    "segment_id": "segment_0001", "binding_status": "visible_entity",
                    "entity_id": "e2",
                },
                {
                    "segment_id": "segment_0002", "binding_status": "offscreen",
                    "entity_id": None,
                },
            ],
        },
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
    draft: MimoSingleCompactAnnotationDraft,
    job: Any,
    *,
    allowed_entity_ids: set[str],
    allowed_reference_labels: set[str],
) -> tuple[list[ValidationIssue], list[str]]:
    issues: list[ValidationIssue] = []
    warnings: list[str] = []
    segment_ids = [segment.segment_id for segment in job.segments]
    for field, rows in (
        ("visual_observation.segment_views", draft.visual_observation.segment_views),
        ("audio_observation.segment_decisions", draft.audio_observation.segment_decisions),
        ("av_grounding.segment_groundings", draft.av_grounding.segment_groundings),
    ):
        if [row.segment_id for row in rows] != segment_ids:
            issues.append(ValidationIssue(
                "segment_inventory_mismatch", field,
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
    views = {row.segment_id: row for row in draft.visual_observation.segment_views}
    for view in draft.visual_observation.segment_views:
        unknown = sorted(set(view.visible_entity_ids) - allowed_entity_ids)
        if unknown:
            issues.append(ValidationIssue("unknown_visual_entity", view.segment_id, str(unknown)))
    entities_by_group: dict[str, set[str]] = {}
    groups_by_entity: dict[str, set[str]] = {}
    decisions = {row.segment_id: row for row in draft.audio_observation.segment_decisions}
    for grounding in draft.av_grounding.segment_groundings:
        if grounding.binding_status != "visible_entity":
            continue
        if grounding.entity_id not in allowed_entity_ids:
            issues.append(ValidationIssue(
                "unknown_entity", grounding.segment_id, str(grounding.entity_id),
            ))
        if grounding.entity_id not in (
            views[grounding.segment_id].visible_entity_ids
            if grounding.segment_id in views else ()
        ):
            issues.append(ValidationIssue(
                "visible_entity_absent_from_visual_segment", grounding.segment_id,
                "visible binding requires exact-segment visible entity",
            ))
        decision = decisions.get(grounding.segment_id)
        if decision is not None and decision.primary_speaker_group is not None:
            group = decision.primary_speaker_group
            entity = grounding.entity_id
            assert entity is not None
            entities_by_group.setdefault(group, set()).add(entity)
            groups_by_entity.setdefault(entity, set()).add(group)
    warnings.extend(
        "compact_group_maps_multiple_entities"
        for entities in entities_by_group.values() if len(entities) > 1
    )
    warnings.extend(
        "compact_entity_maps_multiple_groups"
        for groups in groups_by_entity.values() if len(groups) > 1
    )
    first_groups = list(dict.fromkeys(
        row.primary_speaker_group for row in draft.audio_observation.segment_decisions
        if row.primary_speaker_group is not None
    ))
    if first_groups != [f"g{index}" for index in range(1, len(first_groups) + 1)]:
        issues.append(ValidationIssue(
            "non_contiguous_speaker_groups", "audio_observation.segment_decisions",
            "speaker groups must be contiguous by first chronological appearance",
        ))
    transcribed = [segment for segment in job.segments if segment.asr_status == "transcribed"]
    expected_profiles = list(dict.fromkeys(
        decisions[segment.segment_id].primary_speaker_group
        for segment in transcribed if segment.segment_id in decisions
        and decisions[segment.segment_id].primary_speaker_group is not None
    ))
    if [profile.speaker_group for profile in draft.audio_observation.speaker_voice_profiles] != expected_profiles:
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
        *(block.text for block in draft.visual_observation.visual_blocks),
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
    draft: MimoSingleCompactAnnotationDraft, job: Any,
) -> MimoAVAnnotationDraft:
    # Legacy-only fields are conservative compatibility placeholders, not MiMo evidence.
    values = {
        "schema_version": MIMO25_SCHEMA_VERSION,
        "visual_observation": {
            "visual_blocks": [item.model_dump(mode="json") for item in draft.visual_observation.visual_blocks],
            "segment_views": [
                {**item.model_dump(mode="json"), "entity_observations": []}
                for item in draft.visual_observation.segment_views
            ],
        },
        "audio_observation": {
            "segment_decisions": [
                {
                    **item.model_dump(mode="json"),
                    "resolution": "resolved" if item.primary_speaker_group is not None else "uncertain",
                    "vocal_composition": "uncertain",
                    "secondary_vocal_activity": {
                        "present": False, "speaker_relation": "none", "kind": None,
                    },
                    "confidence": "low", "audio_evidence_codes": ["insufficient_evidence"],
                }
                for item in draft.audio_observation.segment_decisions
            ],
            "speaker_voice_profiles": [
                item.model_dump(mode="json") for item in draft.audio_observation.speaker_voice_profiles
            ],
        },
        "av_grounding": {"segment_groundings": [
            {
                **item.model_dump(mode="json"),
                "primary_speaker_group": audio.primary_speaker_group,
                "speech_presentation": {
                    "visible_entity": "onscreen_spoken", "offscreen": "offscreen_spoken",
                    "no_reliable_entity": "uncertain", "uncertain": "uncertain",
                }[item.binding_status],
                "confidence": "low", "evidence_codes": ["insufficient_evidence"],
            }
            for item, audio in zip(
                draft.av_grounding.segment_groundings,
                draft.audio_observation.segment_decisions,
                strict=True,
            )
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
                "segment_ids_in_order": [segment.segment_id for segment in job.segments],
                "transcribed_dialogue_blocks_in_order": [
                    {"segment_id": segment["segment_id"],
                     "required_dialogue_block": segment["required_dialogue_block"]}
                    for segment in segments if segment["asr_status"] == "transcribed"
                ],
            }
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
                    "name": "MimoSingleCompactAnnotationDraft",
                    "schema": MimoSingleCompactAnnotationDraft.model_json_schema(), "strict": True,
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
        draft, issues = parse_structured_json_issues(raw, MimoSingleCompactAnnotationDraft)
        annotation = None
        if draft is not None:
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
            speech_av_raw_response=raw,
        )

    def _maybe_polish_speaker_markers(self, annotation: MimoAVAnnotationDraft, **_: Any) -> tuple[
        MimoAVAnnotationDraft, MimoSpeakerMarkerPolishAudit, None,
    ]:
        return annotation, MimoSpeakerMarkerPolishAudit(), None
