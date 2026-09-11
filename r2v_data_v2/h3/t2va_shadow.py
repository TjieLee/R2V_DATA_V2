"""Independent no-reference AV core, factual input projection, and T2VA publication."""

from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
import uuid
from collections import Counter
from pathlib import Path
from typing import Annotated, Literal, Protocol, Self

from pydantic import Field, StrictStr, TypeAdapter, model_validator

from r2v_data_v2.h3.diarization_binding import RawDiarizationSegment
from r2v_data_v2.h3.jea_audio_production import CanonicalAudioClip
from r2v_data_v2.h3.jea_target_audio_caption import (
    AUDIO_TIMELINE_DURATION_TOLERANCE_SECONDS,
)
from r2v_data_v2.h3.mimo25_backend import (
    AUDIO_FINALIZE_SYSTEM_PROMPT,
    MIMO25_AUDIO_FINALIZE_PROMPT_VERSION,
    MimoAudioFinalizeDraft,
    MimoSecondaryVocalActivity,
    VocalComposition,
)
from r2v_data_v2.h3.qwen3_asr import Qwen3ASRSegment
from r2v_data_v2.h3.resolved_audio_stems import (
    downstream_stem_root,
    load_resolved_stems,
)
from r2v_data_v2.h3.sam_audio_stem_shadow import (
    _publish_directory,
    sha256_file,
    stem_shadow_root,
    validate_stem_asr_lineage,
)
from r2v_data_v2.h3.schemas import SchemaModel
from r2v_data_v2.h3.t2va_source import T2VAShotSelection

Text = Annotated[StrictStr, Field(min_length=1)]
Hash = Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{64}$")]
SafeID = Annotated[StrictStr, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")]
SpeakerID = Annotated[StrictStr, Field(pattern=r"^S[1-9]\d*$")]
SECTIONS = (
    "integrated_multimodal_description",
    "overall_soundscape",
    "non_diegetic_music",
)
_FORBIDDEN = re.compile(
    r"<(?:Picture|Subject|Audio|Video)\s+[^>]+>|"
    r"\b(?:subject_definitions|summary|retention_analysis|detailed_description|"
    r"integrated_multimodal_description|overall_soundscape|non_diegetic_music)\s*:",
    re.IGNORECASE,
)
_DIALOGUE = re.compile(r"<d>(.*?)</d>", re.DOTALL)
_SPEAKER = re.compile(r"\(S(\d+)\)")
_MODEL_SPEAKER_TOKEN = re.compile(r"\bS[1-9]\d*\b")
_SHOT = re.compile(r"\[Shot ([1-9]\d*)\]")
_CUT = re.compile(r"\s+At (\d{2,}):([0-5]\d)\.(\d{3}),")


def fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def write_json(path: Path, value: SchemaModel | dict) -> None:
    data = value.model_dump(mode="json") if isinstance(value, SchemaModel) else value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    )


def read_rows(path: Path, model: type[SchemaModel]) -> list:
    return [
        model.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


class T2VASpeechFact(SchemaModel):
    segment_id: SafeID
    source_speaker_cluster: Text
    start_time: float = Field(ge=0, allow_inf_nan=False)
    end_time: float = Field(gt=0, allow_inf_nan=False)
    language: Text | None
    text: Text | None

    @model_validator(mode="after")
    def validate_fact(self) -> Self:
        if self.end_time <= self.start_time or (self.text is None) != (
            self.language is None
        ):
            raise ValueError("T2VA speech interval or ASR language/text differs")
        if self.text is not None and (
            not self.text.strip() or not self.language.strip()
        ):
            raise ValueError(
                "T2VA transcribed speech requires exact non-empty language/text"
            )
        return self


class T2VASpeakerAssignment(SchemaModel):
    segment_id: SafeID
    source_speaker_cluster: Text
    speaker_id: SpeakerID
    speech_presentation: Literal[
        "onscreen_spoken", "offscreen_spoken", "voice_over", "uncertain"
    ]
    primary_speaker_ownership_resolved: bool = Field(strict=True)
    vocal_composition: VocalComposition
    secondary_vocal_activity: MimoSecondaryVocalActivity

    @property
    def primary_speaker_group(self) -> str | None:
        # Eligibility evidence does not change the frozen consumer Sx assignment.
        return self.speaker_id if self.primary_speaker_ownership_resolved else None


class T2VAProsePart(SchemaModel):
    kind: Literal["prose"]
    text: Text = Field(max_length=12000)


class T2VASpeechPart(SchemaModel):
    kind: Literal["speech"]
    segment_id: SafeID
    lead_in: Text = Field(
        max_length=512,
        description="Source/presentation/delivery prose only; no dialogue words or speaker markers",
    )


T2VASequencePart = Annotated[
    T2VAProsePart | T2VASpeechPart, Field(discriminator="kind")
]


class T2VAMimoDraft(SchemaModel):
    summary: Text = Field(max_length=1024)
    speaker_assignments: list[T2VASpeakerAssignment]
    integrated_sequence: list[T2VASequencePart] = Field(min_length=1)
    warnings: list[Text]

    @model_validator(mode="after")
    def validate_no_references(self) -> Self:
        values = [
            self.summary,
            *(
                p.text if p.kind == "prose" else p.lead_in
                for p in self.integrated_sequence
            ),
            *self.warnings,
        ]
        for value in values:
            if not value.strip() or _FORBIDDEN.search(value) or "[[" in value:
                raise ValueError(
                    "T2VA fields cannot contain conditioning labels, headers or placeholders"
                )
            if (
                "<" in value
                or ">" in value
                or _SPEAKER.search(value)
                or _MODEL_SPEAKER_TOKEN.search(value)
            ):
                raise ValueError(
                    "T2VA model prose cannot contain dialogue/conditioning syntax or speaker markers"
                )
        if "[" in self.summary or "]" in self.summary or re.search(
            r"\b(?:subject_definitions|summary|retention_analysis|detailed_description|integrated_multimodal_description|overall_soundscape|non_diegetic_music)\s*:"
            r"|\b(?:audio|voice|music)\s+(?:reuse|reference)\b",
            self.summary,
            re.IGNORECASE,
        ):
            raise ValueError("T2VA internal summary cannot contain task/section syntax")
        if any(
            _SHOT.search(p.lead_in)
            for p in self.integrated_sequence
            if p.kind == "speech"
        ):
            raise ValueError("T2VA speech lead-in cannot contain shot syntax")
        _validate_repetition(
            [p.text for p in self.integrated_sequence if p.kind == "prose"]
        )
        return self


def _validate_repetition(parts: list[str]) -> None:
    sentences = [
        " ".join(s.casefold().split())
        for s in re.split(r"(?<=[.!?])\s+|\n+", "\n".join(parts))
    ]
    if any(
        count >= 3
        for sentence, count in Counter(sentences).items()
        if len(sentence) >= 80
    ):
        raise ValueError("T2VA repeated substantial sentence loop")
    for size in range(2, min(8, len(sentences) // 2) + 1):
        for start in range(len(sentences) - 2 * size + 1):
            block = sentences[start : start + size]
            if (
                sum(map(len, block)) >= 160
                and block == sentences[start + size : start + 2 * size]
            ):
                raise ValueError("T2VA repeated contiguous sentence block")


class H3NoReferenceAVCore(T2VAMimoDraft):
    schema_version: Literal["r2v.h3.no_reference_av_core.6"] = (
        "r2v.h3.no_reference_av_core.6"
    )
    overall_soundscape: Text
    non_diegetic_music: Text
    speech_facts: list[T2VASpeechFact]
    target_duration_seconds: float = Field(gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_core(self) -> Self:
        for field in SECTIONS[1:]:
            value = getattr(self, field)
            if (
                "<d>" in value
                or "</d>" in value
                or _SPEAKER.search(value)
                or _SHOT.search(value)
                or _MODEL_SPEAKER_TOKEN.search(value)
                or _FORBIDDEN.search(value)
                or not value.strip()
            ):
                raise ValueError(
                    "dialogue/speaker/shot syntax belongs only in integrated description"
                )
        _validate_t2va_content(self.speech_facts, self.target_duration_seconds, self)
        return self

    @property
    def integrated_multimodal_description(self) -> str:
        return _render_sequence(self.speech_facts, self)


class T2VAAudioEvidence(SchemaModel):
    resolved_root: Text
    resolved_record_fingerprint: Hash
    full_audio_path: Text
    full_audio_sha256: Hash
    speech_path: Text
    speech_sha256: Hash
    music_path: Text
    music_sha256: Hash
    sfx_path: Text
    sfx_sha256: Hash


def check_audio_files(evidence: T2VAAudioEvidence) -> None:
    for kind in ("full_audio", "speech", "music", "sfx"):
        path = Path(getattr(evidence, f"{kind}_path"))
        if not path.is_absolute() or sha256_file(path) != getattr(
            evidence, f"{kind}_sha256"
        ):
            raise ValueError(f"T2VA {kind} evidence hash differs")


def audio_evidence(clip: CanonicalAudioClip, record, root: Path) -> T2VAAudioEvidence:
    if record.status != "ready" or record.clip_uid != clip.clip_uid:
        raise ValueError("T2VA audio requires ready matching resolved stems")
    values = {
        "resolved_root": str(root),
        "resolved_record_fingerprint": record.record_fingerprint,
        "full_audio_path": clip.target_full_audio_path,
        "full_audio_sha256": clip.target_full_audio_sha256,
    }
    for kind in ("speech", "music", "sfx"):
        stem = record.stem(kind)
        if (
            stem.source_audio_path,
            stem.source_audio_sha256,
            stem.source_end_sample,
        ) != (
            clip.target_full_audio_path,
            clip.target_full_audio_sha256,
            clip.frame_count,
        ):
            raise ValueError("T2VA resolved audio differs from canonical target")
        values[f"{kind}_path"] = stem.canonical_stem_path
        values[f"{kind}_sha256"] = stem.canonical_stem_sha256
    evidence = T2VAAudioEvidence(**values)
    check_audio_files(evidence)
    return evidence


class T2VAJob(SchemaModel):
    clip_uid: SafeID
    clip_display_path: Text
    target_video_path: Text
    target_video_sha256: Hash
    target_duration_seconds: float = Field(gt=0, allow_inf_nan=False)
    speech_facts: list[T2VASpeechFact]
    audio_evidence: T2VAAudioEvidence | None = None
    upstream_failure: Text | None = None

    @model_validator(mode="after")
    def validate_speech(self) -> Self:
        ids = [s.segment_id for s in self.speech_facts]
        if len(ids) != len(set(ids)) or self.speech_facts != sorted(
            self.speech_facts,
            key=lambda s: (s.start_time, s.end_time, s.segment_id),
        ):
            raise ValueError("T2VA speech facts must be unique and chronological")
        if any(
            s.end_time
            > self.target_duration_seconds + AUDIO_TIMELINE_DURATION_TOLERANCE_SECONDS
            for s in self.speech_facts
        ):
            raise ValueError("T2VA speech exceeds canonical timeline tolerance")
        return self


def validate_t2va_draft(
    job: T2VAJob, draft: T2VAMimoDraft, audio: MimoAudioFinalizeDraft | None = None
) -> H3NoReferenceAVCore:
    """Bind model placement to immutable ASR facts, without semantic repair."""
    if isinstance(draft, H3NoReferenceAVCore) and (
        draft.speech_facts != job.speech_facts
        or draft.target_duration_seconds != job.target_duration_seconds
    ):
        raise ValueError("T2VA core speech facts differ from authoritative job")
    values = draft.model_dump(include=set(T2VAMimoDraft.model_fields))
    if audio is None:
        if not isinstance(draft, H3NoReferenceAVCore):
            raise ValueError("T2VA requires dedicated audio-finalize output")
        audio = MimoAudioFinalizeDraft(
            overall_soundscape=draft.overall_soundscape,
            non_diegetic_music=draft.non_diegetic_music,
        )
    return H3NoReferenceAVCore(
        **values,
        **audio.model_dump(),
        speech_facts=job.speech_facts,
        target_duration_seconds=job.target_duration_seconds,
    )


def render_t2va_speech(
    fact: T2VASpeechFact, assignment: T2VASpeakerAssignment, part: T2VASpeechPart
) -> str:
    lead_in = part.lead_in
    if (
        assignment.speech_presentation == "voice_over"
        and "says in an off-screen voiceover" not in lead_in
    ):
        lead_in += "; the voice says in an off-screen voiceover"
    return f"{lead_in} ({assignment.speaker_id}) <d>[{fact.language}] {fact.text}</d>"


def _render_sequence(facts: list[T2VASpeechFact], draft: T2VAMimoDraft) -> str:
    by_id = {s.segment_id: s for s in facts}
    assignments = {a.segment_id: a for a in draft.speaker_assignments}
    return " ".join(
        p.text
        if p.kind == "prose"
        else render_t2va_speech(by_id[p.segment_id], assignments[p.segment_id], p)
        for p in draft.integrated_sequence
    )


def _validate_t2va_content(
    facts: list[T2VASpeechFact], duration: float, draft: T2VAMimoDraft
) -> None:
    if len({s.segment_id for s in facts}) != len(facts) or facts != sorted(
        facts, key=lambda s: (s.start_time, s.end_time, s.segment_id)
    ):
        raise ValueError("T2VA core speech facts must be unique and chronological")
    if any(
        s.end_time > duration + AUDIO_TIMELINE_DURATION_TOLERANCE_SECONDS for s in facts
    ):
        raise ValueError("T2VA speech exceeds canonical timeline tolerance")
    _validate_assignments(facts, draft.speaker_assignments)
    if any(
        s.text is not None
        and re.search(r"(?<!\w)" + re.escape(s.text) + r"(?!\w)", draft.summary)
        for s in facts
    ):
        raise ValueError("T2VA internal summary must not copy authoritative ASR text")
    placed = [p.segment_id for p in draft.integrated_sequence if p.kind == "speech"]
    required = [s.segment_id for s in facts if s.text is not None]
    if placed != required:
        message = "T2VA speech placement inventory differs from chronological ASR"
        # Diagnostic only, and only after the authoritative inventory has failed.
        if set(required) - set(placed) and any(
            re.search(
                r"\b(?:the|a|an)\s+(?:man|woman|person|speaker|voice)\s+"
                r"(?:(?:then|again|also)\s+){0,2}(?:speaks|talks|says)\b"
                r"|\bvoice\s+speaking\s+in\b",
                p.text,
                re.IGNORECASE,
            )
            for p in draft.integrated_sequence
            if p.kind == "prose"
        ):
            message += "; possible_prose_substitution_for_required_speech_slots"
        raise ValueError(message)
    for part in draft.integrated_sequence:
        value = part.text if part.kind == "prose" else part.lead_in
        if any(
            s.text is not None
            and re.search(r"(?<!\w)" + re.escape(s.text) + r"(?!\w)", value)
            for s in facts
        ):
            raise ValueError("T2VA model prose must not copy authoritative ASR text")
    caption = _render_sequence(facts, draft)
    validate_t2va_dialogue(facts, draft.speaker_assignments, caption)
    _validate_t2va_shots(_DIALOGUE.sub("", caption), duration)


def _validate_assignments(facts, assignments) -> None:
    if [(a.segment_id, a.source_speaker_cluster) for a in assignments] != [
        (s.segment_id, s.source_speaker_cluster) for s in facts
    ]:
        raise ValueError("T2VA speaker assignment inventory differs from speech facts")
    clusters: dict[str, str] = {}
    speakers: list[str] = []
    for assignment in assignments:
        prior = clusters.setdefault(
            assignment.source_speaker_cluster, assignment.speaker_id
        )
        if prior != assignment.speaker_id:
            raise ValueError(
                "T2VA acoustic source cluster cannot split across speakers"
            )
        if assignment.speaker_id not in speakers:
            speakers.append(assignment.speaker_id)
    if speakers != [f"S{i + 1}" for i in range(len(speakers))]:
        raise ValueError(
            "T2VA speaker IDs must be contiguous by first vocal appearance"
        )


def parse_t2va_semantic(job: T2VAJob, raw: str) -> tuple[T2VAMimoDraft, dict[str, int]]:
    from r2v_data_v2.structured_output import normalize_structured_json_envelope

    payload = json.loads(normalize_structured_json_envelope(raw))
    if not isinstance(payload, dict):
        raise TypeError("T2VA response must be a JSON object")
    assignments = TypeAdapter(list[T2VASpeakerAssignment]).validate_python(
        payload.get("speaker_assignments")
    )
    _validate_assignments(job.speech_facts, assignments)
    by_id = {a.segment_id: a.speaker_id for a in assignments}
    corrections = Counter()
    token = re.compile(r"\(S[1-9]\d*\)|\bS[1-9]\d*\b")
    for part in payload.get("integrated_sequence", []):
        if (
            not isinstance(part, dict)
            or part.get("kind") != "speech"
            or not isinstance(part.get("lead_in"), str)
        ):
            continue
        lead = part["lead_in"]
        found = {m.group().strip("()") for m in token.finditer(lead)}
        if found and found == {by_id.get(part.get("segment_id"))}:
            clean = re.sub(r" {2,}", " ", token.sub("", lead)).strip()
            if (
                not any(c.isalpha() for c in clean)
                or clean[0] in ",;:.!?"
                or clean.count("(") != clean.count(")")
                or re.search(r"[,;:]\s*[,;:]", clean)
            ):
                raise ValueError(
                    "T2VA redundant speaker cleanup leaves malformed lead-in"
                )
            part["lead_in"] = clean
            corrections["redundant_speech_lead_in_speaker_marker"] += 1
    draft = T2VAMimoDraft.model_validate(payload)
    _validate_t2va_content(job.speech_facts, job.target_duration_seconds, draft)
    return draft, dict(corrections)


def validate_t2va_dialogue(
    facts: list[T2VASpeechFact], assignments: list[T2VASpeakerAssignment], caption: str
) -> None:
    speakers = {a.speaker_id for a in assignments}
    if {f"S{m}" for m in _SPEAKER.findall(caption)} - set(speakers):
        raise ValueError("T2VA unknown speaker marker")
    blocks = list(_DIALOGUE.finditer(caption))
    speech = [
        (s, a) for s, a in zip(facts, assignments, strict=True) if s.text is not None
    ]
    if (
        len(blocks) != len(speech)
        or caption.count("<d>") != len(blocks)
        or (caption.count("</d>") != len(blocks))
    ):
        raise ValueError("T2VA exact dialogue inventory differs")
    previous_end = 0
    for block, (fact, assignment) in zip(blocks, speech, strict=True):
        if block.group(1) != f"[{fact.language}] {fact.text}":
            raise ValueError("T2VA exact ASR text/language differs")
        markers = _SPEAKER.findall(caption[previous_end : block.start()])
        if markers != [assignment.speaker_id[1:]] or not caption[
            previous_end : block.start()
        ].rstrip().endswith(f"({assignment.speaker_id})"):
            raise ValueError(
                "T2VA dialogue requires its authoritative speaker marker in the lead-in"
            )
        previous_end = block.end()


def _validate_t2va_shots(visual: str, duration: float) -> None:
    shots = list(_SHOT.finditer(visual))
    if not visual.startswith("[Shot 1]") or [int(s.group(1)) for s in shots] != list(
        range(1, len(shots) + 1)
    ):
        raise ValueError("T2VA description must begin with sequential [Shot 1]")
    previous_time = 0.0
    for index, shot in enumerate(shots):
        tail = visual[shot.end() :]
        if index == 0:
            if re.match(r"\s*(?:At\s+)?\d+:\d", tail, re.IGNORECASE):
                raise ValueError("T2VA first shot must not have a timestamp")
            continue
        cut = _CUT.match(tail)
        if cut is None:
            raise ValueError("T2VA later shots require At MM:SS.mmm cut timestamps")
        minute, second, milli = map(int, cut.groups())
        time = minute * 60 + second + milli / 1000
        if not previous_time < time < duration:
            raise ValueError(
                "T2VA cut timestamps must be chronological and inside target"
            )
        previous_time = time


def render_t2va_prompt(core: H3NoReferenceAVCore) -> str:
    core = H3NoReferenceAVCore.model_validate(core.model_dump())
    return "\n\n".join(f"{field}:\n{getattr(core, field)}" for field in SECTIONS) + "\n"


class T2VABackendProvenance(SchemaModel):
    schema_version: Literal["r2v.h3.t2va_mimo_backend.7"] = "r2v.h3.t2va_mimo_backend.7"
    prompt_version: Literal["h3_t2va_joint_av_v7"] = "h3_t2va_joint_av_v7"
    audio_finalize_prompt_version: Literal["h3_mimo25_audio_finalize_v6"] = (
        MIMO25_AUDIO_FINALIZE_PROMPT_VERSION
    )
    audio_finalize_prompt_sha256: Hash = hashlib.sha256(
        AUDIO_FINALIZE_SYSTEM_PROMPT.encode()
    ).hexdigest()
    audio_finalize_schema_sha256: Hash = fingerprint(
        MimoAudioFinalizeDraft.model_json_schema()
    )
    prompt_sha256: Hash
    response_schema_sha256: Hash
    transport: Literal["sglang", "xiaomi"]
    model: Literal["mimo-v2.5"] = "mimo-v2.5"
    base_url: Text
    media_root: Text
    media_mode: Literal["base64", "http"]
    media_base_url: str | None
    temperature: Literal[0] = 0
    thinking: Literal["disabled"] = "disabled"
    video_fps: Literal[4] = 4
    media_resolution: Literal["default"] = "default"
    max_completion_tokens: int = Field(gt=0)
    maximum_attempts: Literal[1] = 1


class T2VAInventory(SchemaModel):
    schema_version: Literal["r2v.h3.t2va_inventory.3"] = "r2v.h3.t2va_inventory.3"
    shot_selection: T2VAShotSelection
    audio_production_root: Text
    audio_shadow_run_id: SafeID
    t2va_run_id: SafeID
    source_hashes: dict[str, Hash]
    selection_mode: Literal["all", "case_manifest", "random"]
    sample_seed: int | None
    clip_uids: list[SafeID] = Field(min_length=1)
    jobs: list[T2VAJob]
    backend: T2VABackendProvenance
    inventory_fingerprint: Hash

    @model_validator(mode="after")
    def validate_inventory(self) -> Self:
        selection = self.shot_selection
        if (
            [s.clip_uid for s in selection.shots] != self.clip_uids
            or self.source_hashes.get(selection.shot_manifest_path)
            != selection.shot_manifest_sha256
            or any(
                (j.target_video_path, j.target_video_sha256, j.target_duration_seconds)
                != (s.video_path, s.video_sha256, s.duration_seconds)
                for j, s in zip(self.jobs, selection.shots, strict=True)
            )
        ):
            raise ValueError("T2VA jobs differ from authoritative JEA shot selection")
        if self.clip_uids != [j.clip_uid for j in self.jobs] or len(
            set(self.clip_uids)
        ) != len(self.clip_uids):
            raise ValueError("T2VA ordered inventory differs")
        if (
            fingerprint(self.model_dump(mode="json", exclude={"inventory_fingerprint"}))
            != self.inventory_fingerprint
        ):
            raise ValueError("T2VA inventory fingerprint differs")
        return self


class T2VACaseManifest(SchemaModel):
    schema_version: Literal[
        "r2v.h3.t2va_case_manifest.1", "r2v.h3.mimo25_case_manifest.1"
    ] = "r2v.h3.t2va_case_manifest.1"
    clip_uids: list[SafeID] = Field(min_length=1)


def select_clip_uids(
    source_ids: list[str],
    *,
    case_manifest: Path | None = None,
    sample_size: int | None = None,
    sample_seed: int | None = None,
) -> list[str]:
    if len(set(source_ids)) != len(source_ids) or not source_ids:
        raise ValueError("source clip inventory must be nonempty and unique")
    if case_manifest is not None and (
        sample_size is not None or sample_seed is not None
    ):
        raise ValueError("case manifest and random sampling are mutually exclusive")
    if (sample_size is None) != (sample_seed is None):
        raise ValueError("sample size and seed must be supplied together")
    if case_manifest is not None:
        selected = T2VACaseManifest.model_validate_json(
            case_manifest.read_text()
        ).clip_uids
    elif sample_size is not None:
        if not 0 < sample_size <= len(source_ids):
            raise ValueError("sample size outside source inventory")
        selected = random.Random(sample_seed).sample(source_ids, sample_size)
    else:
        selected = source_ids.copy()
    if len(set(selected)) != len(selected) or set(selected) - set(source_ids):
        raise ValueError("selected clips must be unique members of source inventory")
    return selected


def t2va_root(audio_production_root: Path, run_id: str) -> Path:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", run_id) is None:
        raise ValueError("invalid T2VA run ID")
    path = (
        audio_production_root.expanduser().resolve()
        / "t2va_shadow_v1"
        / "runs"
        / run_id
    )
    if path.resolve() != path:
        raise ValueError("T2VA root must not redirect to another namespace")
    return path


def build_t2va_inventory(
    *,
    shot_manifest: Path,
    clips_root: Path | None = None,
    source_videos_root: Path | None = None,
    audio_production_root: Path,
    audio_shadow_run_id: str,
    t2va_run_id: str,
    backend: T2VABackendProvenance,
    case_manifest: Path | None = None,
    sample_size: int | None = None,
    sample_seed: int | None = None,
    shot_index_root: Path | None = None,
) -> T2VAInventory:
    from r2v_data_v2.h3.t2va_source import select_t2va_shots, validate_cached_target

    selection = select_t2va_shots(
        shot_manifest,
        clips_root=clips_root,
        source_videos_root=source_videos_root,
        case_manifest=case_manifest,
        sample_size=sample_size,
        sample_seed=sample_seed,
        shot_index_root=shot_index_root,
    )
    selected = [s.clip_uid for s in selection.shots]
    shots = {s.clip_uid: s for s in selection.shots}
    production = audio_production_root.expanduser().resolve()
    t2va_root(production, t2va_run_id)
    canonical_path = production / "audio/canonical_clips.jsonl"
    canonical = (
        read_rows(canonical_path, CanonicalAudioClip) if canonical_path.exists() else []
    )
    by_clip = {c.clip_uid: c for c in canonical}
    if len(by_clip) != len(canonical):
        raise ValueError("duplicate cached canonical Audio identity")
    for uid in selected:
        if uid in by_clip:
            validate_cached_target(shots[uid], by_clip[uid])

    def job_for(uid, facts, reason, evidence=None):
        shot = shots[uid]
        return T2VAJob(
            clip_uid=uid,
            clip_display_path=shot.clip_display_path,
            target_video_path=shot.video_path,
            target_video_sha256=shot.video_sha256,
            target_duration_seconds=shot.duration_seconds,
            speech_facts=facts,
            audio_evidence=evidence,
            upstream_failure=reason,
        )

    def finish(jobs, paths):
        hashes = {selection.shot_manifest_path: selection.shot_manifest_sha256}
        hashes.update({str(p): sha256_file(p) for p in paths})
        if selection.case_manifest_path:
            hashes[selection.case_manifest_path] = selection.case_manifest_sha256
        values = {
            "schema_version": "r2v.h3.t2va_inventory.3",
            "shot_selection": selection.model_dump(mode="json"),
            "audio_production_root": str(production),
            "audio_shadow_run_id": audio_shadow_run_id,
            "t2va_run_id": t2va_run_id,
            "source_hashes": hashes,
            "selection_mode": selection.selection_mode,
            "sample_seed": sample_seed,
            "clip_uids": selected,
            "jobs": [j.model_dump(mode="json") for j in jobs],
            "backend": backend.model_dump(mode="json"),
        }
        return T2VAInventory(**values, inventory_fingerprint=fingerprint(values))

    root = stem_shadow_root(production, audio_shadow_run_id)
    if not (root / "asr").exists():
        return finish(
            [job_for(uid, [], "audio_preprocessing_required") for uid in selected],
            [canonical_path] if canonical_path.exists() else [],
        )
    asr_provenance, diarization = validate_stem_asr_lineage(
        root / "asr", expected_shadow_root=root
    )
    resolved_root = downstream_stem_root(production, audio_shadow_run_id)
    if (
        diarization.source_stem_root != str(resolved_root)
        or diarization.route != "resolved"
        or diarization.diarization_source_kind != "resolved_speech_stem"
        or asr_provenance.asr_source_kind != "resolved_speech_stem_segments"
    ):
        raise ValueError(
            "T2VA requires finalized named-run resolved speech, never legacy SAM speech"
        )
    resolved, stem_records, _ = load_resolved_stems(resolved_root)
    stems_by_uid = {r.clip_uid: r for r in stem_records}
    if (
        Path(resolved.source_canonical_audio_manifest_path).resolve() != canonical_path
        or resolved.source_canonical_audio_manifest_sha256
        != sha256_file(canonical_path)
    ):
        raise ValueError(
            "T2VA canonical inventory differs from resolved speech lineage"
        )
    raw_path, asr_path = (
        root / "diarization/raw_segments.jsonl",
        root / "asr/segments.jsonl",
    )
    raw = read_rows(raw_path, RawDiarizationSegment)
    asr = read_rows(asr_path, Qwen3ASRSegment)
    raw_by_key = {(s.target_clip_uid, s.segment_id): s for s in raw}
    asr_by_key = {(s.clip_uid, s.segment_id): s for s in asr}
    if (
        len(raw_by_key) != len(raw)
        or len(asr_by_key) != len(asr)
        or raw_by_key.keys() != asr_by_key.keys()
    ):
        raise ValueError("T2VA DiariZen/ASR segment inventory differs")
    jobs = []
    for uid in selected:
        if uid not in by_clip or uid not in resolved.clip_uids:
            jobs.append(job_for(uid, [], "audio_preprocessing_required"))
            continue
        source_job = next(j for j in resolved.jobs if j.clip_uid == uid)
        clip = by_clip[uid]
        if (
            source_job.target_video_path != shots[uid].video_path
            or source_job.target_video_sha256 != shots[uid].video_sha256
            or source_job.source_audio_path != clip.target_full_audio_path
            or source_job.source_audio_sha256 != clip.target_full_audio_sha256
            or source_job.source_frame_count != clip.frame_count
        ):
            raise ValueError("T2VA selected JEA shot differs from resolved cache")
        reason = (
            None if uid in asr_provenance.clip_uids else "resolved_speech_unavailable"
        )
        facts = []
        for key, segment in sorted(
            raw_by_key.items(),
            key=lambda row: (
                row[1].start_time,
                row[1].end_time,
                row[1].segment_id,
            ),
        ):
            if key[0] != uid:
                continue
            result = asr_by_key[key]
            if (
                segment.source_audio_path
                != diarization.speech_stem_paths_by_clip.get(uid)
                or segment.source_audio_sha256
                != diarization.speech_stem_hashes_by_clip.get(uid)
            ):
                raise ValueError(
                    "T2VA raw segment source differs from finalized speech lineage"
                )
            for field in (
                "speaker_cluster_id",
                "source_audio_path",
                "source_start_sample",
                "source_end_sample",
                "source_sample_rate_hz",
                "source_channels",
                "start_time",
                "end_time",
            ):
                if getattr(segment, field) != getattr(result, field):
                    raise ValueError(
                        f"T2VA DiariZen/ASR reconciliation differs: {field}"
                    )
            if result.status == "failed":
                reason = "asr_failed"
                continue
            facts.append(
                T2VASpeechFact(
                    segment_id=segment.segment_id,
                    source_speaker_cluster=segment.speaker_cluster_id,
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    language=result.language,
                    text=result.text,
                )
            )
        stem_record = stems_by_uid[uid]
        evidence = (
            audio_evidence(clip, stem_record, resolved_root)
            if stem_record.status == "ready"
            else None
        )
        if evidence is None:
            reason = reason or "resolved_audio_unavailable"
        jobs.append(job_for(uid, facts, reason, evidence))
    paths = [
        canonical_path,
        raw_path,
        asr_path,
        root / "diarization/stem_provenance.json",
        root / "asr/stem_provenance.json",
        resolved_root / "inventory.json",
        resolved_root / "records.jsonl",
        resolved_root / "summary.json",
    ]
    return finish(jobs, paths)


class T2VACompletion(SchemaModel):
    model_call_count: int = Field(ge=0, le=1)
    response: str | None
    finish_reason: str | None
    usage: dict
    warnings: list[str]
    error: str | None


class T2VARawResponse(T2VACompletion):
    schema_version: Literal["r2v.h3.t2va_raw_response.2"] = "r2v.h3.t2va_raw_response.2"
    clip_uid: SafeID
    request_fingerprint: Hash
    model_call_count: int = Field(ge=0, le=2)
    semantic_model_call_count: int = Field(default=0, ge=0, le=1)
    audio_finalize: T2VACompletion | None = None
    deterministic_corrections: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_calls(self) -> Self:
        if self.model_call_count != self.semantic_model_call_count + (
            self.audio_finalize.model_call_count if self.audio_finalize else 0
        ):
            raise ValueError("T2VA per-stage model call counts differ")
        return self


class T2VARecord(SchemaModel):
    schema_version: Literal["r2v.h3.t2va_record.2"] = "r2v.h3.t2va_record.2"
    clip_uid: SafeID
    inventory_fingerprint: Hash
    request_fingerprint: Hash
    status: Literal["ready", "failed", "skipped"]
    failure_reason: str | None
    model_call_count: int = Field(ge=0, le=2)
    raw_sha256: Hash
    core_sha256: Hash | None
    prompt_sha256: Hash | None
    warnings: list[str]

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        if self.status == "ready":
            if (
                self.failure_reason
                or not self.core_sha256
                or not self.prompt_sha256
                or self.model_call_count != 2
            ):
                raise ValueError(
                    "ready T2VA record requires a validated two-call core/prompt"
                )
        elif not self.failure_reason or self.core_sha256 or self.prompt_sha256:
            raise ValueError("unavailable T2VA cannot publish core/prompt")
        if self.status == "skipped" and self.model_call_count:
            raise ValueError("skipped T2VA cannot call model")
        return self


class T2VASummary(SchemaModel):
    schema_version: Literal["r2v.h3.t2va_summary.2"] = "r2v.h3.t2va_summary.2"
    inventory_fingerprint: Hash
    clip_uids: list[SafeID]
    record_count: int
    ready_count: int
    failed_count: int
    skipped_count: int
    model_call_count: int
    production_artifacts_modified: Literal[False] = False


class T2VABackend(Protocol):
    def provenance(self) -> T2VABackendProvenance: ...
    def annotate(self, job: T2VAJob, request_fingerprint: str) -> T2VARawResponse: ...


def _summary(inventory: T2VAInventory, records: list[T2VARecord]) -> T2VASummary:
    return T2VASummary(
        inventory_fingerprint=inventory.inventory_fingerprint,
        clip_uids=inventory.clip_uids,
        record_count=len(records),
        ready_count=sum(r.status == "ready" for r in records),
        failed_count=sum(r.status == "failed" for r in records),
        skipped_count=sum(r.status == "skipped" for r in records),
        model_call_count=sum(r.model_call_count for r in records),
    )


def request_fingerprint(job: T2VAJob, backend: T2VABackendProvenance) -> str:
    return fingerprint(
        {"job": job.model_dump(mode="json"), "backend": backend.model_dump(mode="json")}
    )


def _check_sources(inventory: T2VAInventory) -> None:
    for name, expected in inventory.source_hashes.items():
        if sha256_file(Path(name)) != expected:
            raise ValueError("T2VA source provenance changed")
    root = stem_shadow_root(
        Path(inventory.audio_production_root), inventory.audio_shadow_run_id
    )
    if str(root / "asr/stem_provenance.json") in inventory.source_hashes:
        validate_stem_asr_lineage(root / "asr", expected_shadow_root=root)
        validate_t2va_audio_lineage(inventory)


def validate_t2va_audio_lineage(inventory: T2VAInventory) -> None:
    production = Path(inventory.audio_production_root)
    root = downstream_stem_root(production, inventory.audio_shadow_run_id)
    resolved, records, _ = load_resolved_stems(root)
    canonical_path = production / "audio/canonical_clips.jsonl"
    if (
        str(canonical_path) != resolved.source_canonical_audio_manifest_path
        or sha256_file(canonical_path)
        != resolved.source_canonical_audio_manifest_sha256
    ):
        raise ValueError("T2VA resolved/canonical audio lineage differs")
    clips = {c.clip_uid: c for c in read_rows(canonical_path, CanonicalAudioClip)}
    by_uid = {r.clip_uid: r for r in records}
    for job in inventory.jobs:
        if job.audio_evidence is not None:
            clip = clips[job.clip_uid]
            if (clip.target_video_path, clip.target_video_sha256) != (
                job.target_video_path,
                job.target_video_sha256,
            ) or audio_evidence(clip, by_uid[job.clip_uid], root) != job.audio_evidence:
                raise ValueError(
                    "T2VA job audio evidence differs from resolved lineage"
                )


def run_t2va_shadow(
    inventory: T2VAInventory, backend: T2VABackend, *, overwrite: bool = False
) -> T2VASummary:
    from r2v_data_v2.structured_output import parse_structured_json_response

    inventory = T2VAInventory.model_validate(inventory.model_dump())
    if inventory.backend != backend.provenance():
        raise ValueError("T2VA backend differs from inventory")
    destination = t2va_root(
        Path(inventory.audio_production_root), inventory.t2va_run_id
    )
    if destination.exists():
        if not overwrite:
            raise FileExistsError(destination)
        old_backend = json.loads((destination / "inventory.json").read_text()).get(
            "backend", {}
        )
        if (old_backend.get("schema_version"), old_backend.get("prompt_version")) != (
            inventory.backend.schema_version,
            inventory.backend.prompt_version,
        ):
            raise ValueError("T2VA contract changed; use a new T2VA run ID")
        old, _, _ = load_t2va_shadow(destination)
        if (old.audio_production_root, old.t2va_run_id) != (
            inventory.audio_production_root,
            inventory.t2va_run_id,
        ):
            raise ValueError("T2VA overwrite ownership differs")
    _check_sources(inventory)
    if any(
        destination == Path(p).resolve() or destination in Path(p).resolve().parents
        for p in [
            *inventory.source_hashes,
            *(j.target_video_path for j in inventory.jobs),
        ]
    ):
        raise ValueError("T2VA output overlaps a source artifact")
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    records = []
    try:
        temporary.mkdir(parents=True)
        write_json(temporary / "inventory.json", inventory)
        for job in inventory.jobs:
            identity = request_fingerprint(job, inventory.backend)
            raw = T2VARawResponse(
                clip_uid=job.clip_uid,
                request_fingerprint=identity,
                model_call_count=0,
                response=None,
                finish_reason=None,
                usage={},
                warnings=[],
                error=job.upstream_failure,
            )
            status, error, core = "skipped", job.upstream_failure, None
            if error is None:
                status = "failed"
                try:
                    if (
                        sha256_file(Path(job.target_video_path))
                        != job.target_video_sha256
                    ):
                        raise ValueError("original target video hash differs")
                    raw = backend.annotate(job, identity)
                    if (
                        raw.clip_uid != job.clip_uid
                        or raw.request_fingerprint != identity
                    ):
                        raise ValueError("T2VA raw response ownership differs")
                    if raw.error:
                        raise ValueError(raw.error)
                    draft, corrections = parse_t2va_semantic(job, raw.response or "")
                    if (
                        corrections != raw.deterministic_corrections
                        or raw.audio_finalize is None
                        or raw.audio_finalize.error
                    ):
                        raise ValueError(
                            "T2VA audio-finalize/correction provenance differs"
                        )
                    audio = parse_structured_json_response(
                        raw.audio_finalize.response or "", MimoAudioFinalizeDraft
                    )
                    core = validate_t2va_draft(job, draft, audio)
                    if (
                        sha256_file(Path(job.target_video_path))
                        != job.target_video_sha256
                    ):
                        core = None
                        raise ValueError(
                            "original target video changed during annotation"
                        )
                    status = "ready"
                except (ValueError, TypeError, OSError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
            raw_path = temporary / "raw" / f"{job.clip_uid}.json"
            write_json(raw_path, raw)
            core_hash = prompt_hash = None
            if core is not None:
                core_path = temporary / "core" / f"{job.clip_uid}.json"
                prompt_path = temporary / "prompts" / f"{job.clip_uid}.txt"
                write_json(core_path, core)
                prompt_path.parent.mkdir(exist_ok=True)
                prompt_path.write_text(render_t2va_prompt(core), encoding="utf-8")
                core_hash, prompt_hash = (
                    sha256_file(core_path),
                    sha256_file(prompt_path),
                )
            records.append(
                T2VARecord(
                    clip_uid=job.clip_uid,
                    inventory_fingerprint=inventory.inventory_fingerprint,
                    request_fingerprint=identity,
                    status=status,
                    failure_reason=error,
                    model_call_count=raw.model_call_count,
                    raw_sha256=sha256_file(raw_path),
                    core_sha256=core_hash,
                    prompt_sha256=prompt_hash,
                    warnings=[
                        *raw.warnings,
                        *(raw.audio_finalize.warnings if raw.audio_finalize else []),
                        *(core.warnings if core else []),
                    ],
                )
            )
        (temporary / "records.jsonl").write_text(
            "".join(r.model_dump_json() + "\n" for r in records)
        )
        summary = _summary(inventory, records)
        write_json(temporary / "summary.json", summary)
        load_t2va_shadow(temporary)
        _check_sources(inventory)
        _publish_directory(temporary, destination, overwrite=overwrite)
        return summary
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def load_t2va_shadow(root: Path) -> tuple[T2VAInventory, list[T2VARecord], T2VASummary]:
    inventory = T2VAInventory.model_validate_json((root / "inventory.json").read_text())
    records = read_rows(root / "records.jsonl", T2VARecord)
    summary = T2VASummary.model_validate_json((root / "summary.json").read_text())
    if [r.clip_uid for r in records] != inventory.clip_uids or summary != _summary(
        inventory, records
    ):
        raise ValueError("T2VA published record/summary inventory differs")
    for job, record in zip(inventory.jobs, records, strict=True):
        if (
            record.inventory_fingerprint != inventory.inventory_fingerprint
            or record.request_fingerprint != request_fingerprint(job, inventory.backend)
        ):
            raise ValueError("T2VA record provenance differs")
        raw_path = root / "raw" / f"{job.clip_uid}.json"
        raw = T2VARawResponse.model_validate_json(raw_path.read_text())
        if (
            sha256_file(raw_path) != record.raw_sha256
            or raw.clip_uid != job.clip_uid
            or raw.request_fingerprint != record.request_fingerprint
            or raw.model_call_count != record.model_call_count
        ):
            raise ValueError("T2VA raw provenance differs")
        if record.status == "ready":
            core_path, prompt_path = (
                root / "core" / f"{job.clip_uid}.json",
                root / "prompts" / f"{job.clip_uid}.txt",
            )
            core = H3NoReferenceAVCore.model_validate_json(core_path.read_text())
            validate_t2va_draft(job, core)
            if (
                sha256_file(core_path) != record.core_sha256
                or sha256_file(prompt_path) != record.prompt_sha256
                or prompt_path.read_text() != render_t2va_prompt(core)
            ):
                raise ValueError("T2VA core/prompt provenance differs")
    return inventory, records, summary
