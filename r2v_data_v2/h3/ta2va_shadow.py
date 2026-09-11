"""Target-only Audio conditioning over frozen T2VA, with no visual identities."""

from __future__ import annotations

import json
import re
import shutil
import uuid
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from r2v_data_v2.h3.audio_reuse import (
    SAMPLE_RATE,
    ReuseSampleRange,
    _check_stem,
    _write_verified,
    probe_canonical_target_frames,
)
from r2v_data_v2.h3.diarization_binding import RawDiarizationSegment
from r2v_data_v2.h3.mimo25_backend import (
    _VOICE_IDENTITY_PROFILE,
    MIMO25_SPEAKER_PROFILE_PROMPT_VERSION,
    SPEAKER_PROFILE_SYSTEM_PROMPT,
    _completion_diagnostic,
    _validate_finish_reason,
)
from r2v_data_v2.h3.mimo25_h3_materializer import project_audio_relationships
from r2v_data_v2.h3.qwen3_asr import Qwen3ASRSegment
from r2v_data_v2.h3.qwen38_h3_recaption import (
    Qwen38H3StructuredResponse,
    RecaptionAudioContract,
    RecaptionSpeechFact,
    _canonical_audio_definition,
    _canonical_audio_retention,
    audio_task_prefix,
    render_h3_prompt,
)
from r2v_data_v2.h3.resolved_audio_stems import load_resolved_stems
from r2v_data_v2.h3.sam_audio_stem_shadow import sha256_file, stem_shadow_root
from r2v_data_v2.h3.schemas import SchemaModel
from r2v_data_v2.h3.speaker_ownership import speaker_ownership_reasons
from r2v_data_v2.h3.t2va_mimo_backend import T2VAMimoBackend, T2VAMimoConfig
from r2v_data_v2.h3.t2va_shadow import (
    H3NoReferenceAVCore,
    T2VAJob,
    _check_sources,
    fingerprint,
    load_t2va_shadow,
    read_rows,
    t2va_root,
    validate_t2va_audio_lineage,
    validate_t2va_dialogue,
)
from r2v_data_v2.structured_output import parse_structured_json_response

VERSION = "r2v.h3.ta2va_shadow.1"
Variant = Literal["target_speech_reuse", "full_audio_reuse"]
Sx = str


class TA2VAProfile(SchemaModel):
    # speaker_group names the already frozen Sx; there is no new gN identity.
    speaker_group: str = Field(pattern=r"^S[1-9]\d*$")
    voice_characteristics: str = Field(strict=True, min_length=1)

    @model_validator(mode="after")
    def validate_traits(self):
        value = self.voice_characteristics
        if (
            not value.strip()
            or _VOICE_IDENTITY_PROFILE.search(value)
            or re.search(r"[<>\[\]]|\bS[1-9]\d*\b", value)
        ):
            raise ValueError("TA2VA profile contains identity/serialization claims")
        return self


class TA2VAProfileDraft(SchemaModel):
    speaker_voice_profiles: list[TA2VAProfile]


class TA2VAAsset(SchemaModel):
    kind: Literal["speaker_speech_reuse", "music_reuse"]
    speaker_id: str | None = Field(default=None, pattern=r"^S[1-9]\d*$")
    source_path: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_frame_count: int = Field(gt=0)
    ranges: list[ReuseSampleRange]
    output_path: str
    output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_rate_hz: Literal[32000] = 32000
    channels: Literal[2] = 2
    container: Literal["FLAC"] = "FLAC"
    sample_format: Literal["PCM_16"] = "PCM_16"
    frame_count: int = Field(gt=0)
    copied_frame_count: int = Field(gt=0)
    silence_tail_frame_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_asset(self):
        if (
            not Path(self.source_path).is_absolute()
            or not Path(self.output_path).is_absolute()
        ):
            raise ValueError("TA2VA asset paths must be absolute")
        if self.kind == "speaker_speech_reuse":
            if not self.speaker_id or not self.ranges:
                raise ValueError("TA2VA speech asset requires actual Sx ranges")
            if any(
                r.target_end_sample > min(self.frame_count, self.source_frame_count)
                for r in self.ranges
            ):
                raise ValueError("TA2VA asset range exceeds source/target")
        elif self.speaker_id is not None or self.ranges:
            raise ValueError("TA2VA music has no speaker ownership")
        return self


class TA2VAProduct(SchemaModel):
    schema_version: Literal["r2v.h3.ta2va_shadow.1"] = VERSION
    clip_uid: str
    variant: Variant
    source_inventory_fingerprint: str
    source_core_sha256: str
    source_resolved_record_fingerprint: str
    status: Literal["ready", "failed"]
    failure_reason: str | None
    audio_references: list[RecaptionAudioContract]
    assets: list[TA2VAAsset]
    speaker_profiles: list[TA2VAProfile]
    prompt: str | None
    warnings: list[str]
    model_call_count: int = Field(ge=0, le=1)
    record_fingerprint: str

    @model_validator(mode="after")
    def validate_record(self):
        if self.record_fingerprint != fingerprint(
            self.model_dump(exclude={"record_fingerprint"})
        ):
            raise ValueError("TA2VA product fingerprint differs")
        if self.variant == "full_audio_reuse" and self.model_call_count:
            raise ValueError("full-audio TA2VA is model-free")
        if self.status == "failed":
            if not self.failure_reason or self.prompt or self.audio_references:
                raise ValueError("failed TA2VA cannot publish conditioning")
            return self
        if self.failure_reason or not self.prompt:
            raise ValueError("ready TA2VA requires a prompt")
        validate_references(self.variant, self.audio_references)
        if re.search(r"<(?:Picture|Subject|Video)\b", self.prompt):
            raise ValueError("TA2VA must not publish visual references")
        if self.variant == "target_speech_reuse":
            expected = [
                a.speaker_id
                for a in self.audio_references
                if a.kind == "speaker_speech_reuse"
            ]
            if [p.speaker_group for p in self.speaker_profiles] != expected:
                raise ValueError("TA2VA profile inventory differs")
        return self


def validate_references(variant: Variant, audios: list[RecaptionAudioContract]):
    if not 1 <= len(audios) <= 3 or [a.audio_index for a in audios] != list(
        range(1, len(audios) + 1)
    ):
        raise ValueError("TA2VA Audio capacity/order differs")
    if any(a.subject_label is not None or a.entity_id is not None for a in audios):
        raise ValueError("TA2VA cannot bind Subjects/entities")
    kinds = [a.kind for a in audios]
    if variant == "full_audio_reuse":
        if kinds != ["full_audio_reuse"]:
            raise ValueError("TA2VA full audio must stand alone")
    elif (
        "speaker_speech_reuse" not in kinds
        or any(k not in {"speaker_speech_reuse", "music_reuse"} for k in kinds)
        or kinds.count("music_reuse") > 1
        or ("music_reuse" in kinds and kinds[-1] != "music_reuse")
    ):
        raise ValueError("TA2VA target speech requires actual speech Audio")
    speakers = [a.speaker_id for a in audios if a.speaker_id is not None]
    if speakers != sorted(set(speakers), key=lambda s: int(s[1:])):
        raise ValueError("TA2VA speakers must preserve Sx order")
    if any(
        a.kind == "speaker_speech_reuse" and not a.voice_characteristics for a in audios
    ):
        raise ValueError("TA2VA speech Audio requires a validated acoustic profile")


def load_source(root: Path):
    root = root.resolve(strict=True)
    inventory, records, _ = load_t2va_shadow(root)
    if root != t2va_root(Path(inventory.audio_production_root), inventory.t2va_run_id):
        raise ValueError("TA2VA requires owned T2VA root")
    _check_sources(inventory)
    validate_t2va_audio_lineage(inventory)
    shadow = stem_shadow_root(
        Path(inventory.audio_production_root), inventory.audio_shadow_run_id
    )
    _, stems, _ = load_resolved_stems(shadow / "resolved_stems_v1")
    raw = read_rows(shadow / "diarization/raw_segments.jsonl", RawDiarizationSegment)
    asr = read_rows(shadow / "asr/segments.jsonl", Qwen3ASRSegment)
    raw_map = {(r.target_clip_uid, r.segment_id): r for r in raw}
    asr_map = {(r.clip_uid, r.segment_id): r for r in asr}
    if len(raw_map) != len(raw) or len(asr_map) != len(asr):
        raise ValueError("duplicate TA2VA upstream segment")
    jobs = {j.clip_uid: j for j in inventory.jobs}
    sources = []
    for record in records:
        if record.status != "ready":
            continue
        job = jobs[record.clip_uid]
        core = H3NoReferenceAVCore.model_validate_json(
            (root / "core" / f"{job.clip_uid}.json").read_text()
        )
        segments = []
        for fact in job.speech_facts:
            key = (job.clip_uid, fact.segment_id)
            r, a = raw_map[key], asr_map[key]
            if (r.speaker_cluster_id, r.start_time, r.end_time) != (
                fact.source_speaker_cluster,
                fact.start_time,
                fact.end_time,
            ) or (a.text, a.language) != (fact.text, fact.language):
                raise ValueError("TA2VA frozen speech identity differs")
            for name in (
                "speaker_cluster_id",
                "source_audio_path",
                "source_start_sample",
                "source_end_sample",
                "source_sample_rate_hz",
                "source_channels",
                "start_time",
                "end_time",
            ):
                if getattr(r, name) != getattr(a, name):
                    raise ValueError("TA2VA raw/ASR sample identity differs")
            segments.append(r)
        sources.append((job, core, segments, record.core_sha256))
    hashes = dict(inventory.source_hashes)
    for relative in ("inventory.json", "records.jsonl", "summary.json"):
        path = root / relative
        hashes[str(path)] = sha256_file(path)
    for job, _, _, _ in sources:
        for folder, ext in (("core", "json"), ("raw", "json"), ("prompts", "txt")):
            path = root / folder / f"{job.clip_uid}.{ext}"
            hashes[str(path)] = sha256_file(path)
        hashes[job.target_video_path] = job.target_video_sha256
        for kind in ("full_audio", "speech", "music", "sfx"):
            evidence = job.audio_evidence
            hashes[getattr(evidence, f"{kind}_path")] = getattr(
                evidence, f"{kind}_sha256"
            )
    return inventory, sources, {s.clip_uid: s for s in stems}, hashes


def sample_ranges(
    job: T2VAJob, core: H3NoReferenceAVCore, segments, frames, stem_frames
):
    """Use authoritative rational samples, never float-time reconstruction."""
    groups: dict[Sx, list[ReuseSampleRange]] = {}
    all_ranges = []
    warnings = []
    blocked = set()
    for fact, assignment, raw in zip(
        core.speech_facts, core.speaker_assignments, segments, strict=True
    ):
        if (raw.source_audio_path, raw.source_audio_sha256) != (
            job.audio_evidence.speech_path,
            job.audio_evidence.speech_sha256,
        ):
            raise ValueError("TA2VA raw segment does not own resolved AuK speech")
        start = round(
            Fraction(raw.source_start_sample * SAMPLE_RATE, raw.source_sample_rate_hz)
        )
        end = round(
            Fraction(raw.source_end_sample * SAMPLE_RATE, raw.source_sample_rate_hz)
        )
        r = ReuseSampleRange(
            **raw.model_dump(
                include={
                    "segment_id",
                    "source_start_sample",
                    "source_end_sample",
                    "source_sample_rate_hz",
                    "start_time",
                    "end_time",
                }
            ),
            target_start_sample=start,
            target_end_sample=end,
        )
        sx = assignment.speaker_id
        reasons = speaker_ownership_reasons(assignment)
        if reasons:
            blocked.add(sx)
            warnings.extend(f"{sx}:{fact.segment_id}:{reason}" for reason in reasons)
        all_ranges.append((sx, r))
        if fact.text is not None:
            groups.setdefault(sx, []).append(r)
            if end > frames or end > stem_frames:
                blocked.add(sx)
                warnings.append(f"{sx}:speech_range_outside_available_audio")
    for i, (sx, a) in enumerate(all_ranges):
        for sy, b in all_ranges[i + 1 :]:
            if sx != sy and max(a.target_start_sample, b.target_start_sample) < min(
                a.target_end_sample, b.target_end_sample
            ):
                blocked.update((sx, sy))
                warnings.append(f"{sx}/{sy}:cross_speaker_overlap")
    return {
        s: groups[s]
        for s in sorted(groups, key=lambda s: int(s[1:]))
        if s not in blocked
    }, warnings


def select_speakers(groups, music_positive: bool):
    speakers = list(groups)
    selected = speakers[:3]
    warnings = [f"{s}:omitted_audio_capacity" for s in speakers[3:]]
    include_music = music_positive and len(selected) < 3
    if music_positive and not include_music:
        warnings.append("music:omitted_audio_capacity")
    return selected, include_music, warnings


def _music_present(description: str) -> bool:
    music = description.strip()
    return bool(music) and music.casefold() not in {"n/a", "unknown"}


def speech_track(pcm, frames, ranges):
    result = np.zeros((frames, 2), dtype=np.int16)
    for r in ranges:
        if r.target_end_sample > min(frames, len(pcm)):
            raise ValueError("TA2VA speech range outside available samples")
        result[r.target_start_sample : r.target_end_sample] = pcm[
            r.target_start_sample : r.target_end_sample
        ]
    return result


def render_product(core, variant, audios):
    validate_references(variant, audios)
    original = core.integrated_multimodal_description
    validate_t2va_dialogue(core.speech_facts, core.speaker_assignments, original)
    speech = [
        RecaptionSpeechFact(
            fact_id=f.segment_id,
            segment_id=f.segment_id,
            speaker_cluster_id=f.source_speaker_cluster,
            speaker_id=a.speaker_id,
            start_time=f.start_time,
            end_time=f.end_time,
            text=f.text,
            language=f.language,
            locked_dialogue_block=f"<d>[{f.language}] {f.text}</d>",
        )
        for f, a in zip(core.speech_facts, core.speaker_assignments, strict=True)
        if f.text is not None
    ]
    caption, music = project_audio_relationships(
        original, core.non_diegetic_music, speech, audios
    )
    blocks = re.findall(r"<d>[\s\S]*?</d>", caption)
    if blocks != [s.locked_dialogue_block for s in speech]:
        raise ValueError("TA2VA exact dialogue changed after Audio projection")
    # The canonical relation sits after Sx, so T2VA's immediate-adjacency check
    # cannot be applied unchanged. Check each original Sx and dialogue explicitly.
    previous = 0
    for block, fact in zip(
        re.finditer(r"<d>[\s\S]*?</d>", caption), speech, strict=True
    ):
        if re.findall(r"\((S[1-9]\d*)\)", caption[previous : block.start()]) != [
            fact.speaker_id
        ]:
            raise ValueError("TA2VA speaker changed after Audio projection")
        previous = block.end()
    response = Qwen38H3StructuredResponse(
        subject_definitions=[_canonical_audio_definition(a) for a in audios],
        summary=audio_task_prefix(audios) + " " + core.summary,
        retention_analysis=[_canonical_audio_retention(a) for a in audios],
        detailed_description=caption,
        overall_soundscape=core.overall_soundscape,
        non_diegetic_music=music,
    )
    for value in [
        *response.subject_definitions,
        response.summary,
        *response.retention_analysis,
        response.overall_soundscape,
        response.non_diegetic_music,
    ]:
        if "<d>" in value or "</d>" in value:
            raise ValueError("TA2VA dialogue outside detailed description")
    prompt = render_h3_prompt(response)
    if re.search(r"<(?:Subject|Picture|Video)\b", prompt):
        raise ValueError("TA2VA contains visual reference syntax")
    return prompt


class TA2VAProfileBackend:
    def __init__(self, config: T2VAMimoConfig, *, client=None):
        self.config = config
        self.client = T2VAMimoBackend(config, client=client).client

    def provenance(self):
        return {
            "version": VERSION,
            "prompt_version": MIMO25_SPEAKER_PROFILE_PROMPT_VERSION,
            "prompt_sha256": fingerprint({"text": SPEAKER_PROFILE_SYSTEM_PROMPT}),
            "response_schema_sha256": fingerprint(
                TA2VAProfileDraft.model_json_schema()
            ),
            "model": self.config.model,
            "base_url": self.config.base_url,
            "transport": self.config.transport,
            "temperature": 0,
            "max_completion_tokens": self.config.max_completion_tokens,
            "maximum_attempts": 1,
            "media_root": str(self.config.media_resolver.media_root),
            "media_mode": self.config.media_resolver.mode,
            "media_base_url": self.config.media_resolver.media_base_url,
        }

    def profile(self, targets, snippets):
        schema = TA2VAProfileDraft.model_json_schema()
        instruction = "SPEAKER-PROFILE TARGETS:\n" + json.dumps(targets)
        if self.config.transport == "xiaomi":
            instruction += "\nRESPONSE SCHEMA:\n" + json.dumps(schema)
        content = [{"type": "text", "text": instruction}]
        for label, path in snippets:
            content.extend(
                [
                    {"type": "text", "text": json.dumps(label)},
                    {
                        "type": "audio_url",
                        "audio_url": {"url": self.config.media_resolver.resolve(path)},
                    },
                ]
            )
        request = {
            "model": self.config.model,
            "temperature": 0,
            "stream": False,
            "max_completion_tokens": self.config.max_completion_tokens,
            "messages": [
                {"role": "system", "content": SPEAKER_PROFILE_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
        }
        if self.config.transport == "sglang":
            request.update(
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "TA2VAProfileDraft",
                        "schema": schema,
                        "strict": True,
                    },
                },
                reasoning_effort="none",
                extra_body={
                    "use_audio_in_video": False,
                    "chat_template_kwargs": {
                        "thinking": False,
                        "enable_thinking": False,
                    },
                },
            )
        else:
            request.update(
                response_format={"type": "json_object"},
                extra_body={
                    "thinking": {"type": "disabled"},
                    "use_audio_in_video": False,
                },
            )
        raw = {
            "response": None,
            "diagnostic": None,
            "model_call_count": 0,
            "error": None,
        }
        profiles = []
        try:
            raw["model_call_count"] = 1
            completion = self.client.chat.completions.create(**request)
            choice = completion.choices[0]
            raw["response"] = choice.message.content
            diagnostic = _completion_diagnostic(
                completion,
                choice,
                modality="speaker_profile_audio_only",
                http_attempt_count=1,
                thinking="disabled",
            )
            raw["diagnostic"] = diagnostic.model_dump(mode="json")
            _validate_finish_reason(diagnostic)
            parsed = parse_structured_json_response(raw["response"], TA2VAProfileDraft)
            profiles = parsed.speaker_voice_profiles
            if [p.speaker_group for p in profiles] != [
                t["speaker_group"] for t in targets
            ]:
                raise ValueError("TA2VA profile changed fixed Sx inventory/order")
        except Exception as exc:  # noqa: BLE001 - preserve the single attempt, no fallback.
            raw["error"] = f"{type(exc).__name__}: {exc}"
            profiles = []
        return profiles, raw


def _verify(hashes):
    for path, digest in hashes.items():
        if not Path(path).is_absolute() or sha256_file(Path(path)) != digest:
            raise ValueError(f"TA2VA source/media hash differs: {path}")


def _json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    )


def run_ta2va_shadow(
    t2va: Path, run_id: str, backend: TA2VAProfileBackend, *, allow_unverified=False
):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise ValueError("invalid TA2VA run ID")
    source, clips, stems, hashes = load_source(t2va)
    output = Path(source.audio_production_root) / "ta2va_shadow_v1/runs" / run_id
    if output.resolve() != output or any(
        Path(p).resolve().is_relative_to(output) for p in hashes
    ):
        raise ValueError("TA2VA output overlaps/redirects source")
    if output.exists():
        raise FileExistsError("TA2VA requires a fresh run ID")
    _verify(hashes)
    values = {
        "schema_version": VERSION,
        "source_t2va_root": str(t2va.resolve()),
        "source_inventory_fingerprint": source.inventory_fingerprint,
        "source_hashes": hashes,
        "clip_uids": [j.clip_uid for j, *_ in clips],
        "output_root": str(output),
        "backend": backend.provenance(),
        "allow_unverified": allow_unverified,
    }
    values["inventory_fingerprint"] = fingerprint(values)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{run_id}.tmp-{uuid.uuid4().hex}")
    products = []
    reuse_exclusions = {}
    unpublished_warnings = []
    try:
        for folder in ("assets", "raw_profiles", "prompts"):
            (temporary / folder).mkdir(parents=True, exist_ok=True)
        for job, core, segments, core_hash in clips:
            evidence = job.audio_evidence
            stem = stems[job.clip_uid]
            frames = probe_canonical_target_frames(
                Path(evidence.full_audio_path), evidence.full_audio_sha256
            )
            base = {
                "clip_uid": job.clip_uid,
                "source_inventory_fingerprint": source.inventory_fingerprint,
                "source_core_sha256": core_hash,
                "source_resolved_record_fingerprint": stem.record_fingerprint,
            }

            def publish(
                variant,
                refs,
                assets,
                profiles,
                warnings,
                calls,
                failure=None,
                *,
                core=core,
                base=base,
                job=job,
            ):
                prompt = None
                try:
                    if not failure:
                        prompt = render_product(core, variant, refs)
                except ValueError as exc:
                    failure = str(exc)
                fields = dict(
                    schema_version=VERSION,
                    **base,
                    variant=variant,
                    status="failed" if failure else "ready",
                    failure_reason=failure,
                    audio_references=[] if failure else refs,
                    assets=assets,
                    speaker_profiles=profiles,
                    prompt=prompt,
                    warnings=warnings,
                    model_call_count=calls,
                )
                serial = {
                    k: [
                        x.model_dump(mode="json") if isinstance(x, SchemaModel) else x
                        for x in v
                    ]
                    if isinstance(v, list)
                    else v
                    for k, v in fields.items()
                }
                product = TA2VAProduct(**fields, record_fingerprint=fingerprint(serial))
                products.append(product)
                if prompt:
                    (
                        temporary / "prompts" / f"{job.clip_uid}.{variant}.txt"
                    ).write_text(prompt)

            full = RecaptionAudioContract(
                audio_index=1,
                audio_label="<Audio 1>",
                kind="full_audio_reuse",
                path=evidence.full_audio_path,
                sha256=evidence.full_audio_sha256,
                retention_marker="fully_copy",
            )
            publish("full_audio_reuse", [full], [], [], [], 0)
            if not any(f.text is not None for f in core.speech_facts):
                continue
            if stem.separation_state != "success" and not allow_unverified:
                raise ValueError(
                    "TA2VA unverified stems require explicit --allow-unverified"
                )
            speech_pcm = _check_stem(
                stem.stem("speech"),
                Path(evidence.full_audio_path),
                evidence.full_audio_sha256,
                frames,
            )
            groups, exclusions = sample_ranges(
                job, core, segments, frames, len(speech_pcm)
            )
            if exclusions:
                reuse_exclusions[job.clip_uid] = exclusions
            selected, include_music, warnings = select_speakers(
                groups, _music_present(core.non_diegetic_music)
            )
            if not selected:
                unpublished_warnings.extend(exclusions)
                continue
            warnings = [*exclusions, *warnings]
            assets, snippets, targets = [], [], []
            directory = temporary / "assets" / job.clip_uid
            directory.mkdir()

            def write_asset(
                name,
                pcm,
                kind,
                sx=None,
                ranges=None,
                source_kind="speech",
                *,
                directory=directory,
                stem=stem,
                job=job,
                frames=frames,
            ):
                digest = _write_verified(directory / name, pcm)
                source_stem = stem.stem(source_kind)
                copied = min(frames, source_stem.canonical_frame_count)
                tail = frames - copied
                if ranges:
                    copied, previous_end = 0, 0
                    for r in ranges:
                        copied += max(
                            0,
                            r.target_end_sample
                            - max(previous_end, r.target_start_sample),
                        )
                        previous_end = max(previous_end, r.target_end_sample)
                    tail = frames - previous_end
                return TA2VAAsset(
                    kind=kind,
                    speaker_id=sx,
                    ranges=ranges or [],
                    source_path=source_stem.canonical_stem_path,
                    source_sha256=source_stem.canonical_stem_sha256,
                    source_frame_count=source_stem.canonical_frame_count,
                    output_path=str(output / "assets" / job.clip_uid / name),
                    output_sha256=digest,
                    frame_count=frames,
                    copied_frame_count=copied,
                    silence_tail_frame_count=tail,
                )

            for sx in selected:
                pcm = speech_track(speech_pcm, frames, groups[sx])
                for r in groups[sx]:
                    name = f"{sx}.{r.segment_id}.snippet.flac"
                    _write_verified(
                        directory / name,
                        speech_pcm[r.target_start_sample : r.target_end_sample],
                    )
                    label = {
                        "speaker_group": sx,
                        "speaker_id": sx,
                        **r.model_dump(mode="json"),
                    }
                    snippets.append((label, directory / name))
                assets.append(
                    write_asset(
                        f"{sx}.flac", pcm, "speaker_speech_reuse", sx, groups[sx]
                    )
                )
                targets.append(
                    {
                        "speaker_group": sx,
                        "speaker_id": sx,
                        "segments": [r.model_dump(mode="json") for r in groups[sx]],
                    }
                )
            try:
                profiles, raw = backend.profile(targets, snippets)
            except Exception as exc:  # noqa: BLE001 - pre-call media/config failure only.
                profiles, raw = (
                    [],
                    {
                        "response": None,
                        "diagnostic": None,
                        "model_call_count": 0,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
            raw["snippets"] = [
                {
                    "label": label,
                    "path": str(output / "assets" / job.clip_uid / p.name),
                    "sha256": sha256_file(p),
                }
                for label, p in snippets
            ]
            _json(temporary / "raw_profiles" / f"{job.clip_uid}.json", raw)
            if raw["error"]:
                publish(
                    "target_speech_reuse",
                    [],
                    assets,
                    [],
                    warnings,
                    raw["model_call_count"],
                    raw["error"],
                )
                continue
            if include_music:
                music_pcm = _check_stem(
                    stem.stem("music"),
                    Path(evidence.full_audio_path),
                    evidence.full_audio_sha256,
                    frames,
                )
                pcm = np.zeros((frames, 2), dtype=np.int16)
                copied = min(frames, len(music_pcm))
                pcm[:copied] = music_pcm[:copied]
                assets.append(
                    write_asset("music.flac", pcm, "music_reuse", source_kind="music")
                )
            profile_by_sx = {p.speaker_group: p.voice_characteristics for p in profiles}
            refs = [
                RecaptionAudioContract(
                    audio_index=i,
                    audio_label=f"<Audio {i}>",
                    kind=a.kind,
                    path=a.output_path,
                    sha256=a.output_sha256,
                    speaker_id=a.speaker_id,
                    voice_characteristics=profile_by_sx.get(a.speaker_id),
                    retention_marker="partially_copy",
                )
                for i, a in enumerate(assets, 1)
            ]
            publish(
                "target_speech_reuse",
                refs,
                assets,
                profiles,
                warnings,
                raw["model_call_count"],
            )
        _verify(hashes)
        load_source(t2va)
        summary = {
            "schema_version": VERSION,
            "inventory_fingerprint": values["inventory_fingerprint"],
            "clip_count": len(clips),
            "product_count": len(products),
            "ready_count": sum(p.status == "ready" for p in products),
            "failed_count": sum(p.status == "failed" for p in products),
            "variant_counts": dict(
                sorted(Counter(p.variant for p in products).items())
            ),
            "audio_kind_counts": dict(
                sorted(
                    Counter(
                        a.kind for p in products for a in p.audio_references
                    ).items()
                )
            ),
            "model_call_count": sum(p.model_call_count for p in products),
            "warning_counts": dict(
                sorted(
                    (
                        Counter(w for p in products for w in p.warnings)
                        + Counter(unpublished_warnings)
                    ).items()
                )
            ),
            "production_artifacts_modified": False,
            "reuse_exclusions": reuse_exclusions,
        }
        _json(temporary / "inventory.json", values)
        _json(temporary / "summary.json", summary)
        (temporary / "records.jsonl").write_text(
            "".join(p.model_dump_json() + "\n" for p in products)
        )
        artifacts = {
            str(p.relative_to(temporary)): sha256_file(p)
            for p in temporary.rglob("*")
            if p.is_file()
        }
        _json(temporary / "artifacts.json", artifacts)
        if output.exists():
            raise FileExistsError(output)
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output
