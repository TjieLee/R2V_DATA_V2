from __future__ import annotations

import hashlib
import html
import json
import math
import shutil
import sys
import uuid
import wave
from array import array
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from r2v_data_v2.h3.diarization_binding import (
    DiarizationInventory,
    RawDiarizationSegment,
)
from r2v_data_v2.h3.diarization_binding import (
    _inventory_fingerprint as _diarization_inventory_fingerprint,
)
from r2v_data_v2.h3.schemas import AudioBindingSidecar, SchemaModel

BACKGROUND_AUDIO_SCOUT_CLIP_VERSION = "r2v.h3.background_audio_scout_clip.1"
BACKGROUND_AUDIO_SCOUT_INVENTORY_VERSION = "r2v.h3.background_audio_scout_inventory.1"
BACKGROUND_AUDIO_SCOUT_SUMMARY_VERSION = "r2v.h3.background_audio_scout_summary.1"
BACKGROUND_AUDIO_SELECTION_VERSION = "r2v.h3.background_audio_pilot_selection.1"
BACKGROUND_AUDIO_SCOUT_OUTPUT_DIRECTORY = "background_audio_scout"
EXPECTED_PRODUCTION_TARGET_COUNT = 75

SCOUT_LABELS = ("background-rich", "clean", "uncertain")
SCOUT_FLAGS = (
    "music",
    "crowd",
    "traffic",
    "nature",
    "machinery",
    "impacts",
    "other_sound_event",
)


def _compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"JSONL line {line_number} in {path} must be an object")
        rows.append(value)
    return rows


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


class SampleSpan(SchemaModel):
    start_sample: int = Field(ge=0)
    end_sample: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_span(self) -> SampleSpan:
        if self.end_sample <= self.start_sample:
            raise ValueError("sample span must have positive duration")
        return self


class BackgroundAudioScoutJob(SchemaModel):
    target_clip_uid: str
    target_full_audio_path: str
    target_full_audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_audio_binding_path: str
    target_audio_binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_sample_rate_hz: int = Field(gt=0)
    source_frame_count: int = Field(gt=0)
    diarized_speech_spans: list[SampleSpan]

    @model_validator(mode="after")
    def validate_job(self) -> BackgroundAudioScoutJob:
        if not self.target_clip_uid.strip():
            raise ValueError("background-audio scout clip identity is empty")
        if self.diarized_speech_spans != sorted(
            self.diarized_speech_spans,
            key=lambda item: (item.start_sample, item.end_sample),
        ):
            raise ValueError("diarized speech spans must be ordered")
        previous_end = 0
        for span in self.diarized_speech_spans:
            if span.start_sample < previous_end:
                raise ValueError("diarized speech spans must already be unioned")
            if span.end_sample > self.source_frame_count:
                raise ValueError("diarized speech span exceeds canonical audio")
            previous_end = span.end_sample
        return self


class BackgroundAudioScoutInventory(SchemaModel):
    schema_version: Literal["r2v.h3.background_audio_scout_inventory.1"] = (
        BACKGROUND_AUDIO_SCOUT_INVENTORY_VERSION
    )
    source_diarization_root: str
    source_diarization_inventory_path: str
    source_diarization_inventory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_diarization_inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_diarization_raw_segments_path: str
    source_diarization_raw_segments_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_audio_root: str
    source_audio_evidence_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_clip_count: Literal[75] = EXPECTED_PRODUCTION_TARGET_COUNT
    selection_applied: Literal[False] = False
    parent_quota_applied: Literal[False] = False
    model_calls: Literal[0] = 0
    automatic_background_selection_applied: Literal[False] = False
    jobs: list[BackgroundAudioScoutJob]
    inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_inventory(self) -> BackgroundAudioScoutInventory:
        if len(self.jobs) != self.target_clip_count:
            raise ValueError("background-audio scout must contain all 75 targets")
        clip_ids = [item.target_clip_uid for item in self.jobs]
        if len(clip_ids) != len(set(clip_ids)):
            raise ValueError("background-audio scout targets must be unique")
        if self.inventory_fingerprint != _inventory_fingerprint(self):
            raise ValueError("background-audio scout inventory fingerprint is invalid")
        return self


class BackgroundAudioScoutClip(SchemaModel):
    schema_version: Literal["r2v.h3.background_audio_scout_clip.1"] = (
        BACKGROUND_AUDIO_SCOUT_CLIP_VERSION
    )
    target_clip_uid: str
    clip_duration_seconds: float = Field(gt=0)
    diarized_speaker_union_seconds: float = Field(ge=0)
    non_speech_seconds: float = Field(ge=0)
    non_speech_ratio: float = Field(ge=0, le=1)
    full_audio_rms_dbfs: float | None = None
    non_speech_rms_dbfs: float | None = None
    non_speech_peak_dbfs: float | None = None
    target_full_audio_path: str
    target_full_audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_metrics(self) -> BackgroundAudioScoutClip:
        if not math.isclose(
            self.diarized_speaker_union_seconds + self.non_speech_seconds,
            self.clip_duration_seconds,
            rel_tol=0,
            abs_tol=1e-9,
        ):
            raise ValueError("speech and non-speech durations must cover the clip")
        expected_ratio = self.non_speech_seconds / self.clip_duration_seconds
        if not math.isclose(
            self.non_speech_ratio,
            expected_ratio,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError("non-speech ratio is inconsistent")
        for value in (
            self.full_audio_rms_dbfs,
            self.non_speech_rms_dbfs,
            self.non_speech_peak_dbfs,
        ):
            if value is not None and (not math.isfinite(value) or value > 0):
                raise ValueError("PCM dBFS diagnostics must be finite and at most zero")
        return self


class BackgroundAudioScoutSummary(SchemaModel):
    schema_version: Literal["r2v.h3.background_audio_scout_summary.1"] = (
        BACKGROUND_AUDIO_SCOUT_SUMMARY_VERSION
    )
    inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_diarization_inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_audio_evidence_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_clip_count: Literal[75] = EXPECTED_PRODUCTION_TARGET_COUNT
    diagnostic_record_count: Literal[75] = EXPECTED_PRODUCTION_TARGET_COUNT
    model_calls: Literal[0] = 0
    dots3_calls: Literal[0] = 0
    learned_audio_classifier_calls: Literal[0] = 0
    automatic_background_selection_applied: Literal[False] = False
    human_selection_required: Literal[True] = True
    required_final_selection_count: Literal[20] = 20
    source_modified: Literal[False] = False


class BackgroundAudioSelectionReview(SchemaModel):
    target_clip_uid: str
    label: Literal["background-rich", "clean", "uncertain"]
    flags: list[
        Literal[
            "music",
            "crowd",
            "traffic",
            "nature",
            "machinery",
            "impacts",
            "other_sound_event",
        ]
    ] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_review(self) -> BackgroundAudioSelectionReview:
        if len(self.flags) != len(set(self.flags)):
            raise ValueError("background-audio review flags must be unique")
        return self


class BackgroundAudioPilotSelection(SchemaModel):
    schema_version: Literal["r2v.h3.background_audio_pilot_selection.1"] = (
        BACKGROUND_AUDIO_SELECTION_VERSION
    )
    source_scout_inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_diarization_inventory_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_audio_evidence_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    selection_method: Literal["manual_human_review_v1"] = "manual_human_review_v1"
    selection_count: Literal[20] = 20
    selected_clip_ids: list[str]
    reviews: list[BackgroundAudioSelectionReview]

    @model_validator(mode="after")
    def validate_selection(self) -> BackgroundAudioPilotSelection:
        if len(self.selected_clip_ids) != 20:
            raise ValueError("background-rich pilot selection must contain 20 clips")
        if len(self.selected_clip_ids) != len(set(self.selected_clip_ids)):
            raise ValueError("background-rich selected clip IDs must be unique")
        review_ids = [item.target_clip_uid for item in self.reviews]
        if len(review_ids) != len(set(review_ids)):
            raise ValueError("background-audio reviews must be unique")
        rich_ids = [
            item.target_clip_uid
            for item in self.reviews
            if item.label == "background-rich"
        ]
        if self.selected_clip_ids != rich_ids:
            raise ValueError(
                "selected clips must equal background-rich reviews in review order"
            )
        return self


def _inventory_fingerprint(
    inventory: BackgroundAudioScoutInventory | dict[str, object],
) -> str:
    values = (
        inventory.model_dump(mode="json", exclude={"inventory_fingerprint"})
        if isinstance(inventory, BackgroundAudioScoutInventory)
        else {
            key: value
            for key, value in inventory.items()
            if key != "inventory_fingerprint"
        }
    )
    return _sha256_text(_compact_json(values))


def union_sample_spans(
    spans: Sequence[tuple[int, int]], *, frame_count: int
) -> list[tuple[int, int]]:
    if frame_count <= 0:
        raise ValueError("canonical PCM frame count must be positive")
    normalized: list[tuple[int, int]] = []
    for start, end in spans:
        if start < 0 or end <= start or end > frame_count:
            raise ValueError("diarized sample span is outside canonical audio")
        normalized.append((start, end))
    merged: list[list[int]] = []
    for start, end in sorted(normalized):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def complement_sample_spans(
    speech_spans: Sequence[tuple[int, int]], *, frame_count: int
) -> list[tuple[int, int]]:
    merged = union_sample_spans(speech_spans, frame_count=frame_count)
    complement: list[tuple[int, int]] = []
    cursor = 0
    for start, end in merged:
        if start > cursor:
            complement.append((cursor, start))
        cursor = end
    if cursor < frame_count:
        complement.append((cursor, frame_count))
    return complement


def _dbfs(amplitude: float) -> float | None:
    return None if amplitude <= 0 else 20.0 * math.log10(amplitude)


def _pcm_metrics(samples: Sequence[int]) -> tuple[float | None, float | None]:
    if not samples:
        return None, None
    rms = (
        math.sqrt(
            math.fsum(float(value) * float(value) for value in samples) / len(samples)
        )
        / 32768.0
    )
    peak = max(abs(value) for value in samples) / 32768.0
    return _dbfs(rms), _dbfs(peak)


def _read_pcm16_mono(
    path: Path, *, expected_sample_rate: int, expected_frame_count: int
) -> array[int]:
    try:
        with wave.open(str(path), "rb") as source:
            if (
                source.getnchannels() != 1
                or source.getsampwidth() != 2
                or source.getframerate() != expected_sample_rate
                or source.getcomptype() != "NONE"
                or source.getnframes() != expected_frame_count
            ):
                raise ValueError(
                    "background-audio scout requires matching mono PCM16 WAV"
                )
            frames = source.readframes(source.getnframes())
    except (OSError, EOFError, wave.Error) as exc:
        raise ValueError("canonical full audio is unreadable") from exc
    samples = array("h")
    samples.frombytes(frames)
    if sys.byteorder != "little":
        samples.byteswap()
    return samples


def analyze_background_audio_job(
    job: BackgroundAudioScoutJob,
) -> BackgroundAudioScoutClip:
    samples = _read_pcm16_mono(
        Path(job.target_full_audio_path),
        expected_sample_rate=job.source_sample_rate_hz,
        expected_frame_count=job.source_frame_count,
    )
    speech_spans = [
        (item.start_sample, item.end_sample) for item in job.diarized_speech_spans
    ]
    non_speech_spans = complement_sample_spans(
        speech_spans,
        frame_count=job.source_frame_count,
    )
    non_speech_samples = array("h")
    for start, end in non_speech_spans:
        non_speech_samples.extend(samples[start:end])
    full_rms, _ = _pcm_metrics(samples)
    non_speech_rms, non_speech_peak = _pcm_metrics(non_speech_samples)
    speech_sample_count = sum(end - start for start, end in speech_spans)
    non_speech_sample_count = job.source_frame_count - speech_sample_count
    duration = job.source_frame_count / job.source_sample_rate_hz
    return BackgroundAudioScoutClip(
        target_clip_uid=job.target_clip_uid,
        clip_duration_seconds=duration,
        diarized_speaker_union_seconds=(
            speech_sample_count / job.source_sample_rate_hz
        ),
        non_speech_seconds=non_speech_sample_count / job.source_sample_rate_hz,
        non_speech_ratio=non_speech_sample_count / job.source_frame_count,
        full_audio_rms_dbfs=full_rms,
        non_speech_rms_dbfs=non_speech_rms,
        non_speech_peak_dbfs=non_speech_peak,
        target_full_audio_path=job.target_full_audio_path,
        target_full_audio_sha256=job.target_full_audio_sha256,
    )


def _validate_diarization_inventory(inventory: DiarizationInventory) -> None:
    expected = _diarization_inventory_fingerprint(
        source_pairs_sha256=inventory.source_pairs_sha256,
        source_asr_inventory_fingerprint=inventory.source_asr_inventory_fingerprint,
        mode=inventory.mode,
        targets=inventory.targets,
    )
    if inventory.inventory_fingerprint != expected:
        raise ValueError("source DiariZen inventory fingerprint is inconsistent")
    if (
        inventory.mode != "production"
        or inventory.selected_target_count != EXPECTED_PRODUCTION_TARGET_COUNT
        or inventory.source_target_count != EXPECTED_PRODUCTION_TARGET_COUNT
        or inventory.bounded_selection_applied
        or inventory.parent_quota_applied
    ):
        raise ValueError("background-audio scout requires complete production DiariZen")


def build_background_audio_scout_inventory(
    *, audio_run_root: Path
) -> BackgroundAudioScoutInventory:
    root = audio_run_root.expanduser().resolve(strict=True)
    diarization_root = (root / "production" / "diarization").resolve(strict=True)
    audio_root = (root / "production" / "audio").resolve(strict=True)
    inventory_path = (diarization_root / "inventory.json").resolve(strict=True)
    raw_segments_path = (diarization_root / "raw_segments.jsonl").resolve(strict=True)
    if (
        inventory_path.parent != diarization_root
        or raw_segments_path.parent != diarization_root
    ):
        raise ValueError("DiariZen scout source path escaped production root")
    inventory = DiarizationInventory.model_validate_json(
        inventory_path.read_text(encoding="utf-8")
    )
    _validate_diarization_inventory(inventory)
    raw_segments = [
        RawDiarizationSegment.model_validate(item)
        for item in _read_jsonl(raw_segments_path)
    ]
    target_ids = {item.target_clip_uid for item in inventory.targets}
    if any(item.target_clip_uid not in target_ids for item in raw_segments):
        raise ValueError("DiariZen raw segment references an unknown production target")
    segments_by_clip: dict[str, list[RawDiarizationSegment]] = defaultdict(list)
    for item in raw_segments:
        segments_by_clip[item.target_clip_uid].append(item)

    jobs: list[BackgroundAudioScoutJob] = []
    audio_evidence: list[dict[str, str]] = []
    for target in inventory.targets:
        sidecar_path = Path(target.target_audio_binding_path).resolve(strict=True)
        if sidecar_path != audio_root and audio_root not in sidecar_path.parents:
            raise ValueError("target audio sidecar escaped production Audio root")
        sidecar = AudioBindingSidecar.model_validate_json(
            sidecar_path.read_text(encoding="utf-8")
        )
        if sidecar.status != "ready" or sidecar.evidence is None:
            raise ValueError("background-audio scout requires ready Audio sidecars")
        audio = sidecar.evidence.audio
        if audio.status != "ready" or audio.full_audio_path is None:
            raise ValueError("background-audio scout requires canonical full audio")
        audio_path = Path(audio.full_audio_path).resolve(strict=True)
        expected_audio_path = Path(target.source_audio_path).resolve(strict=True)
        if (
            sidecar.clip_uid != target.target_clip_uid
            or audio_path != expected_audio_path
            or sidecar.source_video_path != target.target_video_path
        ):
            raise ValueError("Audio and DiariZen target provenance disagree")
        audio_hash = _sha256_file(audio_path)
        if audio_hash != target.source_audio_sha256:
            raise ValueError("canonical full-audio hash differs from DiariZen")
        clip_segments = segments_by_clip[target.target_clip_uid]
        for segment in clip_segments:
            if (
                segment.source_audio_sha256 != target.source_audio_sha256
                or Path(segment.source_audio_path).resolve(strict=True) != audio_path
                or segment.source_sample_rate_hz != target.source_sample_rate_hz
            ):
                raise ValueError("DiariZen segment audio provenance is inconsistent")
        union = union_sample_spans(
            [
                (segment.source_start_sample, segment.source_end_sample)
                for segment in clip_segments
            ],
            frame_count=target.source_frame_count,
        )
        sidecar_hash = _sha256_file(sidecar_path)
        jobs.append(
            BackgroundAudioScoutJob(
                target_clip_uid=target.target_clip_uid,
                target_full_audio_path=str(audio_path),
                target_full_audio_sha256=audio_hash,
                target_audio_binding_path=str(sidecar_path),
                target_audio_binding_sha256=sidecar_hash,
                source_sample_rate_hz=target.source_sample_rate_hz,
                source_frame_count=target.source_frame_count,
                diarized_speech_spans=[
                    SampleSpan(start_sample=start, end_sample=end)
                    for start, end in union
                ],
            )
        )
        audio_evidence.append(
            {
                "target_clip_uid": target.target_clip_uid,
                "target_full_audio_sha256": audio_hash,
                "target_audio_binding_sha256": sidecar_hash,
            }
        )

    audio_fingerprint = _sha256_text(_compact_json(audio_evidence))
    values: dict[str, object] = {
        "schema_version": BACKGROUND_AUDIO_SCOUT_INVENTORY_VERSION,
        "source_diarization_root": str(diarization_root),
        "source_diarization_inventory_path": str(inventory_path),
        "source_diarization_inventory_sha256": _sha256_file(inventory_path),
        "source_diarization_inventory_fingerprint": inventory.inventory_fingerprint,
        "source_diarization_raw_segments_path": str(raw_segments_path),
        "source_diarization_raw_segments_sha256": _sha256_file(raw_segments_path),
        "source_audio_root": str(audio_root),
        "source_audio_evidence_fingerprint": audio_fingerprint,
        "target_clip_count": len(jobs),
        "selection_applied": False,
        "parent_quota_applied": False,
        "model_calls": 0,
        "automatic_background_selection_applied": False,
        "jobs": [item.model_dump(mode="json") for item in jobs],
    }
    return BackgroundAudioScoutInventory(
        **values,
        inventory_fingerprint=_sha256_text(_compact_json(values)),
    )


def _verify_sources(inventory: BackgroundAudioScoutInventory) -> None:
    sources = (
        (
            inventory.source_diarization_inventory_path,
            inventory.source_diarization_inventory_sha256,
        ),
        (
            inventory.source_diarization_raw_segments_path,
            inventory.source_diarization_raw_segments_sha256,
        ),
    )
    for path_value, expected in sources:
        path = Path(path_value)
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError("frozen background-audio scout source changed")
    for job in inventory.jobs:
        for path_value, expected in (
            (job.target_full_audio_path, job.target_full_audio_sha256),
            (job.target_audio_binding_path, job.target_audio_binding_sha256),
        ):
            path = Path(path_value)
            if not path.is_file() or _sha256_file(path) != expected:
                raise ValueError("frozen background-audio evidence changed")


def _display_dbfs(value: float | None) -> str:
    return "null" if value is None else f"{value:.2f}"


def _review_html(
    *,
    inventory: BackgroundAudioScoutInventory,
    records: Sequence[BackgroundAudioScoutClip],
    media_names: dict[str, str],
) -> str:
    rows = []
    for index, record in enumerate(records):
        radios = " ".join(
            f"<label><input type='radio' name='label-{html.escape(record.target_clip_uid)}' value='{label}' onchange='saveReview()'>{label}</label>"
            for label in SCOUT_LABELS
        )
        flags = " ".join(
            f"<label><input type='checkbox' data-flag='{flag}' onchange='saveReview()'>{flag}</label>"
            for flag in SCOUT_FLAGS
        )
        rows.append(
            f"<tr class='case' data-index='{index}' data-clip='{html.escape(record.target_clip_uid)}' "
            f"data-rms='{record.non_speech_rms_dbfs if record.non_speech_rms_dbfs is not None else -999}' "
            f"data-ratio='{record.non_speech_ratio}' data-seconds='{record.non_speech_seconds}'>"
            f"<td>{html.escape(record.target_clip_uid)}<br><audio controls preload='none' src='media/{html.escape(media_names[record.target_clip_uid])}'></audio></td>"
            f"<td>{record.clip_duration_seconds:.3f}</td>"
            f"<td>{record.diarized_speaker_union_seconds:.3f}</td>"
            f"<td>{record.non_speech_seconds:.3f}</td>"
            f"<td>{record.non_speech_ratio:.4f}</td>"
            f"<td>{_display_dbfs(record.non_speech_rms_dbfs)}</td>"
            f"<td>{_display_dbfs(record.full_audio_rms_dbfs)}</td>"
            f"<td><div>{radios}</div><div>{flags}</div></td></tr>"
        )
    order = [item.target_clip_uid for item in records]
    return f"""<!doctype html><html><head><meta charset='utf-8'>
<title>H3 background audio scout</title><style>
body{{font-family:system-ui,sans-serif;margin:20px;color:#171717;background:#f4f5f6}}
header,main{{max-width:1500px;margin:0 auto 16px;background:white;padding:16px;border:1px solid #bbb}}
table{{width:100%;border-collapse:collapse}}th,td{{border:1px solid #bbb;padding:7px;vertical-align:top}}
th{{position:sticky;top:0;background:#eee}}audio{{width:260px}}label{{display:inline-block;margin:2px 8px 2px 0}}
button,select{{margin:4px;padding:7px 10px}}
</style></head><body><header><h1>H3 Background Audio Scout</h1>
<p>Diagnostics are navigation aids only. High non-speech energy is not an automatic background-sound decision.</p>
<p id='progress'>Reviewed 0 / {EXPECTED_PRODUCTION_TARGET_COUNT}; background-rich 0</p>
<button onclick="sortCases('rms')">Highest non-speech RMS</button>
<button onclick="sortCases('ratio')">Highest non-speech ratio</button>
<button onclick="sortCases('seconds')">Longest non-speech duration</button>
<select id='filter' onchange='filterCases()'><option value='all'>all</option><option value='background-rich'>background-rich</option><option value='clean'>clean</option><option value='uncertain'>uncertain</option><option value='unlabeled'>unlabeled</option></select>
<button onclick='exportSelection()'>Export 20-clip selection JSON</button>
<button onclick='clearReviews()'>Clear review labels</button></header><main>
<table><thead><tr><th>clip/audio</th><th>duration</th><th>speech union</th><th>non-speech sec</th><th>non-speech ratio</th><th>non-speech RMS dBFS</th><th>full RMS dBFS</th><th>human review</th></tr></thead><tbody id='cases'>{"".join(rows)}</tbody></table>
</main><script>
const clipOrder={json.dumps(order)};
const sourceScoutFingerprint={json.dumps(inventory.inventory_fingerprint)};
const sourceDiarizationFingerprint={json.dumps(inventory.source_diarization_inventory_fingerprint)};
const sourceAudioFingerprint={json.dumps(inventory.source_audio_evidence_fingerprint)};
const labels={json.dumps(list(SCOUT_LABELS))};const flags={json.dumps(list(SCOUT_FLAGS))};
const keyPrefix='h3-background-audio-scout-';
function stateFor(clip){{try{{return JSON.parse(localStorage.getItem(keyPrefix+clip)||'null')}}catch(e){{return null}}}}
function saveReview(){{document.querySelectorAll('.case').forEach(row=>{{const selected=row.querySelector("input[type='radio']:checked");const selectedFlags=[...row.querySelectorAll("input[type='checkbox']:checked")].map(i=>i.dataset.flag);if(selected)localStorage.setItem(keyPrefix+row.dataset.clip,JSON.stringify({{label:selected.value,flags:selectedFlags}}));else localStorage.removeItem(keyPrefix+row.dataset.clip);}});updateCounts();filterCases();}}
function restore(){{document.querySelectorAll('.case').forEach(row=>{{const state=stateFor(row.dataset.clip);if(!state)return;row.querySelectorAll("input[type='radio']").forEach(i=>i.checked=i.value===state.label);row.querySelectorAll("input[type='checkbox']").forEach(i=>i.checked=(state.flags||[]).includes(i.dataset.flag));}});updateCounts();}}
function reviewRows(){{const rows=[];clipOrder.forEach(clip=>{{const state=stateFor(clip);if(state&&labels.includes(state.label))rows.push({{target_clip_uid:clip,label:state.label,flags:(state.flags||[]).filter(flag=>flags.includes(flag))}});}});return rows;}}
function updateCounts(){{const reviews=reviewRows();const rich=reviews.filter(row=>row.label==='background-rich').length;document.getElementById('progress').textContent=`Reviewed ${{reviews.length}} / {EXPECTED_PRODUCTION_TARGET_COUNT}; background-rich ${{rich}} / 20`;}}
function sortCases(metric){{const tbody=document.getElementById('cases');[...tbody.children].sort((a,b)=>Number(b.dataset[metric])-Number(a.dataset[metric])||Number(a.dataset.index)-Number(b.dataset.index)).forEach(row=>tbody.appendChild(row));}}
function filterCases(){{const value=document.getElementById('filter').value;document.querySelectorAll('.case').forEach(row=>{{const state=stateFor(row.dataset.clip);const label=state?state.label:'unlabeled';row.hidden=value!=='all'&&value!==label;}});}}
function exportSelection(){{const reviews=reviewRows();const selected=reviews.filter(row=>row.label==='background-rich').map(row=>row.target_clip_uid);if(selected.length!==20){{alert(`Exactly 20 background-rich clips are required; currently ${{selected.length}}.`);return;}}const payload={{schema_version:'{BACKGROUND_AUDIO_SELECTION_VERSION}',source_scout_inventory_fingerprint:sourceScoutFingerprint,source_diarization_inventory_fingerprint:sourceDiarizationFingerprint,source_audio_evidence_fingerprint:sourceAudioFingerprint,selection_method:'manual_human_review_v1',selection_count:20,selected_clip_ids:selected,reviews:reviews}};const blob=new Blob([JSON.stringify(payload,null,2)+'\n'],{{type:'application/json'}});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='background_audio_pilot_selection.json';a.click();URL.revokeObjectURL(a.href);}}
function clearReviews(){{if(!confirm('Clear background-audio scout labels?'))return;clipOrder.forEach(clip=>localStorage.removeItem(keyPrefix+clip));document.querySelectorAll('.case input').forEach(input=>input.checked=false);updateCounts();filterCases();}}
restore();
</script></body></html>"""


def _publish_directory(temporary: Path, destination: Path, *, overwrite: bool) -> None:
    if not destination.exists():
        temporary.replace(destination)
        return
    if not overwrite:
        raise FileExistsError(f"background-audio scout output exists: {destination}")
    backup = destination.with_name(f".{destination.name}.old-{uuid.uuid4().hex}")
    destination.replace(backup)
    try:
        temporary.replace(destination)
    except Exception:
        backup.replace(destination)
        raise
    shutil.rmtree(backup)


def background_audio_scout_output_root(audio_run_root: Path) -> Path:
    return (
        audio_run_root.expanduser().resolve(strict=False)
        / BACKGROUND_AUDIO_SCOUT_OUTPUT_DIRECTORY
    )


def run_background_audio_scout(
    *,
    inventory: BackgroundAudioScoutInventory,
    output_root: Path,
    overwrite: bool = False,
) -> BackgroundAudioScoutSummary:
    if inventory.inventory_fingerprint != _inventory_fingerprint(inventory):
        raise ValueError("background-audio scout inventory fingerprint is inconsistent")
    source_audio_root = Path(inventory.source_audio_root).resolve(strict=True)
    expected_output = (
        source_audio_root.parents[1] / BACKGROUND_AUDIO_SCOUT_OUTPUT_DIRECTORY
    )
    destination = output_root.expanduser().resolve(strict=False)
    if destination != expected_output:
        raise ValueError("background-audio scout output root is fixed")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"background-audio scout output exists: {destination}")
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    try:
        temporary.mkdir()
        (temporary / "media").mkdir()
        records = [analyze_background_audio_job(job) for job in inventory.jobs]
        media_names: dict[str, str] = {}
        for job in inventory.jobs:
            source = Path(job.target_full_audio_path)
            name = f"{job.target_clip_uid}{source.suffix.lower() or '.wav'}"
            (temporary / "media" / name).symlink_to(source)
            media_names[job.target_clip_uid] = name
        _verify_sources(inventory)
        summary = BackgroundAudioScoutSummary(
            inventory_fingerprint=inventory.inventory_fingerprint,
            source_diarization_inventory_fingerprint=(
                inventory.source_diarization_inventory_fingerprint
            ),
            source_audio_evidence_fingerprint=(
                inventory.source_audio_evidence_fingerprint
            ),
        )
        _write_json(temporary / "inventory.json", inventory.model_dump(mode="json"))
        (temporary / "diagnostics.jsonl").write_text(
            "".join(
                _compact_json(item.model_dump(mode="json")) + "\n" for item in records
            ),
            encoding="utf-8",
        )
        _write_json(temporary / "summary.json", summary.model_dump(mode="json"))
        (temporary / "report.html").write_text(
            _review_html(
                inventory=inventory,
                records=records,
                media_names=media_names,
            ),
            encoding="utf-8",
        )
        _publish_directory(temporary, destination, overwrite=overwrite)
        return summary
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
