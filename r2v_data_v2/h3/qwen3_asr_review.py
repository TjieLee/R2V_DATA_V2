from __future__ import annotations

import hashlib
import html
import json
import shutil
import subprocess
import uuid
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Protocol

from pydantic import Field, model_validator

from r2v_data_v2.h3.jea_audio_production import JEAInPair
from r2v_data_v2.h3.qwen3_asr import Qwen3ASRSegment
from r2v_data_v2.h3.schemas import SchemaModel

QWEN3_ASR_REVIEW_MANIFEST_VERSION = "r2v.h3.qwen3_asr_review_manifest.1"
QWEN3_ASR_HUMAN_QA_VERSION = "r2v.h3.qwen3_asr_human_qa.1"


class Qwen3ASRReviewClip(SchemaModel):
    clip_uid: str
    clip_display_path: str
    target_video_path: str
    review_video_path: str
    segment_count: int = Field(gt=0)
    segment_ids: list[str]

    @model_validator(mode="after")
    def validate_segments(self) -> Qwen3ASRReviewClip:
        if self.segment_count != len(self.segment_ids):
            raise ValueError("Qwen3 ASR review clip segment count is inconsistent")
        return self


class Qwen3ASRReviewManifest(SchemaModel):
    schema_version: Literal[
        "r2v.h3.qwen3_asr_review_manifest.1"
    ] = QWEN3_ASR_REVIEW_MANIFEST_VERSION
    source_qwen_asr_root: str
    source_pairs_root: str
    clip_count: int = Field(ge=0)
    segment_count: int = Field(ge=0)
    transcribed_count: int = Field(ge=0)
    empty_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    clips: list[Qwen3ASRReviewClip]

    @model_validator(mode="after")
    def validate_counts(self) -> Qwen3ASRReviewManifest:
        if self.clip_count != len(self.clips):
            raise ValueError("Qwen3 ASR review clip count is inconsistent")
        if self.segment_count != sum(item.segment_count for item in self.clips):
            raise ValueError("Qwen3 ASR review segment count is inconsistent")
        if self.segment_count != (
            self.transcribed_count + self.empty_count + self.failed_count
        ):
            raise ValueError("Qwen3 ASR review status counts do not reconcile")
        return self


class Qwen3ASRReviewMediaBackend(Protocol):
    def transcode_video(
        self,
        *,
        source_video_path: Path,
        destination_path: Path,
    ) -> None: ...


class FFmpegQwen3ASRReviewMediaBackend:
    def __init__(
        self,
        *,
        ffmpeg: str = "ffmpeg",
        timeout_seconds: float = 900.0,
    ) -> None:
        if not ffmpeg.strip() or timeout_seconds <= 0:
            raise ValueError("review ffmpeg and positive timeout are required")
        self.ffmpeg = ffmpeg
        self.timeout_seconds = timeout_seconds

    def transcode_video(
        self,
        *,
        source_video_path: Path,
        destination_path: Path,
    ) -> None:
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            completed = subprocess.run(
                [
                    self.ffmpeg,
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(source_video_path),
                    "-map",
                    "0:v:0",
                    "-map",
                    "0:a:0",
                    "-vf",
                    "scale=trunc(min(960\\,iw)/2)*2:-2",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "23",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-ac",
                    "2",
                    "-b:a",
                    "128k",
                    "-movflags",
                    "+faststart",
                    str(destination_path),
                ],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                check=False,
                text=True,
                timeout=self.timeout_seconds,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(
                f"Qwen3 ASR review proxy failed: {type(exc).__name__}: {exc}"
            ) from exc
        if completed.returncode != 0 or not destination_path.is_file():
            raise RuntimeError(
                "Qwen3 ASR review proxy failed with code "
                f"{completed.returncode}: {completed.stderr.strip()}"
            )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL row {line_number}: {path}") from exc
            if not isinstance(value, dict):
                raise TypeError(f"JSONL row {line_number} is not an object: {path}")
            rows.append(value)
    return rows


def _write_json(path: Path, value: SchemaModel) -> None:
    path.write_text(value.model_dump_json(indent=2) + "\n", encoding="utf-8")


def _safe_media_name(clip_uid: str) -> str:
    digest = hashlib.sha256(clip_uid.encode("utf-8")).hexdigest()
    return f"{digest}.mp4"


def _load_sources(
    audio_production_root: Path,
) -> tuple[Path, Path, list[Qwen3ASRSegment], dict[str, JEAInPair]]:
    root = audio_production_root.expanduser().resolve(strict=True)
    asr_root = (root / "asr").resolve(strict=True)
    pairs_root = (root / "pairs").resolve(strict=True)
    rows = [
        Qwen3ASRSegment.model_validate(value)
        for value in _read_jsonl(asr_root / "segments.jsonl")
    ]
    identities: set[tuple[str, str]] = set()
    for row in rows:
        identity = (row.clip_uid, row.segment_id)
        if identity in identities:
            raise ValueError("duplicate Qwen3 ASR segment identity")
        identities.add(identity)
    rows.sort(
        key=lambda item: (
            item.clip_display_path,
            item.source_start_sample,
            item.segment_id,
        )
    )
    pairs: dict[str, JEAInPair] = {}
    for value in _read_jsonl(pairs_root / "in_pairs.jsonl"):
        pair = JEAInPair.model_validate(value)
        if pair.target_clip_uid in pairs:
            raise ValueError("duplicate JEA in-pair target clip")
        pairs[pair.target_clip_uid] = pair
    return asr_root, pairs_root, rows, pairs


def _validate_target_video(row: Qwen3ASRSegment, pair: JEAInPair) -> Path:
    if row.clip_display_path != pair.target_clip_display_path:
        raise ValueError("Qwen3 ASR clip display path differs from in-pair")
    source = Path(pair.target_video_path).expanduser()
    if not source.is_absolute():
        raise ValueError("JEA target review video path must be absolute")
    try:
        resolved = source.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Qwen3 ASR review target video is missing: {source}"
        ) from exc
    if not resolved.is_file():
        raise ValueError("Qwen3 ASR review target video is not a file")
    return resolved


def _format_time(value: float) -> str:
    minutes, seconds = divmod(value, 60)
    return f"{int(minutes):02d}:{seconds:05.2f}"


def _review_html(
    *,
    manifest: Qwen3ASRReviewManifest,
    rows: Sequence[Qwen3ASRSegment],
) -> str:
    rows_by_clip: dict[str, list[tuple[int, Qwen3ASRSegment]]] = defaultdict(list)
    row_metadata: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        rows_by_clip[row.clip_uid].append((index, row))
        row_metadata.append(
            {
                "clip_uid": row.clip_uid,
                "clip_display_path": row.clip_display_path,
                "segment_id": row.segment_id,
                "speaker_cluster_id": row.speaker_cluster_id,
                "entity_id": row.entity_id,
                "start_time": row.start_time,
                "end_time": row.end_time,
                "status": row.status,
                "text": row.text,
                "language": row.language,
                "failure_reason": row.failure_reason,
                "storage_key": (
                    "h3-qwen3-asr-review-v1:"
                    f"{row.clip_uid}:{row.segment_id}"
                ),
            }
        )
    clips = []
    for clip_index, clip in enumerate(manifest.clips):
        segment_cards = []
        for row_index, row in rows_by_clip[clip.clip_uid]:
            if row.status == "empty":
                output = "<strong class='empty-output'>[EMPTY OUTPUT]</strong>"
            elif row.status == "failed":
                output = (
                    "<strong class='failed-output'>[FAILED]</strong> "
                    + html.escape(row.failure_reason or "unknown failure")
                )
            else:
                output = "<blockquote>" + html.escape(row.text or "") + "</blockquote>"
            entity = html.escape(row.entity_id or "UNBOUND")
            language = html.escape(row.language or "[none]")
            segment_cards.append(
                f"<article class='segment-row' id='segment-{row_index}' "
                f"data-row-index='{row_index}' data-start='{row.start_time}' "
                f"data-end='{row.end_time}' onclick='selectSegment({row_index})'>"
                f"<h3>{html.escape(row.segment_id)}</h3>"
                f"<p><strong>{_format_time(row.start_time)} - {_format_time(row.end_time)}</strong><br>"
                f"speaker_cluster_id: {html.escape(row.speaker_cluster_id)}<br>"
                f"entity_id: {entity}<br>language: {language}<br>"
                f"status: {html.escape(row.status)}</p>"
                f"<p><strong>Qwen:</strong></p>{output}"
                f"<button type='button' onclick='event.stopPropagation();playSegment({row_index})'>&#9654; Play segment</button>"
                "<fieldset class='qa'><legend>Human QA</legend>"
                + " ".join(
                    f"<label><input type='radio' name='qa-{row_index}' value='{label}' "
                    f"onchange='saveReview({row_index})'>{label}</label>"
                    for label in ("CORRECT", "WRONG", "UNCERTAIN")
                )
                + f"<label class='note'>Note <input type='text' id='note-{row_index}' "
                f"oninput='saveReview({row_index})' maxlength='500'></label></fieldset>"
                "</article>"
            )
        clips.append(
            f"<section class='clip-card' id='clip-{clip_index}' data-clip-index='{clip_index}'>"
            f"<h2>{html.escape(clip.clip_display_path)}</h2>"
            f"<video id='video-{clip_index}' controls preload='metadata' "
            f"src='{html.escape(clip.review_video_path)}'></video>"
            + "".join(segment_cards)
            + "</section>"
        )
    review_data = {
        "schema_version": QWEN3_ASR_HUMAN_QA_VERSION,
        "source_qwen_asr_root": manifest.source_qwen_asr_root,
        "segment_count": manifest.segment_count,
        "clip_count": manifest.clip_count,
        "status_counts": {
            "transcribed": manifest.transcribed_count,
            "empty": manifest.empty_count,
            "failed": manifest.failed_count,
        },
        "clips": [
            {
                "clip_uid": item.clip_uid,
                "clip_display_path": item.clip_display_path,
            }
            for item in manifest.clips
        ],
        "rows": row_metadata,
    }
    review_json = json.dumps(
        review_data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).replace("<", "\\u003c")
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Qwen3 ASR human review</title>
<style>
body{{font-family:system-ui,sans-serif;margin:0;background:#f2f4f5;color:#171717}}
header{{position:sticky;top:0;z-index:4;background:#fff;border-bottom:1px solid #bbb;padding:14px 20px}}
.toolbar,.counts{{display:flex;flex-wrap:wrap;gap:10px;align-items:center}}button{{padding:7px 11px}}
main{{max-width:1120px;margin:20px auto;padding:0 16px}}.clip-card{{display:none}}.clip-card.active{{display:block}}
.clip-card>h2{{overflow-wrap:anywhere}}video{{width:100%;max-height:560px;background:#111}}
.segment-row{{background:#fff;border:1px solid #bbb;margin:14px 0;padding:14px;cursor:pointer}}
.segment-row.selected{{border:3px solid #1769aa;padding:12px}}.segment-row.active-time{{background:#eef7ff}}
blockquote{{margin:8px 0;padding:10px;border-left:4px solid #999;background:#f7f7f7}}
.qa{{margin-top:12px;display:flex;flex-wrap:wrap;gap:12px;align-items:center}}.note input{{min-width:260px}}
.empty-output{{color:#8a5400}}.failed-output{{color:#a40000}}select{{max-width:460px}}
</style></head><body>
<header><div class="toolbar"><strong>Qwen3 ASR review</strong>
<button onclick="previousClip()">Previous Clip</button><button onclick="nextClip()">Next Clip</button>
<select id="clip-select" onchange="showClip(Number(this.value))">{''.join(f'<option value="{index}">{html.escape(item.clip_display_path)}</option>' for index, item in enumerate(manifest.clips))}</select>
<button onclick="exportQAJSON()">Export QA JSON</button></div>
<div class="counts"><span>Total: <b id="total-count">0</b></span><span>Reviewed: <b id="reviewed-count">0</b></span>
<span>Correct: <b id="correct-count">0</b></span><span>Wrong: <b id="wrong-count">0</b></span>
<span>Uncertain: <b id="uncertain-count">0</b></span><span>Remaining: <b id="remaining-count">0</b></span>
<span>Transcribed: <b>{manifest.transcribed_count}</b></span><span>Empty: <b>{manifest.empty_count}</b></span>
<span>Failed: <b>{manifest.failed_count}</b></span></div></header>
<main>{''.join(clips)}</main>
<script>
const reviewData = {review_json};
const allowedLabels = new Set(['CORRECT','WRONG','UNCERTAIN']);
let currentClipIndex = 0; let currentRowIndex = 0; let stopAt = null;
function stateFor(row){{try{{return JSON.parse(localStorage.getItem(row.storage_key)||'null')||{{}};}}catch(error){{return {{}};}}}}
function saveReview(index){{const row=reviewData.rows[index];const selected=document.querySelector(`input[name="qa-${{index}}"]:checked`);const note=document.getElementById(`note-${{index}}`).value;const state={{review_label:selected?selected.value:null,review_note:note}};if(state.review_label||state.review_note)localStorage.setItem(row.storage_key,JSON.stringify(state));else localStorage.removeItem(row.storage_key);updateCounts();}}
function restoreReviews(){{reviewData.rows.forEach((row,index)=>{{const state=stateFor(row);if(allowedLabels.has(state.review_label)){{const input=document.querySelector(`input[name="qa-${{index}}"][value="${{state.review_label}}"]`);if(input)input.checked=true;}}document.getElementById(`note-${{index}}`).value=state.review_note||'';}});updateCounts();}}
function counts(){{const result={{CORRECT:0,WRONG:0,UNCERTAIN:0,UNLABELED:0}};reviewData.rows.forEach(row=>{{const label=stateFor(row).review_label;if(allowedLabels.has(label))result[label]++;else result.UNLABELED++;}});return result;}}
function updateCounts(){{const value=counts();document.getElementById('total-count').textContent=reviewData.segment_count;document.getElementById('reviewed-count').textContent=reviewData.segment_count-value.UNLABELED;document.getElementById('correct-count').textContent=value.CORRECT;document.getElementById('wrong-count').textContent=value.WRONG;document.getElementById('uncertain-count').textContent=value.UNCERTAIN;document.getElementById('remaining-count').textContent=value.UNLABELED;}}
function showClip(index){{if(index<0||index>=reviewData.clips.length)return;document.querySelectorAll('.clip-card').forEach(card=>card.classList.remove('active'));document.getElementById(`clip-${{index}}`).classList.add('active');currentClipIndex=index;document.getElementById('clip-select').value=String(index);const first=document.querySelector(`#clip-${{index}} .segment-row`);if(first)selectSegment(Number(first.dataset.rowIndex));}}
function previousClip(){{showClip(Math.max(0,currentClipIndex-1));}}function nextClip(){{showClip(Math.min(reviewData.clips.length-1,currentClipIndex+1));}}
function selectSegment(index){{currentRowIndex=index;document.querySelectorAll('.segment-row').forEach(row=>row.classList.remove('selected'));document.getElementById(`segment-${{index}}`).classList.add('selected');}}
function playSegment(index){{selectSegment(index);const row=reviewData.rows[index];const clipIndex=reviewData.clips.findIndex(clip=>clip.clip_uid===row.clip_uid);showClip(clipIndex);selectSegment(index);const video=document.getElementById(`video-${{clipIndex}}`);video.currentTime=row.start_time;stopAt=row.end_time;video.play();}}
function syncActiveTime(video,clipIndex){{const time=video.currentTime;document.querySelectorAll(`#clip-${{clipIndex}} .segment-row`).forEach(element=>{{const row=reviewData.rows[Number(element.dataset.rowIndex)];element.classList.toggle('active-time',row.start_time<=time&&time<row.end_time);}});if(stopAt!==null&&time>=stopAt){{video.pause();stopAt=null;}}}}
function exportQAJSON(){{reviewData.rows.forEach((row,index)=>saveReview(index));const qaCounts=counts();const rows=reviewData.rows.map(row=>{{const state=stateFor(row);return {{clip_uid:row.clip_uid,clip_display_path:row.clip_display_path,segment_id:row.segment_id,speaker_cluster_id:row.speaker_cluster_id,entity_id:row.entity_id,start_time:row.start_time,end_time:row.end_time,status:row.status,text:row.text,language:row.language,failure_reason:row.failure_reason,review_label:allowedLabels.has(state.review_label)?state.review_label:null,review_note:state.review_note||''}};}});const payload={{schema_version:reviewData.schema_version,source_qwen_asr_root:reviewData.source_qwen_asr_root,segment_count:reviewData.segment_count,summary:{{correct:qaCounts.CORRECT,wrong:qaCounts.WRONG,uncertain:qaCounts.UNCERTAIN,unlabeled:qaCounts.UNLABELED}},rows}};const blob=new Blob([JSON.stringify(payload,null,2)+'\\n'],{{type:'application/json'}});const url=URL.createObjectURL(blob);const link=document.createElement('a');link.href=url;link.download='qwen3_asr_human_qa.json';link.click();URL.revokeObjectURL(url);}}
document.querySelectorAll('video').forEach((video,index)=>video.addEventListener('timeupdate',()=>syncActiveTime(video,index)));
document.addEventListener('keydown',event=>{{if(['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName))return;if(event.key==='1'||event.key==='2'||event.key==='3'){{const label={{'1':'CORRECT','2':'WRONG','3':'UNCERTAIN'}}[event.key];const input=document.querySelector(`input[name="qa-${{currentRowIndex}}"][value="${{label}}"]`);if(input){{input.checked=true;saveReview(currentRowIndex);}}}}else if(event.key==='j'||event.key==='J'||event.key==='ArrowDown'){{event.preventDefault();selectSegment(Math.min(reviewData.rows.length-1,currentRowIndex+1));document.getElementById(`segment-${{currentRowIndex}}`).scrollIntoView({{block:'center'}});}}else if(event.key==='k'||event.key==='K'||event.key==='ArrowUp'){{event.preventDefault();selectSegment(Math.max(0,currentRowIndex-1));document.getElementById(`segment-${{currentRowIndex}}`).scrollIntoView({{block:'center'}});}}else if(event.code==='Space'){{event.preventDefault();playSegment(currentRowIndex);}}}});
restoreReviews();if(reviewData.clips.length)showClip(0);
</script></body></html>"""


def _publish_directory(temporary: Path, destination: Path, *, overwrite: bool) -> None:
    if not destination.exists():
        temporary.replace(destination)
        return
    if not overwrite:
        raise FileExistsError(f"Qwen3 ASR review output already exists: {destination}")
    backup = destination.with_name(f".{destination.name}.old-{uuid.uuid4().hex}")
    destination.replace(backup)
    try:
        temporary.replace(destination)
    except Exception:
        backup.replace(destination)
        raise
    shutil.rmtree(backup)


def generate_qwen3_asr_review(
    *,
    audio_production_root: Path,
    output_root: Path | None = None,
    media_backend: Qwen3ASRReviewMediaBackend,
    overwrite: bool = False,
) -> Qwen3ASRReviewManifest:
    root = audio_production_root.expanduser().resolve(strict=True)
    asr_root, pairs_root, rows, pairs = _load_sources(root)
    destination = (
        output_root.expanduser().resolve(strict=False)
        if output_root is not None
        else root / "asr_review"
    )
    for protected_name in ("asr", "pairs", "diarization", "h3"):
        protected = (root / protected_name).resolve(strict=False)
        if destination == protected or protected in destination.parents:
            raise ValueError("Qwen3 ASR review output cannot replace production stages")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Qwen3 ASR review output already exists: {destination}")
    rows_by_clip: dict[str, list[Qwen3ASRSegment]] = defaultdict(list)
    for row in rows:
        if row.clip_uid not in pairs:
            raise ValueError("Qwen3 ASR clip has no canonical JEA in-pair")
        rows_by_clip[row.clip_uid].append(row)
    clip_order = sorted(
        rows_by_clip,
        key=lambda clip_uid: (
            rows_by_clip[clip_uid][0].clip_display_path,
            clip_uid,
        ),
    )
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    try:
        temporary.mkdir()
        media_root = temporary / "media"
        media_root.mkdir()
        clips: list[Qwen3ASRReviewClip] = []
        for clip_uid in clip_order:
            pair = pairs[clip_uid]
            clip_rows = rows_by_clip[clip_uid]
            source_video = _validate_target_video(clip_rows[0], pair)
            if any(
                item.clip_display_path != clip_rows[0].clip_display_path
                for item in clip_rows
            ):
                raise ValueError("Qwen3 ASR clip rows have inconsistent display paths")
            media_name = _safe_media_name(clip_uid)
            media_backend.transcode_video(
                source_video_path=source_video,
                destination_path=media_root / media_name,
            )
            clips.append(
                Qwen3ASRReviewClip(
                    clip_uid=clip_uid,
                    clip_display_path=clip_rows[0].clip_display_path,
                    target_video_path=pair.target_video_path,
                    review_video_path=f"media/{media_name}",
                    segment_count=len(clip_rows),
                    segment_ids=[item.segment_id for item in clip_rows],
                )
            )
        status_counts = Counter(item.status for item in rows)
        manifest = Qwen3ASRReviewManifest(
            source_qwen_asr_root=str(asr_root),
            source_pairs_root=str(pairs_root),
            clip_count=len(clips),
            segment_count=len(rows),
            transcribed_count=status_counts["transcribed"],
            empty_count=status_counts["empty"],
            failed_count=status_counts["failed"],
            clips=clips,
        )
        _write_json(temporary / "manifest.json", manifest)
        (temporary / "review.html").write_text(
            _review_html(manifest=manifest, rows=rows),
            encoding="utf-8",
        )
        _publish_directory(temporary, destination, overwrite=overwrite)
        return manifest
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


__all__ = [
    "FFmpegQwen3ASRReviewMediaBackend",
    "Qwen3ASRReviewManifest",
    "Qwen3ASRReviewMediaBackend",
    "generate_qwen3_asr_review",
]
