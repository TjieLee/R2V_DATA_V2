"""Read-only post-product frame conditioning; no inference or Audio selection."""
from __future__ import annotations

import re
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from r2v_data_v2.h3.audio_reuse_finalizer import PREPARED_STAGE, PRODUCTS_STAGE
from r2v_data_v2.h3.audio_reuse_materializer import (
    AUDIO_REUSE_MATERIALIZER_VERSION,
    AudioReuseProduct,
    AudioReuseProductSummary,
    ReuseAudioReference,
    _hash,
)
from r2v_data_v2.h3.audio_reuse_prepared import (
    AudioReusePreparedSource,
    AudioReusePreparedSummary,
    validate_prepared_inputs,
)
from r2v_data_v2.h3.jea_final_renderer import FinalH3SampleV2, FinalQwen3SpeechSegment
from r2v_data_v2.h3.mimo25_av_reconcile import MimoClipJob, MimoInventory
from r2v_data_v2.h3.mimo25_h3_materializer import (
    _materialize_sample,
    _prepare_materialization_context,
    validate_product_dialogue_sections,
)
from r2v_data_v2.h3.qwen38_h3_recaption import (
    ConditioningVariant,
    RecaptionSubjectContract,
)
from r2v_data_v2.h3.sam_audio_stem_shadow import (
    SAMAudioStemRecord,
    sha256_file,
    stem_shadow_root,
)
from r2v_data_v2.h3.schemas import SchemaModel

VisualReferenceMode = Literal["reference", "first_frame", "last_frame", "first_last_frame"]
VISUAL_TASKS = {
    "reference": "reference generation", "first_frame": "first frame completion",
    "last_frame": "last frame completion", "first_last_frame": "first-and-last frame completion",
}
DERIVED_MODES = ("first_frame", "last_frame", "first_last_frame")
FRAME_STAGE = "h3_frame_conditioned_products_v1"
PROJECTION_VERSION = "h3_frame_conditioning_projection_v1"
EXTRACTION_VERSION = "exact_decoded_first_last_png_v1"
SECTIONS = ("subject_definitions", "summary", "retention_analysis", "detailed_description",
            "overall_soundscape", "non_diegetic_music")
_PICTURE = re.compile(r"<Picture [1-9]\d*>")
_DIALOGUE = re.compile(r"<d>[\s\S]*?</d>")


def split_sections(prompt: str) -> dict[str, str]:
    headers = list(re.finditer(r"(?:\A|\n\n)(" + "|".join(SECTIONS) + r"):\n", prompt))
    if not prompt.startswith("subject_definitions:\n") or [m[1] for m in headers] != list(SECTIONS):
        raise ValueError("frame projection requires exactly six ordered H3 sections")
    return {m[1]: prompt[m.end():headers[i + 1].start() if i + 1 < len(headers) else len(prompt)]
            for i, m in enumerate(headers)}


def project_prompt(prompt: str, mode: VisualReferenceMode, duration: float) -> str:
    """Apply only the explicitly defined Picture/task/frame text projection."""
    if mode not in DERIVED_MODES:
        raise ValueError("only derived frame modes are published")
    sections = split_sections(prompt)
    expected_dialogue = _DIALOGUE.findall(sections["detailed_description"])
    validate_product_dialogue_sections(prompt, expected_dialogue)
    if re.findall(r"\[Shot (\d+)\]", sections["detailed_description"]) != ["1"]:
        raise ValueError("frame projection v1 requires a single [Shot 1]")
    audio_rows = [line for line in sections["retention_analysis"].splitlines() if line.startswith("<Audio ")]
    for name, body in sections.items():
        if name in {"subject_definitions", "retention_analysis"}:
            body = "".join(line for line in body.splitlines(keepends=True)
                           if not re.match(r"^<Picture [1-9]\d*>", line))
        sections[name] = _PICTURE.sub("<Picture 1>", body)
    # Transcript and Audio-owned rows are immutable, even in unusual inputs that
    # quote Picture tokens. Never silently change those to satisfy the projection.
    if (audio_rows != [line for line in sections["retention_analysis"].splitlines() if line.startswith("<Audio ")]
            or _DIALOGUE.findall(sections["detailed_description"]) != expected_dialogue):
        raise ValueError("Picture normalization would alter protected dialogue/Audio text")
    prefix = re.match(r"^\[([^\]]+)\]", sections["summary"])
    if prefix is None:
        raise ValueError("source summary task prefix missing")
    tasks = prefix[1].split(" + ")
    if tasks[0] != "reference generation" or any(t not in {"audio reference", "audio reuse"} for t in tasks[1:]):
        raise ValueError("source summary must start with exact reference generation task")
    tasks[0] = VISUAL_TASKS[mode]
    if mode == "first_last_frame":
        alignment = ("For the target video, <Picture 1> (from [Shot 1]) is fully referenced at 0.00 seconds, "
                     f"and <Picture 2> (from [Shot 1]) is fully referenced at {duration:.2f} seconds.")
        roles = [(1, "first", "opening"), (2, "last", "ending")]
    else:
        time = 0.0 if mode == "first_frame" else duration
        alignment = (f"For the target video, at {time:.2f} seconds into the target video, "
                     "<Picture 1> (from [Shot 1]) is fully referenced.")
        roles = [(1, "first", "opening")] if mode == "first_frame" else [(1, "last", "ending")]
    remainder = sections["summary"][prefix.end():]
    sections["summary"] = "[" + " + ".join(tasks) + "] " + alignment + (remainder if remainder.startswith(" ") else " " + remainder)
    definitions = [f"<Picture {i}> is the {role} frame of [Shot 1], serving as the exact {edge}-frame reference for the target video."
                   for i, role, edge in roles]
    retention = [f"<Picture {i}> ([Shot 1] {role} frame): fully_preserved - <Picture {i}> is fully preserved as the exact {edge} frame of the target video."
                 for i, role, edge in roles]
    sections["subject_definitions"] = "\n".join(definitions) + "\n" + sections["subject_definitions"]
    sections["retention_analysis"] = "\n".join(retention) + "\n" + sections["retention_analysis"]
    if mode != "first_frame":
        label = "<Picture 2>" if mode == "first_last_frame" else "<Picture 1>"
        ending = f"The target video ends on the exact visual state established by {label}."
        if not sections["detailed_description"].rstrip().endswith(ending):
            sections["detailed_description"] += " " + ending
    rendered = "\n\n".join(f"{name}:\n{sections[name]}" for name in SECTIONS)
    validate_product_dialogue_sections(rendered, expected_dialogue)
    return rendered


class FrameConditioningPicture(SchemaModel):
    picture_index: int = Field(ge=1, le=2)
    picture_label: str
    frame_role: Literal["first_frame", "last_frame"]
    shot_index: Literal[1] = 1
    image_path: str
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_video_path: str
    source_video_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_label(self) -> FrameConditioningPicture:
        if self.picture_label != f"<Picture {self.picture_index}>":
            raise ValueError("frame Picture label/index mismatch")
        return self


class FrameMetadata(SchemaModel):
    clip_uid: str
    source_video_path: str
    source_video_sha256: str
    first_frame_path: str
    first_frame_sha256: str
    last_frame_path: str
    last_frame_sha256: str
    source_duration_seconds: float = Field(gt=0, allow_inf_nan=False)
    extraction_policy_version: Literal["exact_decoded_first_last_png_v1"] = EXTRACTION_VERSION
    projection_version: Literal["h3_frame_conditioning_projection_v1"] = PROJECTION_VERSION


class FrameConditionedProduct(SchemaModel):
    schema_version: Literal["r2v.h3.frame_conditioned_product.1"] = "r2v.h3.frame_conditioned_product.1"
    projection_version: Literal["h3_frame_conditioning_projection_v1"] = PROJECTION_VERSION
    sample_id: str
    source_product_sample_id: str
    source_product_record_fingerprint: str
    source_product_records_sha256: str
    source_h3_sample_id: str
    clip_uid: str
    pair_type: Literal["canonical", "in_pair", "cross_pair"]
    conditioning_variant: ConditioningVariant
    visual_reference_mode: VisualReferenceMode
    visual_task: str
    target_video_path: str
    target_video_sha256: str
    target_duration_seconds: float = Field(gt=0, allow_inf_nan=False)
    frame_references: list[FrameConditioningPicture]
    subjects: list[RecaptionSubjectContract]
    audio_references: list[ReuseAudioReference]
    corrected_speech_segments: list[FinalQwen3SpeechSegment]
    rendered_h3_prompt: str
    warnings: list[str]
    record_fingerprint: str

    @model_validator(mode="after")
    def validate_record(self) -> FrameConditionedProduct:
        mode = self.visual_reference_mode
        if (mode not in DERIVED_MODES or self.visual_task != VISUAL_TASKS[mode]
                or self.sample_id != f"{self.source_product_sample_id}/{mode}"
                or [p.picture_index for p in self.frame_references] != list(range(1, len(self.frame_references) + 1))
                or [p.frame_role for p in self.frame_references] != (
                    ["first_frame", "last_frame"] if mode == "first_last_frame" else [mode])
                or any(s.source_picture_labels != ["<Picture 1>"] for s in self.subjects)
                or self.record_fingerprint != _hash(self.model_dump(mode="json", exclude={"record_fingerprint"}))):
            raise ValueError("frame product contract/fingerprint mismatch")
        split_sections(self.rendered_h3_prompt)
        return self


class FrameConditionedSummary(SchemaModel):
    schema_version: Literal["r2v.h3.frame_conditioned_product_summary.1"] = "r2v.h3.frame_conditioned_product_summary.1"
    projection_version: Literal["h3_frame_conditioning_projection_v1"] = PROJECTION_VERSION
    source_shadow_root: str
    source_ready_product_count: int = Field(ge=0)
    derived_product_count: int = Field(ge=0)
    visual_reference_mode_counts: dict[str, int]
    conditioning_variant_counts: dict[str, int]
    audio_kind_counts: dict[str, int]
    source_hashes: dict[str, str]
    frame_hashes: dict[str, str]
    records_sha256: str
    model_call_count: Literal[0] = 0
    production_artifacts_modified: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> FrameConditionedSummary:
        if (self.derived_product_count != 3 * self.source_ready_product_count
                or self.visual_reference_mode_counts != {m: self.source_ready_product_count for m in DERIVED_MODES}
                or sum(self.conditioning_variant_counts.values()) != self.derived_product_count):
            raise ValueError("frame projection summary must contain exactly 3N records")
        return self


@dataclass(frozen=True)
class ProjectionSource:
    product: AudioReuseProduct
    job: MimoClipJob
    subjects: list[RecaptionSubjectContract]


def _verify_hashes(hashes: dict[str, str]) -> None:
    if any(sha256_file(Path(p)) != digest for p, digest in hashes.items()):
        raise ValueError("frame projection input hash mismatch")


def _owned_file(path: Path, root: Path) -> Path:
    relative = path.relative_to(root)
    current = root
    for part in relative.parts:
        current = current / part
        if part in {".", ".."} or current.is_symlink():
            raise ValueError("frame sidecar/media symlink or traversal is not allowed")
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("frame sidecar/media escapes owned stage or is missing")
    return path.resolve(strict=True)


def load_projection_sources(shadow: Path) -> tuple[list[ProjectionSource], dict[str, str]]:
    """Reconstruct frozen products with existing materialization, never trust prose alone."""
    prepared, products = shadow / PREPARED_STAGE, shadow / PRODUCTS_STAGE
    for root in (prepared, products, shadow / "separation"):
        if root.is_symlink() or root.resolve().parent != shadow.resolve():
            raise ValueError("frame source stage escapes current run")
    hashes = {}

    def read(path, model, *, rows=False):
        path = _owned_file(path, shadow)
        hashes[str(path)] = sha256_file(path)
        content = path.read_text()
        return [model.model_validate_json(x) for x in content.splitlines() if x.strip()] if rows else model.model_validate_json(content)

    summary = read(products / "summary.json", AudioReuseProductSummary)
    records = read(products / "records.jsonl", AudioReuseProduct, rows=True)
    prep_summary = read(prepared / "summary.json", AudioReusePreparedSummary)
    inventory = read(prepared / "inventory.json", MimoInventory)
    annotations = read(prepared / "records.jsonl", AudioReusePreparedSource, rows=True)
    samples = read(prepared / "h3/samples.jsonl", FinalH3SampleV2, rows=True)
    stems = read(shadow / "separation/records.jsonl", SAMAudioStemRecord, rows=True)
    for path in (prepared / "inventory.json", prepared / "records.jsonl", prepared / "h3/samples.jsonl", shadow / "separation/records.jsonl"):
        if summary.source_hashes.get(str(path)) != hashes[str(path.resolve())]:
            raise ValueError("frame source product is not bound to current prepared/separation inputs")
    if (summary.materializer_version != AUDIO_REUSE_MATERIALIZER_VERSION
            or summary.clip_uids != [j.clip_uid for j in inventory.jobs]
            or len({p.sample_id for p in records}) != len(records)
            or summary.sample_count != len(records)
            or summary.ready_count != sum(p.status == "ready" for p in records)
            or prep_summary.inventory_fingerprint != inventory.inventory_fingerprint
            or prep_summary.prepared_records_sha256 != hashes[str(prepared / "records.jsonl")]
            or prep_summary.prepared_samples_sha256 != hashes[str(prepared / "h3/samples.jsonl")]
            or inventory.source_h3_samples_sha256 != prep_summary.prepared_samples_sha256):
        raise ValueError("frame source inventory/summary mismatch")
    validate_prepared_inputs(inventory, annotations, samples, stems)
    for dependencies in (summary.source_hashes, prep_summary.source_hashes):
        _verify_hashes(dependencies)
        for path, digest in dependencies.items():
            if path in hashes and hashes[path] != digest:
                raise ValueError("frame source dependency snapshots differ")
            hashes[path] = digest
    jobs = {j.clip_uid: j for j in inventory.jobs}
    anns = {r.clip_uid: r for r in annotations}
    by_sample = {s.sample_id: s for s in samples}
    result = []
    for product in records:
        sample = by_sample.get(product.source_h3_sample_id)
        if (sample is None or product.clip_uid != sample.clip_uid
                or product.source_h3_sample_sha256 != _hash(sample)
                or product.source_mimo_record_fingerprint != anns[product.clip_uid].source_reconcile_record_fingerprint
                or product.materializer_version != AUDIO_REUSE_MATERIALIZER_VERSION):
            raise ValueError("frame source product provenance mismatch")
        if product.status != "ready":
            continue
        job, record = jobs[product.clip_uid], anns[product.clip_uid]
        if record.status != "ready":
            raise ValueError("frame source annotation unavailable")
        options = {"conditioning_variant": product.conditioning_variant,
                   "reuse_audio_contracts": [r.contract for r in product.audio_references]}
        context = _prepare_materialization_context(sample, job, record, **options)
        corrected, rendered, _ = _materialize_sample(sample, job, record, **options)
        if (context.corrected != product.corrected_speech_segments or corrected != context.corrected
                or rendered != product.rendered_h3_prompt):
            raise ValueError("frame source product differs from deterministic reconstruction")
        for path, digest in [(job.target_video_path, job.target_video_sha256),
                             *[(r.contract.path, r.contract.sha256) for r in product.audio_references]]:
            if path in hashes and hashes[path] != digest:
                raise ValueError("frame source media provenance differs")
            hashes[path] = digest
        result.append(ProjectionSource(product, job, context.contract.subjects))
    _verify_hashes(hashes)
    return result, dict(sorted(hashes.items()))


def extract_frames(job: MimoClipJob, stage: Path, published: Path, *, ffmpeg: str) -> FrameMetadata:
    """Decode first frame and full reversed sequence: no seek, resize, crop, or FPS conversion."""
    video = Path(job.target_video_path)
    if sha256_file(video) != job.target_video_sha256:
        raise ValueError("target video changed before frame extraction")
    stage.mkdir(parents=True)
    for name, filters in (("first", []), ("last", ["-vf", "reverse"])):
        subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-noautorotate", "-i", str(video),
                        "-map", "0:v:0", "-an", "-sn", "-dn", *filters,
                        "-frames:v", "1", "-fps_mode", "passthrough", "-threads", "1",
                        str(stage / f"{name}.png")], check=True, capture_output=True)
        from PIL import Image
        with Image.open(stage / f"{name}.png") as image:
            if image.format != "PNG" or min(image.size) <= 0:
                raise ValueError("invalid extracted PNG")
            image.verify()
    if sha256_file(video) != job.target_video_sha256:
        raise ValueError("target video changed during frame extraction")
    metadata = FrameMetadata(
        clip_uid=job.clip_uid, source_video_path=job.target_video_path, source_video_sha256=job.target_video_sha256,
        source_duration_seconds=job.target_duration_seconds,
        first_frame_path=str(published / "first.png"), first_frame_sha256=sha256_file(stage / "first.png"),
        last_frame_path=str(published / "last.png"), last_frame_sha256=sha256_file(stage / "last.png"),
    )
    (stage / "metadata.json").write_text(metadata.model_dump_json(indent=2) + "\n")
    return metadata


def project_product(source: ProjectionSource, metadata: FrameMetadata, mode: VisualReferenceMode,
                    records_hash: str) -> FrameConditionedProduct:
    product, job = source.product, source.job
    if (metadata.clip_uid != job.clip_uid or metadata.source_video_path != job.target_video_path
            or metadata.source_video_sha256 != job.target_video_sha256
            or metadata.source_duration_seconds != job.target_duration_seconds):
        raise ValueError("frame metadata differs from source job")
    roles = ["first_frame", "last_frame"] if mode == "first_last_frame" else [mode]
    pictures = [FrameConditioningPicture(
        picture_index=i, picture_label=f"<Picture {i}>", frame_role=role,
        image_path=getattr(metadata, role + "_path"), image_sha256=getattr(metadata, role + "_sha256"),
        source_video_path=job.target_video_path, source_video_sha256=job.target_video_sha256,
    ).model_dump(mode="json") for i, role in enumerate(roles, 1)]
    values = {
        "schema_version": "r2v.h3.frame_conditioned_product.1", "projection_version": PROJECTION_VERSION,
        "sample_id": f"{product.sample_id}/{mode}", "source_product_sample_id": product.sample_id,
        "source_product_record_fingerprint": product.record_fingerprint, "source_product_records_sha256": records_hash,
        "source_h3_sample_id": product.source_h3_sample_id, "clip_uid": product.clip_uid, "pair_type": product.pair_type,
        "conditioning_variant": product.conditioning_variant, "visual_reference_mode": mode, "visual_task": VISUAL_TASKS[mode],
        "target_video_path": job.target_video_path, "target_video_sha256": job.target_video_sha256,
        "target_duration_seconds": job.target_duration_seconds, "frame_references": pictures,
        "subjects": [{**s.model_dump(mode="json"), "source_picture_labels": ["<Picture 1>"]} for s in source.subjects],
        "audio_references": [r.model_dump(mode="json") for r in product.audio_references],
        "corrected_speech_segments": [s.model_dump(mode="json") for s in product.corrected_speech_segments],
        "warnings": product.warnings, "rendered_h3_prompt": project_prompt(product.rendered_h3_prompt, mode, job.target_duration_seconds),
    }
    return FrameConditionedProduct(**values, record_fingerprint=_hash(values))


def materialize_frame_conditioned_products(*, audio_production_root: Path, shadow_run_id: str,
                                         ffmpeg: str = "ffmpeg", output_root: Path | None = None) -> FrameConditionedSummary:
    shadow = stem_shadow_root(audio_production_root, shadow_run_id).resolve(strict=True)
    output = (output_root or shadow / FRAME_STAGE).expanduser()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output = output.resolve()
    if output.parent != shadow or output.name.startswith("."):
        raise ValueError("frame output must be a new direct child of the current shadow run")
    sources, hashes = load_projection_sources(shadow)
    if any(Path(p).resolve().is_relative_to(output) for p in hashes):
        raise ValueError("frame output overlaps source inputs")
    for source in sources:
        if Path(source.job.clip_uid).name != source.job.clip_uid or source.job.clip_uid in {".", ".."}:
            raise ValueError("unsafe frame clip path")
        project_prompt(source.product.rendered_h3_prompt, "first_frame", source.job.target_duration_seconds)
    records = []
    frames = {}
    with tempfile.TemporaryDirectory(prefix=".frame-projection-", dir=shadow) as temporary:
        stage = Path(temporary) / "projection"
        stage.mkdir()
        for source in sources:
            uid = source.job.clip_uid
            if uid not in frames:
                frames[uid] = extract_frames(source.job, stage / "frames" / uid, output / "frames" / uid, ffmpeg=ffmpeg)
            records.extend(project_product(source, frames[uid], mode, hashes[str(shadow / PRODUCTS_STAGE / "records.jsonl")])
                           for mode in DERIVED_MODES)
        (stage / "records.jsonl").write_text("".join(r.model_dump_json() + "\n" for r in records))
        for uid, metadata in frames.items():
            _verify_hashes({str(stage / "frames" / uid / f"{role}.png"): getattr(metadata, role + "_frame_sha256")
                            for role in ("first", "last")})
        frame_hashes = {str(output / p.relative_to(stage)): sha256_file(p) for p in (stage / "frames").rglob("*") if p.is_file()}
        summary = FrameConditionedSummary(
            source_shadow_root=str(shadow), source_ready_product_count=len(sources), derived_product_count=len(records),
            visual_reference_mode_counts={m: len(sources) for m in DERIVED_MODES},
            conditioning_variant_counts=dict(sorted(Counter(r.conditioning_variant for r in records).items())),
            audio_kind_counts=dict(sorted(Counter(a.contract.kind for r in records for a in r.audio_references).items())),
            source_hashes=hashes, frame_hashes=frame_hashes, records_sha256=sha256_file(stage / "records.jsonl"),
        )
        (stage / "summary.json").write_text(summary.model_dump_json(indent=2) + "\n")
        _verify_hashes(hashes)
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
        stage.rename(output)
    return summary


def load_frame_conditioned_products(root: Path, shadow: Path) -> tuple[list[FrameConditionedProduct], dict[str, str]]:
    """QA validation only: no frame extraction or output publication."""
    if root.is_symlink() or root.resolve().parent != shadow.resolve():
        raise ValueError("frame projection belongs to a different shadow run")
    root, shadow = root.resolve(), shadow.resolve()
    for name in ("summary.json", "records.jsonl"):
        _owned_file(root / name, root)
    summary = FrameConditionedSummary.model_validate_json((root / "summary.json").read_text())
    sources, hashes = load_projection_sources(shadow)
    if summary.source_shadow_root != str(shadow) or summary.source_hashes != hashes:
        raise ValueError("frame projection source lineage is stale")
    if summary.records_sha256 != sha256_file(root / "records.jsonl"):
        raise ValueError("frame projection records hash mismatch")
    if any(not Path(p).resolve().is_relative_to(root / "frames") for p in summary.frame_hashes):
        raise ValueError("frame media escapes owned stage")
    for path in summary.frame_hashes:
        _owned_file(Path(path), root)
    _verify_hashes(summary.frame_hashes)
    records = [FrameConditionedProduct.model_validate_json(line) for line in (root / "records.jsonl").read_text().splitlines() if line.strip()]
    expected = []
    expected_paths = set()
    for source in sources:
        if Path(source.job.clip_uid).name != source.job.clip_uid or source.job.clip_uid in {".", ".."}:
            raise ValueError("unsafe frame clip path")
        folder = root / "frames" / source.job.clip_uid
        metadata = FrameMetadata.model_validate_json((folder / "metadata.json").read_text())
        for role in ("first", "last"):
            path = folder / f"{role}.png"
            if (getattr(metadata, role + "_frame_path") != str(path)
                    or summary.frame_hashes.get(str(path)) != getattr(metadata, role + "_frame_sha256")):
                raise ValueError("frame media metadata mismatch")
        expected_paths.update(str(folder / name) for name in ("metadata.json", "first.png", "last.png"))
        expected.extend(project_product(source, metadata, mode, hashes[str(shadow / PRODUCTS_STAGE / "records.jsonl")]) for mode in DERIVED_MODES)
    if (set(summary.frame_hashes) != expected_paths or records != expected
            or summary.source_ready_product_count != len(sources) or summary.derived_product_count != len(records)
            or summary.conditioning_variant_counts != dict(Counter(r.conditioning_variant for r in records))
            or summary.audio_kind_counts != dict(Counter(a.contract.kind for r in records for a in r.audio_references))):
        raise ValueError("frame projection differs from deterministic current products")
    hashes.update(summary.frame_hashes)
    hashes.update({str(root / name): sha256_file(root / name) for name in ("records.jsonl", "summary.json")})
    return records, hashes
