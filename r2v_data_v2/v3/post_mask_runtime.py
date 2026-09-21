"""Downstream-only Post-Mask execution. Caller holds the exclusive shard lock.

The main worker must be isolated to ``sam_gpu`` before importing/loading CUDA;
Boogu owns a separate interpreter on ``boogu_gpu``. ExecutionSettings is never
persisted into the semantic RunStorage config. No upstream stage is dispatched.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from r2v_data_v2.reconciliation import write_json_atomic
from r2v_data_v2.v3.config import RuntimeConfig, V3Config
from r2v_data_v2.v3.post_mask_production import (
    ShardPaths,
    hydrate_shard,
    initialize_shard,
)
from r2v_data_v2.v3.profiling import QwenConcurrencyGate, qwen_concurrency_gate
from r2v_data_v2.v3.runtime import ClipScopedStorage
from r2v_data_v2.v3.schemas import ClipRecord, DatasetRecord, DatasetSample
from r2v_data_v2.v3.storage import DatasetExporter, RunStorage, evaluate_export_state

PHASES = (
    "remove",
    "pair",
    "reference_edit",
    "reference_integrity",
    "instruct",
    "subject_attributes",
)

SCHEDULER_MODES = ("legacy_serial", "wavefront_v2", "parallel_review_v21")


@dataclass(frozen=True)
class ExecutionSettings:
    runtime: RuntimeConfig
    sam_gpu: str
    boogu_gpu: str
    qwen_lock_directory: Path
    clip_inflight: int = 8
    shard_inflight: int = 2
    scheduler_mode: str = "wavefront_v2"

    def __post_init__(self) -> None:
        from r2v_data_v2.v3 import config as config_module

        if type(self.clip_inflight) is not int or self.clip_inflight <= 0:
            raise ValueError("clip_inflight must be a positive integer")
        if type(self.shard_inflight) is not int or self.shard_inflight <= 0:
            raise ValueError("shard_inflight must be a positive integer")
        if self.scheduler_mode not in SCHEDULER_MODES:
            raise ValueError(
                f"unsupported Post-Mask scheduler mode: {self.scheduler_mode}"
            )

        if not self.sam_gpu.isdigit() or not self.boogu_gpu.isdigit():
            raise ValueError("Post-Mask requires physical numeric GPU IDs")
        if self.sam_gpu == self.boogu_gpu:
            raise ValueError("SAM and Boogu must use distinct GPUs")
        if not self.qwen_lock_directory.resolve().is_relative_to(
            config_module.ALLOWED_WRITABLE_ROOT.resolve()
        ):
            raise ValueError(
                "Qwen lock directory must stay inside allowed writable root"
            )

    def workers_for(self, stage: str) -> int:
        if stage in {"remove", "reference_edit", "subject_attributes"}:
            return 1 if self.scheduler_mode == "legacy_serial" else self.clip_inflight
        # Pair retains one complete donor view; existing CPU concurrency stays.
        if stage not in {"reference_integrity", "instruct"}:
            return 1
        return min(self.runtime.cpu_workers, getattr(self.runtime.stage_workers, stage))

    def qwen_gate(self) -> QwenConcurrencyGate:
        return QwenConcurrencyGate(
            self.runtime.qwen_max_inflight,
            lock_directory=self.qwen_lock_directory,
        )


@dataclass(frozen=True)
class ShardResult:
    completed: bool
    ready: int
    excluded: int
    corrupt: int
    sample_count: int
    retryable_clip_uids: tuple[str, ...]
    stage_counts: dict[str, dict[str, int | float]]


class _ShardStorage:
    """Exclude corrupt/stale destinations while keeping all admitted donors."""

    def __init__(
        self, storage: RunStorage, clip_uids: tuple[str, ...], write_lock=None
    ):
        self._storage = storage
        self.clip_uids = clip_uids
        self.write_lock = write_lock if write_lock is not None else threading.RLock()

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._storage, name)
        if callable(value) and name.startswith(
            ("write_", "append_", "update_", "create_")
        ):

            def write(*args, **kwargs):
                with self.write_lock:
                    return value(*args, **kwargs)

            return write
        return value

    def iter_clips(self) -> Iterator[ClipRecord]:
        for uid in self.clip_uids:
            yield self._storage.read_clip(uid)


def _scratch_root(run_root: Path, stage: str) -> Path:
    if stage not in PHASES:
        raise ValueError("unknown Post-Mask phase")
    root = run_root / ".post_mask_scratch" / stage
    if root.is_symlink() or root.parent.is_symlink():
        raise ValueError("scratch owner cannot be a symlink")
    return root


def cleanup_phase_scratch(run_root: Path, stage: str) -> None:
    """Only remove proven run-owned abandoned request directories.

    Call only while holding the shard lock and after the backend has closed.
    Generated references, candidates, attribute sidecars and logs are retained.
    """
    root = _scratch_root(run_root, stage)
    if not root.exists():
        return
    owner = root / ".owner.json"
    expected = {"run_root": str(run_root), "stage": stage}
    if (
        owner.is_symlink()
        or not owner.is_file()
        or json.loads(owner.read_text()) != expected
    ):
        raise ValueError("scratch directory has no matching owner")
    patterns = ["r2v-boogu-*"]
    if stage == "reference_edit":
        patterns.append("sam3-review-*")
    for pattern in patterns:
        for request in root.glob(pattern):
            if request.is_symlink():
                raise ValueError("owned scratch request cannot be a symlink")
            if request.is_dir():
                shutil.rmtree(request)


def _prepare_scratch(run_root: Path, stage: str) -> Path:
    root = _scratch_root(run_root, stage)
    if root.exists():
        cleanup_phase_scratch(run_root, stage)
    else:
        root.mkdir(parents=True)
        write_json_atomic(
            root / ".owner.json", {"run_root": str(run_root), "stage": stage}
        )
    return root


def _clip_digest(clip: ClipRecord) -> str:
    value = clip.model_dump(mode="json", exclude={"export"})
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _attribute_receipt(storage: RunStorage, uid: str) -> Path:
    return storage.clip_dir(uid) / ".post_mask_attributes.json"


#: Every final image one accepted attribute record can point at. The record's
#: own crop, its accepted base and every published variant are all part of the
#: final attribute authority, so the receipt digests the pixels, not just the
#: JSON that names them.
_ATTRIBUTE_RECORD_IMAGE_FIELDS = (
    "image_path",
    "accepted_base_image_path",
    "default_image_path",
)
_ATTRIBUTE_VARIANT_NAMES = ("alpha", "bbox", "generated_background")


def _attribute_record_images(record: Any) -> list[str]:
    """Every non-null final image path one attribute record references."""
    paths: list[str] = []
    for field in _ATTRIBUTE_RECORD_IMAGE_FIELDS:
        value = getattr(record, field, None)
        if isinstance(value, str) and value:
            paths.append(value)
    variants = getattr(record, "variants", None)
    if variants is not None:
        for name in _ATTRIBUTE_VARIANT_NAMES:
            variant = getattr(variants, name, None)
            value = getattr(variant, "image_path", None) if variant is not None else None
            if isinstance(value, str) and value:
                paths.append(value)
    return paths


def _contained_attribute_path(root: Path, relative: str) -> tuple[str, Path]:
    """Validate one final attribute artifact path and return its owner form."""
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ValueError("attribute artifact path must be relative")
    if ".." in candidate.parts:
        raise ValueError("attribute artifact path must not escape its root")
    resolved_root = root.resolve(strict=False)
    resolved = (resolved_root / candidate).resolve(strict=False)
    resolved.relative_to(resolved_root)
    if not resolved.is_file():
        raise ValueError(f"attribute artifact is missing: {relative}")
    return candidate.as_posix(), resolved


def _attribute_artifacts(storage: RunStorage, clip: ClipRecord) -> dict[str, str]:
    """Digest every durable artifact one clip's Subject Attributes published.

    The owner JSONs, the enriched sample and every final image the accepted
    records reference: the attribute crop, the accepted base and each variant
    PNG. Paths are validated as relative, contained and present, and the mapping
    is deduplicated and sorted so the digest is deterministic.
    """
    from r2v_data_v2.v3.subject_attributes import _load_durable_owner_artifact

    root = storage.root / "subject_attributes"
    collected: dict[str, Path] = {}
    for path in sorted((root / "owners" / clip.clip_uid).glob("*.json")):
        artifact = _load_durable_owner_artifact(
            path,
            output_root=root,
            sample_id=clip.clip_uid,
            owner_entity_id=path.stem,
        )
        if artifact is None:
            raise ValueError("invalid durable attribute owner artifact")
        collected[str(path.relative_to(root))] = path
        for record in artifact.records:
            for relative in _attribute_record_images(record):
                key, resolved = _contained_attribute_path(root, relative)
                collected[key] = resolved
    sample = root / "samples" / f"{clip.clip_uid}.json"
    if sample.exists():
        collected[str(sample.relative_to(root))] = sample
    return {
        key: hashlib.sha256(collected[key].read_bytes()).hexdigest()
        for key in sorted(collected)
    }


def _expected_attribute_receipt(storage: RunStorage, uid: str) -> dict[str, Any]:
    """The exact final attribute receipt of one clip, recomputed from disk."""
    clip = storage.read_clip(uid)
    return {
        "clip": _clip_digest(clip),
        "artifacts": _attribute_artifacts(storage, clip),
    }


def _publish_attribute_receipt(
    storage: RunStorage, uid: str, expected: Mapping[str, Any]
) -> None:
    """Create-or-verify one final attribute receipt; never silently overwrite."""
    path = _attribute_receipt(storage, uid)
    if path.is_file():
        try:
            existing = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"final attribute receipt is unreadable: {path}"
            ) from exc
        if existing != dict(expected):
            raise ValueError(f"final attribute receipt drifted: {path}")
        return
    write_json_atomic(path, dict(expected))


def _export_tree_sha256(paths: ShardPaths) -> str:
    """Combined digest of every published export file and its relative path."""
    digest = hashlib.sha256()
    for path in sorted(
        item for item in paths.export_root.rglob("*") if item.is_file()
    ):
        digest.update(path.relative_to(paths.export_root).as_posix().encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _subject_attribute_receipts_sha256(
    storage: RunStorage, clip_uids: Sequence[str]
) -> str:
    """Combined digest of one shard's final attribute receipts."""
    digest = hashlib.sha256()
    for uid in sorted(str(item) for item in clip_uids):
        path = _attribute_receipt(storage, uid)
        if not path.is_file():
            raise ValueError(f"final attribute receipt is missing: {path}")
        digest.update(uid.encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _attributes_complete(storage: RunStorage, clip: ClipRecord) -> bool:
    path = _attribute_receipt(storage, clip.clip_uid)
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text()) == {
            "clip": _clip_digest(clip),
            "artifacts": _attribute_artifacts(storage, clip),
        }
    except (OSError, ValueError):
        return False


def _terminal(clip: ClipRecord) -> bool:
    return (clip.pairing is not None and clip.pairing.status == "rejected") or (
        clip.reference_edit is not None and clip.reference_edit.status == "failed"
    )


def phase_needed(storage: RunStorage, clip: ClipRecord, stage: str) -> bool:
    config = storage.config
    if stage == "remove":
        bg = clip.references.background
        return (
            config.remove.enabled and bg is not None and bg.status == "pending_remove"
        )
    if _terminal(clip):
        return False
    if stage == "pair":
        return clip.pairing is None
    if stage == "reference_edit":
        return config.reference_edit.enabled and clip.reference_edit is None
    if stage == "reference_integrity":
        return config.reference_integrity.enabled and (
            clip.reference_integrity is None
            or clip.reference_integrity.status != "ready"
        )
    if stage == "instruct":
        return clip.instruction is None or clip.instruction.status != "ready"
    if stage == "subject_attributes":
        return not _attributes_complete(storage, clip)
    raise ValueError("unknown Post-Mask phase")


class _AttributeCompletion:
    def __init__(self, backend: Any, storage: RunStorage):
        self.backend, self.storage = backend, storage

    def attribute_completion(
        self, *, source_path: Path, output_path: Path, instruction: str, seed: int
    ) -> dict[str, object]:
        from PIL import Image

        from r2v_data_v2.v3.profiling import profile_model_call
        from r2v_data_v2.v3.reference_edit_boogu import resolve_boogu_1k_size

        root = (self.storage.root / "subject_attributes").resolve()
        source_path = source_path.resolve()
        output_path = output_path.resolve()
        source_path.relative_to(root)
        output_path.relative_to(root / "completion_candidates")
        with Image.open(source_path) as opened:
            opened.load()
            source = opened.convert("RGB")
        config = self.storage.config.reference_edit
        width, height = resolve_boogu_1k_size(
            *source.size,
            target_area=config.target_area,
            alignment=config.alignment,
        )
        started = time.perf_counter()
        with profile_model_call(
            component="boogu_attribute_completion",
            operation="complete_attribute",
            retry_index=0,
            model=str(config.model_path),
            input_text_chars=len(instruction),
            input_image_count=1,
            metadata={"thinking_enabled": False, "instruction_rewrite_enabled": False},
        ):
            result = self.backend.edit(
                source_rgb=source,
                instruction=instruction,
                width=width,
                height=height,
                thinking_enabled=False,
                instruction_rewrite_enabled=False,
                seed=seed,
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
        try:
            temporary.write_bytes(result.png_bytes)
            temporary.replace(output_path)
        finally:
            temporary.unlink(missing_ok=True)
        return {
            "width": width,
            "height": height,
            "model_call_time_seconds": time.perf_counter() - started,
            "thinking_enabled": False,
            "instruction_rewrite_enabled": False,
        }


class _SerializedSam:
    """Serialize only model entrypoints, not the surrounding clip algorithm."""

    def __init__(self, backend, resources):
        self.backend, self.resources = backend, resources

    def track(self, **kwargs):
        return self.resources.sam_call(self.backend.track, **kwargs)

    def segment_frame(self, **kwargs):
        return self.resources.sam_call(self.backend.segment_frame, **kwargs)

    def segment_generated_frame(self, **kwargs):
        return self.resources.sam_call(self.backend.segment_generated_frame, **kwargs)


class DownstreamPhaseAdapter:
    """One lazy phase resource owner; existing functions own all model policy."""

    def __init__(
        self,
        stage: str,
        storage: RunStorage,
        execution: ExecutionSettings,
        *,
        resources=None,
    ):
        self.stage, self.storage, self.execution = stage, storage, execution
        self.config = storage.config
        self.stack = ExitStack()
        self.lock = getattr(storage, "write_lock", threading.RLock())
        self.resources = resources
        self.kwargs: dict[str, Any] = {}

    def _own(self, resource: Any) -> Any:
        close = getattr(resource, "close", None)
        if callable(close):
            self.stack.callback(close)
        return resource

    def _boogu(self) -> Any:
        return self.resources

    def __enter__(self) -> Self:
        try:
            if self.resources is None:
                from r2v_data_v2.v3.post_mask_resources import PostMaskWorkerResources

                self.resources = self.stack.enter_context(
                    PostMaskWorkerResources(
                        self.config,
                        self.execution,
                        runtime_root=self.execution.qwen_lock_directory.parent
                        / "post-mask-runtime"
                        / uuid.uuid4().hex,
                    )
                )
            self._initialize()
        except BaseException:
            self.stack.close()
            raise
        return self

    def __exit__(self, *args: object) -> None:
        self.stack.close()

    def _initialize(self) -> None:
        stage, config = self.stage, self.config
        if (
            stage in {"remove", "reference_edit", "subject_attributes"}
            and os.environ.get("CUDA_VISIBLE_DEVICES") != self.execution.sam_gpu
        ):
            raise ValueError(
                "Post-Mask worker must be isolated to sam_gpu before model loading"
            )
        if stage == "remove":
            from r2v_data_v2.v3.boogu_remove_backend import (
                BooguBackgroundRemovalBackend,
            )
            from r2v_data_v2.v3.config import BOOGU_REMOVE_BACKEND
            from r2v_data_v2.v3.removal_judge import QwenBackgroundRemovalJudge
            from r2v_data_v2.v3.remove import remove_backgrounds

            if config.remove.backend != BOOGU_REMOVE_BACKEND:
                raise ValueError("Post-Mask production removal requires Boogu backend")
            self.function = remove_backgrounds
            self.kwargs = {
                "backend": BooguBackgroundRemovalBackend(
                    self._boogu(),
                    target_area=config.reference_edit.target_area,
                    alignment=config.reference_edit.alignment,
                ),
                "judge": self._own(
                    QwenBackgroundRemovalJudge(config.qwen.background_remove_judge)
                ),
            }
        elif stage == "pair":
            from r2v_data_v2.v3.pair import pair_clips

            # One full-shard call lets the existing implementation lazily own
            # its Qwen judges and donor second pass, with no policy duplication.
            self.function = pair_clips
        elif stage == "reference_edit":
            from r2v_data_v2.v3.reference_edit import reference_edit_clips
            from r2v_data_v2.v3.reference_edit_boogu import (
                QwenBooguReferenceEditJudge,
                Sam3BooguReferenceReviewer,
            )
            from r2v_data_v2.v3.scale_collapse_fallback_guard import (
                QwenScaleCollapseFallbackJudge,
            )

            self.function = reference_edit_clips
            scratch = _prepare_scratch(self.storage.root, stage)
            self.stack.callback(cleanup_phase_scratch, self.storage.root, stage)
            backend = self._boogu()
            judge = self._own(
                QwenBooguReferenceEditJudge(config.qwen.reference_edit_judge)
            )
            segmenter = self.resources.sam_backend(config.sam3)
            reviewer = Sam3BooguReferenceReviewer(
                _SerializedSam(segmenter, self.resources),
                temporary_root=scratch,
                max_area_growth_ratio=config.reference_edit.sam_max_area_growth_ratio,
                max_significant_components=config.reference_edit.sam_max_significant_components,
                min_candidate_scale_ratio=config.reference_edit.min_candidate_scale_ratio,
                max_candidate_center_shift=config.reference_edit.max_candidate_center_shift,
            )
            scale_judge = (
                self._own(
                    QwenScaleCollapseFallbackJudge(config.qwen.reference_edit_judge)
                )
                if config.reference_edit.scale_collapse_fallback_guard_mode == "qwen_v1"
                else None
            )
            self.kwargs = {
                "backend": backend,
                "judge": judge,
                "sam_reviewer": reviewer,
                "scale_collapse_judge": scale_judge,
                "manage_backend_lifecycle": False,
            }
            if self.execution.scheduler_mode == "parallel_review_v21":
                self.kwargs.update(
                    review_execution="parallel_independent",
                    review_observer=self.resources.record_parallel_review,
                )
        elif stage == "reference_integrity":
            from r2v_data_v2.v3.reference_integrity import (
                QwenReferenceIntegrityJudge,
                reference_integrity_clips,
            )

            self.function = reference_integrity_clips
            self.kwargs = {
                "judge": self._own(
                    QwenReferenceIntegrityJudge(config.qwen.reference_integrity_judge)
                )
            }
        elif stage == "instruct":
            from r2v_data_v2.v3.instruction import instruct_clips

            self.function = instruct_clips
            self.kwargs = {"client": None}  # Existing deterministic policy.
        elif stage == "subject_attributes":
            from r2v_data_v2.v3.subject_attributes import (
                QwenSubjectAttributeClient,
                QwenSubjectAttributeCompletionJudge,
                Sam3AttributeFrameSegmenter,
            )

            if config.subject_attribute_gme.enabled:
                raise ValueError(
                    "Post-Mask production requires subject attribute GME disabled"
                )
            client = self._own(QwenSubjectAttributeClient(config.qwen.candidate_judge))
            self.kwargs = {
                "discovery_client": client,
                "review_client": client,
                "segmentation_backend": _SerializedSam(
                    Sam3AttributeFrameSegmenter(
                        config.sam3, backend=self.resources.sam_backend(config.sam3)
                    ),
                    self.resources,
                ),
            }
            if config.subject_attributes.completion.enabled:
                self.kwargs.update(
                    completion_backend=_AttributeCompletion(
                        self._boogu(), self.storage
                    ),
                    completion_judge=self._own(
                        QwenSubjectAttributeCompletionJudge(
                            config.qwen.candidate_judge,
                            completion_component="qwen_attribute_completion_review",
                        )
                    ),
                )
        else:
            raise ValueError("unknown Post-Mask phase")

    def run(self, clip_uid: str | None = None) -> dict[str, Any]:
        if self.stage == "subject_attributes":
            from r2v_data_v2.v3.subject_attributes import (
                _load_durable_owner_artifact,
                process_subject_attribute_clip,
            )

            assert clip_uid is not None
            clip = self.storage.read_clip(clip_uid)
            accepted = clip.model_copy(
                update={
                    "export": evaluate_export_state(
                        clip,
                        require_reference_edit=self.config.reference_edit.enabled,
                        require_reference_integrity=self.config.reference_integrity.enabled,
                    )
                }
            )
            root = self.storage.root / "subject_attributes"
            result = process_subject_attribute_clip(
                self.config,
                storage=self.storage,
                output_root=root,
                clip=accepted,
                overwrite=False,
                **self.kwargs,
            )
            counts: dict[str, Any] = dict(result.to_counts())
            if result.totals.owner_processing_failures:
                counts["subject_attribute_owner_failures"] = (
                    result.totals.owner_processing_failures
                )
            # Owner artifacts (including fail-closed failures) are the existing
            # durable policy. Exceptions that left no artifact remain retryable.
            durable_failures = 0
            for path in (root / "owners" / clip_uid).glob("*.json"):
                artifact = _load_durable_owner_artifact(
                    path,
                    output_root=root,
                    sample_id=clip_uid,
                    owner_entity_id=path.stem,
                )
                if artifact is not None:
                    durable_failures += artifact.metrics.failures
            counts["retryable_pending"] = max(
                0, int(counts.get("failures", 0)) - durable_failures
            )
            if not counts["retryable_pending"]:
                write_json_atomic(
                    _attribute_receipt(self.storage, clip_uid),
                    {
                        "clip": _clip_digest(clip),
                        "artifacts": _attribute_artifacts(self.storage, clip),
                    },
                )
            return counts
        scoped = (
            self.storage
            if self.stage == "pair"
            else ClipScopedStorage(self.storage, clip_uid, shared_write_lock=self.lock)
        )
        return self.function(
            self.config, scoped, overwrite=False, **self.kwargs
        ).to_dict()


def _export_identity(storage: RunStorage, paths: ShardPaths) -> dict[str, Any]:
    digest = hashlib.sha256()
    clip_count = 0
    for clip in storage.iter_clips():
        digest.update(
            json.dumps(
                {
                    "clip_uid": clip.clip_uid,
                    "source": clip.source.model_dump(mode="json"),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        digest.update(b"\n")
        clip_count += 1
    return {
        "shard_identity": json.loads(paths.identity_path.read_text()),
        "git_commit": storage.read_run().git_commit,
        "clip_count": clip_count,
        "clip_sources_sha256": digest.hexdigest(),
    }


def _validate_publication(storage: RunStorage, paths: ShardPaths) -> DatasetRecord:
    from r2v_data_v2.v3.reference_variants import ReferenceVariantsManifestRecord
    from r2v_data_v2.v3.schemas import render_annotation_plain_text
    from r2v_data_v2.v3.subject_attributes import EnrichedSample

    identity = _export_identity(storage, paths)
    intent = paths.state_root / "export_identity.json"
    if not intent.is_file() or json.loads(intent.read_text()) != identity:
        raise ValueError("export publication provenance/identity mismatch")
    dataset = DatasetRecord.model_validate_json(
        (paths.export_root / "dataset.json").read_text()
    )
    run, config = storage.read_run(), storage.config
    if (
        dataset.config_hash != run.config_hash
        or dataset.git_commit != run.git_commit
        or dataset.dataset_version != paths.export_root.name
        or dataset.annotation_model != Path(config.qwen.annotation.model).name
        or dataset.background_remove_backend != config.remove.backend
    ):
        raise ValueError("export dataset identity mismatch")
    samples = [
        DatasetSample.model_validate_json(line)
        for line in (paths.export_root / "samples.jsonl").read_text().splitlines()
        if line.strip()
    ]
    expected = {
        clip.clip_uid
        for clip in storage.iter_clips()
        if evaluate_export_state(
            clip,
            require_reference_edit=config.reference_edit.enabled,
            require_reference_integrity=config.reference_integrity.enabled,
        ).accepted
    }
    if (
        len(samples) != dataset.sample_count
        or {s.sample_id for s in samples} != expected
        or len({s.sample_id for s in samples}) != len(samples)
        or sum(len(s.references) for s in samples) != dataset.reference_count
    ):
        raise ValueError("export publication sample inventory mismatch")
    for sample in samples:
        clip = storage.read_clip(sample.sample_id)
        assert clip.annotation is not None and clip.instruction is not None
        caption = (
            render_annotation_plain_text(
                clip.annotation.instruction_template,
                clip.annotation.entities,
                clip.annotation.background,
            )
            if clip.annotation.instruction_template
            else clip.annotation.t2v_caption
        )
        if (
            sample.target_video != clip.source.video_path
            or sample.source.parent_video_id != clip.source.parent_video_id
            or sample.source.clip_suffix != clip.source.clip_suffix
            or sample.r2v_instruction != clip.instruction.r2v_instruction
            or sample.t2v_caption != caption
        ):
            raise ValueError("export publication source mismatch")
        for reference in sample.references:
            path = paths.export_root / reference.image_path
            if (
                not path.resolve().is_relative_to(paths.export_root.resolve())
                or not path.is_file()
            ):
                raise ValueError("export publication reference is missing/unsafe")
    enriched_path = paths.export_root / "enriched_samples.jsonl"
    variants_path = paths.export_root / "reference_variants.jsonl"
    enriched = (
        [
            EnrichedSample.model_validate_json(line)
            for line in enriched_path.read_text().splitlines()
            if line.strip()
        ]
        if enriched_path.exists()
        else []
    )
    variants = (
        [
            ReferenceVariantsManifestRecord.model_validate_json(line)
            for line in variants_path.read_text().splitlines()
            if line.strip()
        ]
        if variants_path.exists()
        else []
    )
    DatasetExporter(config, storage)._validate_export_tree(
        paths.export_root, samples, enriched, variants
    )
    return dataset


def _cleanup_export_staging(storage: RunStorage, paths: ShardPaths) -> None:
    intent = paths.state_root / "export_identity.json"
    if not intent.is_file():
        return
    if json.loads(intent.read_text()) != _export_identity(storage, paths):
        raise ValueError("export staging owner identity mismatch")
    pattern = re.compile(re.escape(f".{paths.export_root.name}.tmp-") + r"[0-9a-f]{32}")
    for path in paths.export_root.parent.glob(f".{paths.export_root.name}.tmp-*"):
        if not pattern.fullmatch(path.name):
            continue
        if path.is_symlink():
            raise ValueError("export staging owner cannot be a symlink")
        if path.is_dir():
            shutil.rmtree(path)


def _record_phase_result(
    storage, stage, affected, values, error, counts, pending, emit
):
    """Keep V1 terminal/retryable bookkeeping shared by both execution paths."""
    for key, value in values.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            counts[key] = counts.get(key, 0) + value
    for uid in affected:
        clip = storage.read_clip(uid)
        retryable = phase_needed(storage, clip, stage) or (
            error is not None and not _terminal(clip)
        )
        if values.get("retryable_pending", 0):
            retryable = True
        if retryable:
            pending.add(uid)
        if (
            error is not None
            or retryable
            or values.get("failures", 0)
            or _terminal(clip)
            and clip.reference_edit is not None
            and clip.reference_edit.status == "failed"
        ):
            reason = str(error) if error else "phase has no durable successful outcome"
            diagnostics = (
                {
                    "failures": values.get("failures", 0),
                    "retryable_pending": values.get("retryable_pending", 0),
                }
                if stage == "subject_attributes"
                else {}
            )
            if stage == "subject_attributes" and values.get(
                "subject_attribute_owner_failures"
            ):
                diagnostics["subject_attribute_owner_failures"] = values[
                    "subject_attribute_owner_failures"
                ]
            storage.append_failure(
                stage=stage,
                clip_uid=uid,
                reason=reason,
                details={"retryable": retryable, **diagnostics},
            )
            counts["clip_failures"] = counts.get("clip_failures", 0) + 1
            emit(
                "post_mask_clip_failed",
                stage=stage,
                clip_uid=uid,
                reason=reason,
                retryable=retryable,
                **diagnostics,
            )


def _post_pair_wavefront(
    storage, execution, resources, gate, pending, emit, stage_counts, adapter_factory
):
    from r2v_data_v2.v3.post_mask_wavefront import bounded_results

    stages = PHASES[2:]
    counts_lock = threading.Lock()
    adapter_lock = threading.Lock()
    adapters, setup_errors = {}, {}
    stage_limits = {
        stage: threading.Semaphore(execution.workers_for(stage))
        for stage in ("reference_integrity", "instruct")
    }
    for stage in stages:
        stage_counts[stage] = {"scheduled": 0}
    with ExitStack() as stack:

        def adapter_for(stage):
            # Lazy initialization is serialized, not the per-clip algorithms.
            with adapter_lock:
                if stage in setup_errors:
                    raise setup_errors[stage]
                if stage not in adapters:
                    extra = (
                        {"resources": resources}
                        if adapter_factory is DownstreamPhaseAdapter
                        else {}
                    )
                    try:
                        adapters[stage] = stack.enter_context(
                            adapter_factory(stage, storage, execution, **extra)
                        )
                    except Exception as exc:
                        setup_errors[stage] = exc
                        raise
                return adapters[stage]

        def clip_pipeline(uid):
            with resources.activity("clip_pipelines"), qwen_concurrency_gate(gate):
                for stage in stages:
                    if resources.stopping:
                        with counts_lock:
                            pending.add(uid)
                        return
                    clip = storage.read_clip(uid)
                    if not phase_needed(storage, clip, stage):
                        continue
                    with counts_lock:
                        counts = stage_counts[stage]
                        counts["scheduled"] += 1
                        if counts["scheduled"] == 1:
                            emit("post_mask_stage_started", stage=stage)
                    values, error = {}, None
                    try:
                        adapter = adapter_for(stage)
                        with ExitStack() as call_stack:
                            if stage in stage_limits:
                                call_stack.enter_context(stage_limits[stage])
                            values = adapter.run(uid)
                    except Exception as exc:  # noqa: BLE001 - same clip isolation as V1
                        error = exc
                    with counts_lock:
                        _record_phase_result(
                            storage, stage, [uid], values, error, counts, pending, emit
                        )
                        if uid in pending:
                            return

        for _ in bounded_results(
            clip_pipeline,
            (uid for uid in storage.clip_uids if uid not in pending),
            execution.clip_inflight,
            on_interrupt=resources.stop_accepting,
        ):
            pass
    for stage in stages:
        counts = stage_counts[stage]
        counts["skipped_existing_or_ineligible"] = (
            len(storage.clip_uids) - counts["scheduled"]
        )
        storage.update_stage_counts(
            stage, {k: v for k, v in counts.items() if isinstance(v, int)}
        )
        emit("post_mask_stage_completed", stage=stage, counters=counts)


def _run_post_mask_shard(
    base_config: V3Config,
    *,
    entity_mask_root: Path,
    paths: ShardPaths,
    git_commit: str,
    execution: ExecutionSettings,
    event_callback: Callable[[dict[str, Any]], None] | None = None,
    adapter_factory: Callable[..., Any] = DownstreamPhaseAdapter,
    resources=None,
) -> ShardResult:
    """Execute one locked shard; false completion means retryable work remains.

    The caller owns worker process isolation and the cross-process shard lock.
    Terminal algorithm rejections/failures remain durable and are not overwritten.
    """

    def emit(event: str, **details: Any) -> None:
        if event_callback is not None:
            event_callback({"event": event, "shard": paths.shard_path.stem, **details})

    emit("post_mask_shard_started")
    storage = initialize_shard(base_config, paths, git_commit=git_commit)
    hydrated = hydrate_shard(
        storage,
        entity_mask_root=entity_mask_root,
        shard_path=paths.shard_path,
        paths=paths,
    )
    emit(
        "post_mask_hydrate_completed",
        ready=hydrated.ready,
        excluded=hydrated.excluded,
        corrupt=hydrated.corrupt,
    )
    selected = _ShardStorage(storage, hydrated.clip_uids)
    _cleanup_export_staging(selected, paths)
    stage_counts: dict[str, dict[str, int | float]] = {}
    pending: set[str] = set()
    gate = resources.qwen_gate
    if not paths.export_root.exists():
        # 61fc7a7 legacy keeps every logical stage as a whole-shard barrier,
        # including its existing CPU concurrency inside integrity/instruct.
        legacy_serial = execution.scheduler_mode == "legacy_serial"
        for stage in PHASES if legacy_serial else PHASES[:2]:
            todo = [
                uid
                for uid in hydrated.clip_uids
                if uid not in pending
                and phase_needed(selected, selected.read_clip(uid), stage)
            ]
            counts: dict[str, int | float] = {
                "scheduled": len(todo),
                "skipped_existing_or_ineligible": len(hydrated.clip_uids) - len(todo),
            }
            results: list[tuple[str | None, dict[str, Any], Exception | None]] = []
            if todo:
                emit("post_mask_stage_started", stage=stage, scheduled=len(todo))
                phase_storage = selected
                if stage == "pair":
                    # Both pair passes enumerate this view. Keep completed
                    # eligible donors, but never advance a predecessor failure.
                    phase_storage = _ShardStorage(
                        storage,
                        tuple(uid for uid in hydrated.clip_uids if uid not in pending),
                        selected.write_lock,
                    )
                try:
                    extra = (
                        {"resources": resources}
                        if adapter_factory is DownstreamPhaseAdapter
                        else {}
                    )
                    with adapter_factory(
                        stage, phase_storage, execution, **extra
                    ) as adapter:

                        def invoke(
                            uid: str | None,
                        ) -> tuple[str | None, dict[str, Any], Exception | None]:
                            try:
                                with qwen_concurrency_gate(gate):
                                    return uid, adapter.run(uid), None
                            except Exception as exc:  # noqa: BLE001 - isolate clip failures
                                return uid, {}, exc

                        if stage == "pair":
                            results = [invoke(None)]
                        elif execution.workers_for(stage) == 1:
                            results = [invoke(uid) for uid in todo]
                        else:
                            from r2v_data_v2.v3.post_mask_wavefront import (
                                bounded_results,
                            )

                            results = list(
                                bounded_results(
                                    invoke,
                                    todo,
                                    execution.workers_for(stage),
                                    on_interrupt=resources.stop_accepting,
                                )
                            )
                except Exception as exc:  # noqa: BLE001 - close resources, preserve shard progress
                    results.append((None, {}, exc))
                for uid, values, error in results:
                    _record_phase_result(
                        selected,
                        stage,
                        todo if uid is None else [uid],
                        values,
                        error,
                        counts,
                        pending,
                        emit,
                    )
            stage_counts[stage] = counts
            storage.update_stage_counts(
                stage,
                {key: value for key, value in counts.items() if isinstance(value, int)},
            )
            emit("post_mask_stage_completed", stage=stage, counters=counts)
        if not legacy_serial:
            _post_pair_wavefront(
                selected,
                execution,
                resources,
                gate,
                pending,
                emit,
                stage_counts,
                adapter_factory,
            )
        if pending:
            emit("post_mask_shard_incomplete", retryable_clip_uids=sorted(pending))
            return ShardResult(
                False,
                hydrated.ready,
                hydrated.excluded,
                hydrated.corrupt,
                0,
                tuple(sorted(pending)),
                stage_counts,
            )
        from r2v_data_v2.v3.subject_attributes import (
            reconcile_subject_attribute_outputs,
        )

        reconcile_subject_attribute_outputs(
            storage=selected,
            output_root=storage.root / "subject_attributes",
            owner_limit=None,
            invocation_wall_time_seconds=0.0,
        )
        write_json_atomic(
            paths.state_root / "export_identity.json", _export_identity(selected, paths)
        )
        emit("post_mask_stage_started", stage="export")
        DatasetExporter(storage.config, selected).export(overwrite=False)
    dataset = _validate_publication(selected, paths)
    emit(
        "post_mask_stage_completed",
        stage="export",
        counters={"sample_count": dataset.sample_count},
    )
    completed = {
        "identity": _export_identity(selected, paths),
        "sample_count": dataset.sample_count,
    }
    write_json_atomic(paths.state_root / "completed.json", completed)
    emit(
        "post_mask_shard_completed",
        ready=hydrated.ready,
        excluded=hydrated.excluded,
        corrupt=hydrated.corrupt,
        sample_count=dataset.sample_count,
    )
    return ShardResult(
        True,
        hydrated.ready,
        hydrated.excluded,
        hydrated.corrupt,
        dataset.sample_count,
        (),
        stage_counts,
    )


def run_post_mask_shard(
    base_config: V3Config,
    *,
    entity_mask_root: Path,
    paths: ShardPaths,
    git_commit: str,
    execution: ExecutionSettings,
    event_callback=None,
    adapter_factory=DownstreamPhaseAdapter,
    resources=None,
) -> ShardResult:
    from r2v_data_v2.v3.post_mask_resources import PostMaskWorkerResources

    with ExitStack() as stack:
        if resources is None:
            resources = stack.enter_context(
                PostMaskWorkerResources(
                    base_config,
                    execution,
                    runtime_root=paths.state_root / "runtime" / uuid.uuid4().hex,
                )
            )
        return _run_post_mask_shard(
            base_config,
            entity_mask_root=entity_mask_root,
            paths=paths,
            git_commit=git_commit,
            execution=execution,
            event_callback=event_callback,
            adapter_factory=adapter_factory,
            resources=resources,
        )
