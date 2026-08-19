from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import r2v_data_v2.h3.text_usability as usability_module
from r2v_data_v2.h3.asr_transcription import ASRDecoderDiagnostics
from r2v_data_v2.h3.text_usability import (
    TEXT_USABILITY_LANGUAGE_PROBABILITY_THRESHOLD,
    TEXT_USABILITY_OUTPUT_DIRECTORY,
    TEXT_USABILITY_POLICY_VERSION,
    TextUsabilityPolicy,
    TextUsabilitySegment,
    assess_text_usability,
    plan_text_usability,
    publish_text_usability,
    text_is_trusted,
)
from tests.test_h3_asr_v2_text_calibration import _write_asr_root


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in root.rglob("*")
        if path.is_file()
    }


@pytest.fixture
def production_source(tmp_path: Path):
    root = tmp_path / "audio-run"
    inventory, records = _write_asr_root(
        root / "production" / "asr_v2", mode="production"
    )
    return root, inventory, records


def _diagnostics(language_probability: float | None) -> ASRDecoderDiagnostics:
    return ASRDecoderDiagnostics(
        detected_language="zh",
        language_probability=language_probability,
        avg_log_probability=-99.0,
        no_speech_probability=0.99,
        compression_ratio=9.0,
        decoder_segment_count=1,
    )


def _record_with(
    base,
    *,
    status: str = "transcribed",
    text: str | None = "raw transcript",
    language_probability: float | None = 0.65,
    **updates: object,
):
    return base.model_copy(
        update={
            "status": status,
            "text": text,
            "diagnostics": _diagnostics(language_probability),
            **updates,
        }
    )


def _assess(record, fingerprint: str = "f" * 64) -> TextUsabilitySegment:
    return assess_text_usability(
        record,
        source_inventory_fingerprint=fingerprint,
    )


def test_policy_freezes_exact_rounded_language_probability_threshold() -> None:
    policy = TextUsabilityPolicy()

    assert policy.version == TEXT_USABILITY_POLICY_VERSION
    assert policy.language_probability_threshold == 0.65
    assert TEXT_USABILITY_LANGUAGE_PROBABILITY_THRESHOLD == 0.65
    assert policy.language_probability_threshold != 0.680419921875
    assert policy.language_probability_threshold != 0.516479492188
    assert policy.language_probability_threshold != 0.5
    assert policy.policy_validated is True
    assert len(policy.fingerprint()) == 64

    with pytest.raises(ValueError, match="0.65"):
        TextUsabilityPolicy(language_probability_threshold=0.5)


@pytest.mark.parametrize(
    ("language_probability", "expected"),
    [
        (0.65, True),
        (0.649999999999, False),
        (0.650000000001, True),
    ],
)
def test_language_probability_boundary_is_inclusive(
    production_source, language_probability: float, expected: bool
) -> None:
    _, _, records = production_source
    record = _record_with(records[0], language_probability=language_probability)

    assert text_is_trusted(record) is expected
    assessment = _assess(record)
    assert (assessment.text_status == "trusted") is expected


@pytest.mark.parametrize("status", ["uncertain", "failed"])
def test_non_transcribed_backend_status_is_hidden(
    production_source, status: str
) -> None:
    _, _, records = production_source
    record = _record_with(
        records[0],
        status=status,
        text=None,
        language_probability=1.0,
    )

    assessment = _assess(record)
    assert assessment.text_status == "hidden"
    assert assessment.trusted_text is None
    assert assessment.reason_codes == ["raw_text_unavailable"]


def test_empty_text_and_missing_language_probability_are_hidden(
    production_source,
) -> None:
    _, _, records = production_source
    empty = _assess(_record_with(records[0], text="", language_probability=1.0))
    missing = _assess(
        _record_with(records[0], text="source text", language_probability=None)
    )

    assert empty.reason_codes == ["raw_text_unavailable"]
    assert missing.reason_codes == ["language_probability_unavailable"]
    assert empty.trusted_text is missing.trusted_text is None


def test_trusted_text_is_exact_raw_asr_text_without_rewriting(
    production_source,
) -> None:
    _, _, records = production_source
    raw = "  多蒙八贤王力保。 Mixed CASE!  "
    assessment = _assess(_record_with(records[0], text=raw))

    assert assessment.text_status == "trusted"
    assert assessment.trusted_text == raw
    assert assessment.source_raw_text_sha256 == hashlib.sha256(raw.encode()).hexdigest()


def test_identity_fields_are_preserved_and_never_gate_text(production_source) -> None:
    _, _, records = production_source
    base = records[0]
    mapped = _assess(_record_with(base, language_probability=0.8))
    ambiguous = _assess(
        _record_with(
            base,
            language_probability=0.8,
            cluster_binding_status="ambiguous",
            entity_id=None,
            entity_occurrence_id=None,
            identity_scope="unresolved",
        )
    )
    unbound = _assess(
        _record_with(
            base,
            language_probability=0.8,
            cluster_binding_status="unbound",
            entity_id=None,
            entity_occurrence_id=None,
            identity_scope="unresolved",
        )
    )

    assert (
        mapped.text_status == ambiguous.text_status == unbound.text_status == "trusted"
    )
    assert mapped.entity_id == base.entity_id
    assert mapped.entity_occurrence_id == base.entity_occurrence_id
    assert mapped.speaker_cluster_id == base.speaker_cluster_id
    assert ambiguous.cluster_binding_status == "ambiguous"
    assert unbound.cluster_binding_status == "unbound"


def test_production_publication_covers_all_179_segments_dynamically(
    production_source,
) -> None:
    root, inventory, _ = production_source
    plan = plan_text_usability(
        audio_run_root=root,
        expected_source_inventory_fingerprint=inventory.inventory_fingerprint,
    )
    summary = publish_text_usability(
        audio_run_root=root,
        expected_source_inventory_fingerprint=inventory.inventory_fingerprint,
    )
    output = root / TEXT_USABILITY_OUTPUT_DIRECTORY
    rows = [
        json.loads(line)
        for line in (output / "segments.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert plan["source_segment_count"] == 179
    assert plan["output_root"] == str(output)
    assert summary.segment_count == len(inventory.jobs) == len(rows) == 179
    assert [(row["target_clip_uid"], row["segment_id"]) for row in rows] == [
        (job.target_clip_uid, job.segment_id) for job in inventory.jobs
    ]
    assert sorted(path.name for path in output.iterdir()) == [
        "inventory.json",
        "segments.jsonl",
        "summary.json",
    ]
    assert not any("202" in path.name for path in output.parent.iterdir())


def test_publication_is_read_only_for_asr_voice_pair_and_embedding_assets(
    production_source,
) -> None:
    root, inventory, _ = production_source
    for relative, content in (
        ("production/primary_voice/sentinel.json", b"voice"),
        ("production/pairs/sentinel.json", b"pairs"),
        ("production/embedding/sentinel.json", b"embedding"),
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    before = _tree_hashes(root)

    summary = publish_text_usability(
        audio_run_root=root,
        expected_source_inventory_fingerprint=inventory.inventory_fingerprint,
    )
    after = _tree_hashes(root)

    assert {
        path: digest
        for path, digest in after.items()
        if not path.startswith("production/text_usability/")
    } == before
    assert summary.raw_asr_preserved is True
    assert summary.voice_reference_quality_independent is True
    assert summary.speaker_entity_identity_independent is True
    assert summary.identity_used_as_text_gate is False
    assert summary.final_renderer_applied is False


def test_source_change_before_publish_fails_closed_and_cleans_temporary_output(
    production_source,
) -> None:
    root, inventory, _ = production_source
    source_segments = root / "production" / "asr_v2" / "segments.jsonl"
    original = source_segments.read_bytes()

    def mutate_source() -> None:
        source_segments.write_bytes(original + b"\n")

    with pytest.raises(ValueError, match="source changed"):
        publish_text_usability(
            audio_run_root=root,
            expected_source_inventory_fingerprint=inventory.inventory_fingerprint,
            before_publish=mutate_source,
        )
    assert not (root / TEXT_USABILITY_OUTPUT_DIRECTORY).exists()
    assert not list((root / "production").glob(".text_usability.tmp-*"))


def test_overwrite_is_atomic_and_output_schemas_reconcile(production_source) -> None:
    root, inventory, _ = production_source
    summary = publish_text_usability(
        audio_run_root=root,
        expected_source_inventory_fingerprint=inventory.inventory_fingerprint,
    )
    output = root / TEXT_USABILITY_OUTPUT_DIRECTORY
    first = _tree_hashes(output)
    repeated = publish_text_usability(
        audio_run_root=root,
        expected_source_inventory_fingerprint=inventory.inventory_fingerprint,
        overwrite=True,
    )
    second = _tree_hashes(output)

    assert repeated == summary
    assert second == first
    assert not list((root / "production").glob(".text_usability.old-*"))
    inventory_payload = json.loads((output / "inventory.json").read_text())
    summary_payload = json.loads((output / "summary.json").read_text())
    assert inventory_payload["schema_version"] == "r2v.h3.text_usability_inventory.1"
    assert summary_payload["schema_version"] == "r2v.h3.text_usability_summary.1"
    assert inventory_payload["policy_validated"] is True
    assert inventory_payload["whisper_calls"] == 0
    assert inventory_payload["diarizen_calls"] == 0
    assert inventory_payload["gpu_calls"] == 0


def test_no_model_backend_gpu_or_renderer_dependency_is_present() -> None:
    source = Path(usability_module.__file__).read_text(encoding="utf-8")

    assert "FasterWhisperASRBackend" not in source
    assert "DiariZenBackend" not in source
    assert "torch" not in source
    assert "cuda" not in source.lower()
    assert "<d>" not in source
