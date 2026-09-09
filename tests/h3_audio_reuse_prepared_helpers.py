"""Synthetic frozen stem records; no legacy MimoRecord provenance fabrication."""
from r2v_data_v2.h3.audio_reuse_prepared import (
    _FrozenStemRecord,
    _hash,
    load_reconcile_sources,
)
from r2v_data_v2.h3.mimo25_backend import (
    MimoBackendConfig,
    MimoCompletionDiagnostic,
    MimoMediaResolver,
    MimoUsage,
)


def stem_record_values(tmp_path, job, annotation, stem, *, backend_version=61, failed=False):
    provenance = MimoBackendConfig(
        media_resolver=MimoMediaResolver(mode="base64", media_root=tmp_path),
        api_key="fixture", transport="sglang",
    ).provenance().model_dump(mode="json", exclude={"configuration_fingerprint"})
    provenance["schema_version"] = f"r2v.h3.mimo25_backend.{backend_version}"
    provenance["configuration_fingerprint"] = _hash(provenance)
    modalities = ("target_video_visual_only", "target_video_speech_assembly", "target_video_audio_finalize")
    values = {
        "schema_version": "r2v.h3.mimo25_stem_reconcile.14",
        "clip_uid": job.clip_uid, "source_job_fingerprint": job.request_fingerprint,
        "source_stem_record_fingerprint": stem.record_fingerprint,
        "policy_version": "h3_mimo25_stem_reconcile_v6", "backend_provenance": provenance,
        "status": "failed" if failed else "ready", "annotation": annotation.model_dump(mode="json"),
        "failure_code": "fixture_failure" if failed else None,
        "failure_reason": "Frozen semantic failure" if failed else None, "failure_issues": [],
        "raw_responses": ["visual", "speech", "finalize"],
        "visual_raw_response": "visual", "speech_av_raw_response": "speech",
        "speaker_profile_raw_response": None, "audio_finalize_raw_response": "finalize",
        "diagnostics": [MimoCompletionDiagnostic(
            input_modality=m, usage=MimoUsage(), http_attempt_count=1,
        ).model_dump(mode="json") for m in modalities],
        "speaker_marker_polish_attempted": False, "speaker_marker_polish_applied": False,
        "speaker_marker_polish_needs_review": None, "speaker_marker_polish_raw_response": None,
        "speaker_marker_polish_error": None, "av_model_call_count": 2, "visual_model_call_count": 1,
        "audio_model_call_count": 0, "text_model_call_count": 0, "model_call_count": 3,
    }
    for kind in ("music", "sfx"):
        for field in ("description", "raw_response", "error"):
            values[f"{kind}_stem_{field}"] = None
    values["record_fingerprint"] = _hash(values)
    _FrozenStemRecord.model_validate(values)
    return values


def prepared_record(tmp_path, job, annotation, stem):
    import json

    root = tmp_path / "frozen-reconcile-fixture"
    root.mkdir(exist_ok=True)
    values = stem_record_values(tmp_path, job, annotation, stem)
    (root / "records.jsonl").write_text(json.dumps(values) + "\n")
    return load_reconcile_sources(root)[0]
