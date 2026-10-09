"""Durable response boundary around the unchanged Two-step backend."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

from openai import OpenAI

from r2v_data_v2.h3.mimo25_backend import MimoBackendFailure
from r2v_data_v2.h3.mimo26_two_step_backend import TwoStepOpenAIMimo26Backend
from r2v_data_v2.h3.t2va_production import atomic_json


class InterruptedRequestUnresolved(RuntimeError):
    pass


def _json_completion(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, SimpleNamespace):
        return {key: _json_completion(item) for key, item in vars(value).items()}
    if isinstance(value, dict):
        return {key: _json_completion(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_completion(item) for item in value]
    return value


class DurableCompletions:
    def __init__(self, root, client, *, clip_uid, event=None):
        self.root, self.clip_uid = Path(root), clip_uid
        self.root.mkdir(parents=True, exist_ok=True)
        self.client = client.with_options(max_retries=0) if isinstance(client, OpenAI) else client
        self.event = event or (lambda *args: None)
        self.new_request_count = self.response_replay_count = 0
        self.unresolved_turns = []
        self.request_settings_match = True
        self.chat = SimpleNamespace(completions=self)

    def create(self, **payload):
        name = payload["response_format"]["json_schema"]["name"]
        turn = {"MimoVisualDraft": "visual", "MimoJointAVAudioDraft": "joint"}[name]
        response_path = self.root / f"{turn}.response.json"
        intent_path = self.root / f"{turn}.request.json"
        if response_path.exists():
            self.response_replay_count += 1
            return json.loads(response_path.read_text())["completion"]
        if intent_path.exists():
            self.unresolved_turns.append(turn)
            raise InterruptedRequestUnresolved(f"interrupted_request_unresolved:{turn}")
        if not self.request_settings_match:
            raise ValueError("resume request settings differ from saved inference provenance")
        started = time.time()
        atomic_json(intent_path, {"clip_uid": self.clip_uid, "turn": turn,
                                 "started_at": started, "model": payload["model"]})
        self.event("request_started", turn)
        self.new_request_count += 1
        try:
            completion = self.client.chat.completions.create(**payload)
        except Exception:
            self.unresolved_turns.append(turn)
            raise
        atomic_json(response_path, {"clip_uid": self.clip_uid, "turn": turn,
            "started_at": started, "completed_at": time.time(),
            "completion": _json_completion(completion)})
        self.event("response_saved", turn)
        return completion

    def audit(self):
        return {"new_request_count": self.new_request_count,
                "response_replay_count": self.response_replay_count,
                "unresolved_turns": self.unresolved_turns,
                "model_call_count": sum((self.root / f"{turn}.request.json").is_file()
                                        for turn in ("visual", "joint"))}


def reconcile_group_job(*, root, config, client, job, stems, event=None) -> dict:
    durable = DurableCompletions(root, client, clip_uid=job.clip_uid, event=event)
    backend = TwoStepOpenAIMimo26Backend(config, stem_records_by_clip={}, client=durable)
    current_provenance = backend.provenance.model_dump(mode="json")
    provenance_path = Path(root) / "inference_provenance.json"
    if not provenance_path.exists():
        atomic_json(provenance_path, current_provenance)
    inference_provenance = json.loads(provenance_path.read_text())
    # Old responses retain their actual settings. Never issue a missing turn under
    # different settings and label it as part of the same inference.
    durable.request_settings_match = (
        {k: v for k, v in inference_provenance.items() if k != "configuration_fingerprint"}
        == {k: v for k, v in current_provenance.items() if k != "configuration_fingerprint"})
    common = {"clip_uid": job.clip_uid, "backend_provenance": inference_provenance,
              "postprocessing_version": "ra2va_group_durable_two_step_v1"}
    try:
        result = backend.reconcile(job, segment_ids=[s.segment_id for s in job.segments],
            transcribed_segment_ids=[s.segment_id for s in job.segments if s.asr_status == "transcribed"],
            allowed_entity_ids={s.entity_id for s in job.reference_subjects if s.kind == "entity"},
            allowed_reference_labels={s.subject_label for s in job.reference_subjects}
                                     | {p.picture_label for p in job.reference_images},
            auxiliary_audio_paths={kind: Path(path) for kind, path in stems.items()})
        values = {**common, "status": "ready", "annotation": result.annotation.model_dump(mode="json"),
            "raw_responses": list(result.raw_responses),
            "visual_raw_response": result.visual_raw_response, "speech_av_raw_response": result.speech_av_raw_response,
            "diagnostics": [d.model_dump(mode="json") for d in result.diagnostics],
            "deterministic_correction_counts": result.deterministic_correction_counts}
    except MimoBackendFailure as exc:
        values = {**common, "status": "failed", "failure_code": (
            "interrupted_request_unresolved" if durable.unresolved_turns else exc.code),
            "failure_reason": exc.reason,
            "failure_issues": [vars(issue) for issue in exc.issues],
            "annotation": exc.annotation.model_dump(mode="json") if exc.annotation else None,
            "raw_responses": list(exc.raw_responses), "visual_raw_response": exc.visual_raw_response,
            "speech_av_raw_response": exc.speech_av_raw_response,
            "diagnostics": [d.model_dump(mode="json") for d in exc.diagnostics]}
    values.update(durable.audit())
    atomic_json(Path(root) / "reconcile.json", values)
    return values
