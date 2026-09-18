"""Deterministic model-job identity for the Post-Mask resource-epoch scheduler.

This module is the *semantic model-call boundary* of ``resource_epoch_v3``. It
answers exactly one question: "what is this model call, independent of when or
where it runs?" It deliberately contains no policy, no prompt, no threshold and
no accept/reject rule. Those live in the existing semantic modules and are
consumed here only as an opaque :func:`semantic_input_digest`.

A job identity binds every value that may legitimately change *what* the model
is asked to do:

    schema version, job type, resource type, canonical shard, clip,
    owner/entity/attribute target, attempt index, seed, semantic input digest,
    dependency result digests, model identity

It must never contain execution state: hostname, ``RANK``, ``WORLD_SIZE``,
physical GPU id, PID, runtime concurrency or wall-clock time. Two runs that
differ only in placement must produce byte-identical job identities, otherwise
resume could not reuse a receipt and the scheduler could not be restarted under
a different topology.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

# Bump only when the meaning of a job identity changes. A version bump is
# intentionally disruptive: every existing receipt becomes unreadable and its
# work is re-planned rather than silently reused under new semantics.
JOB_SCHEMA_VERSION = "post_mask_resource_epoch_v3/1"

RESOURCE_QWEN = "qwen"
RESOURCE_BOOGU = "boogu"
RESOURCE_SAM = "sam"
RESOURCE_TYPES = (RESOURCE_QWEN, RESOURCE_BOOGU, RESOURCE_SAM)

# Commit outcomes. Only ``completed`` may skip a later model call.
OUTCOME_COMPLETED = "completed"
OUTCOME_TERMINAL_REJECT = "terminal_reject"
OUTCOME_RETRYABLE_FAILED = "retryable_failed"

COMMITTED_OUTCOMES = (OUTCOME_COMPLETED, OUTCOME_TERMINAL_REJECT)


def canonical_json(value: Any) -> str:
    """Serialize ``value`` to a byte-stable JSON string.

    Sorting keys and removing optional whitespace keeps digests stable across
    processes, interpreters and dict insertion orders.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def semantic_input_digest(value: Any) -> str:
    """Digest semantic model inputs (prompts, image identities, thresholds...).

    The value is supplied by the existing semantic call sites, never rebuilt
    here, so a policy change automatically changes every dependent job identity.
    """
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _pairs(value: Mapping[str, str] | None) -> tuple[tuple[str, str], ...]:
    if not value:
        return ()
    return tuple(sorted((str(k), str(v)) for k, v in value.items()))


@dataclass(frozen=True, order=True)
class ModelJob:
    """One immutable, addressable model call.

    ``target`` carries the semantic address inside a clip, for example
    ``(("owner", "ent-3"), ("attribute", "upper_body"))``. ``attempt_index`` and
    ``seed`` are the *conditional* round identifiers: a second-seed or
    candidate-2 call is a different job, but it is only ever planned once the
    existing policy has unlocked it.
    """

    job_type: str
    resource: str
    canonical_shard: str
    clip_uid: str
    input_digest: str
    model_identity: str
    target: tuple[tuple[str, str], ...] = ()
    attempt_index: int = 0
    seed: int | None = None
    dependency_digests: tuple[tuple[str, str], ...] = ()
    schema_version: str = JOB_SCHEMA_VERSION
    #: Free-form, execution-only ordering hint. Never part of ``identity()``.
    label: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        if self.resource not in RESOURCE_TYPES:
            raise ValueError(f"unknown resource type: {self.resource}")
        if not self.job_type:
            raise ValueError("job_type is required")
        if not self.canonical_shard:
            raise ValueError("canonical_shard is required")
        if not self.clip_uid:
            raise ValueError("clip_uid is required")
        if not self.model_identity:
            raise ValueError("model_identity is required")

    @classmethod
    def create(
        cls,
        *,
        job_type: str,
        resource: str,
        canonical_shard: str,
        clip_uid: str,
        semantic_inputs: Any,
        model_identity: str,
        target: Mapping[str, str] | None = None,
        attempt_index: int = 0,
        seed: int | None = None,
        dependencies: Mapping[str, str] | None = None,
        schema_version: str = JOB_SCHEMA_VERSION,
        label: str = "",
    ) -> ModelJob:
        return cls(
            job_type=job_type,
            resource=resource,
            canonical_shard=canonical_shard,
            clip_uid=clip_uid,
            input_digest=semantic_input_digest(semantic_inputs),
            model_identity=model_identity,
            target=_pairs(target),
            attempt_index=attempt_index,
            seed=seed,
            dependency_digests=_pairs(dependencies),
            schema_version=schema_version,
            label=label,
        )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "job_type": self.job_type,
            "resource": self.resource,
            "canonical_shard": self.canonical_shard,
            "clip_uid": self.clip_uid,
            "target": [list(pair) for pair in self.target],
            "attempt_index": self.attempt_index,
            "seed": self.seed,
            "input_digest": self.input_digest,
            "dependency_digests": [list(pair) for pair in self.dependency_digests],
            "model_identity": self.model_identity,
        }

    def identity(self) -> str:
        """Stable digest that excludes every execution-only field."""
        return hashlib.sha256(
            canonical_json(self.identity_payload()).encode("utf-8")
        ).hexdigest()

    def job_id(self) -> str:
        return self.identity()[:16]

    def plan_record(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["job_id"] = self.job_id()
        payload["job_identity"] = self.identity()
        return payload
