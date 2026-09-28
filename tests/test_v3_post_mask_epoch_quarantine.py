"""Clip-local Post-Mask quarantine must not swallow infrastructure faults."""

import errno
import json
from pathlib import Path

import pytest
from PIL import UnidentifiedImageError
from pydantic import BaseModel, ValidationError

from r2v_data_v2.v3.post_mask_epoch_quarantine import ClipQuarantine
from r2v_data_v2.v3.post_mask_epoch_resources import EpochResourceError
from r2v_data_v2.v3.post_mask_epoch_state import PlanMismatchError


class _RequiredField(BaseModel):
    value: int


class _Storage:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.failures: list[dict[str, object]] = []

    def clip_dir(self, clip_uid: str) -> Path:
        return self.root / "clips" / clip_uid

    def append_failure(self, **kwargs: object) -> None:
        self.failures.append(kwargs)
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / "failures.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(kwargs) + "\n")


def test_quarantine_records_local_failure_once_and_preserves_first_stage(tmp_path):
    storage = _Storage(tmp_path)
    events: list[tuple[str, dict[str, object]]] = []
    quarantine = ClipQuarantine(emit=lambda event, **fields: events.append((event, fields)))
    missing = FileNotFoundError(2, "No such file", str(storage.clip_dir("bad") / "clip.json"))

    assert quarantine.record_if_local("shard-1", storage, "bad", "pair", missing)
    assert quarantine.record_if_local("shard-1", storage, "bad", "reference_edit", missing)
    assert quarantine.contains("shard-1", "bad")
    assert quarantine.count == 1
    assert len(storage.failures) == 1
    assert storage.failures[0]["stage"] == "pair"
    assert storage.failures[0]["clip_uid"] == "bad"
    assert len(events) == 1
    assert events[0][0] == "post_mask_clip_quarantined"
    assert events[0][1]["stage"] == "pair"
    assert "clip.json" in str(events[0][1]["reason"])


def test_quarantine_accepts_malformed_current_clip_only(tmp_path):
    storage = _Storage(tmp_path)
    quarantine = ClipQuarantine()
    malformed = ValidationError.from_exception_data(
        "_RequiredField", [{"type": "missing", "loc": ("value",), "input": {}}]
    )

    assert not quarantine.record_if_local("shard-1", storage, "bad", "pair", malformed)
    assert quarantine.record_if_local(
        "shard-1", storage, "bad", "pair", malformed,
        known_clip_artifact_read=True,
    )
    assert quarantine.count == 1


def test_quarantine_does_not_consume_global_or_authority_errors(tmp_path):
    storage = _Storage(tmp_path)
    quarantine = ClipQuarantine()
    cases = (
        PermissionError(13, "Permission denied", str(storage.clip_dir("bad") / "clip.json")),
        OSError(errno.EIO, "shared filesystem I/O failure", str(storage.clip_dir("bad") / "clip.json")),
        FileNotFoundError(2, "No such file", str(tmp_path / "run.json")),
        EpochResourceError("Qwen unavailable"),
        PlanMismatchError("frozen plan drift"),
        ValueError("policy mismatch"),
    )

    for error in cases:
        assert not quarantine.record_if_local("shard-1", storage, "bad", "pair", error)
    assert quarantine.count == 0
    assert storage.failures == []


def test_plain_value_error_requires_explicit_current_clip_read_context(tmp_path):
    storage = _Storage(tmp_path)
    quarantine = ClipQuarantine()
    invalid = ValueError("sampled frame dimensions mismatch")

    assert not quarantine.record_if_local("shard-1", storage, "bad", "pair", invalid)
    assert quarantine.record_if_local(
        "shard-1", storage, "bad", "pair", invalid,
        known_clip_artifact_read=True,
    )
    assert quarantine.count == 1


def test_unidentified_image_requires_explicit_current_clip_read_context(tmp_path):
    storage = _Storage(tmp_path)
    quarantine = ClipQuarantine()
    corrupt = UnidentifiedImageError("cannot identify image file")
    assert corrupt.filename is None

    assert not quarantine.record_if_local("shard-1", storage, "bad", "remove", corrupt)
    assert quarantine.record_if_local(
        "shard-1", storage, "bad", "remove", corrupt,
        known_clip_artifact_read=True,
    )
    assert quarantine.count == 1
    assert len(storage.failures) == 1


@pytest.mark.parametrize(
    "error",
    [PermissionError("permission denied"), OSError(errno.EIO, "shared filesystem I/O failure")],
)
def test_known_clip_read_still_propagates_infrastructure_io(tmp_path, error):
    storage = _Storage(tmp_path)
    quarantine = ClipQuarantine()

    assert not quarantine.record_if_local(
        "shard-1", storage, "bad", "remove", error,
        known_clip_artifact_read=True,
    )
    assert quarantine.count == 0
    assert storage.failures == []


def test_group_quarantine_restores_scope_but_not_unrecorded_missing_clip(tmp_path):
    storage = _Storage(tmp_path)
    group = tmp_path / "group"
    first = ClipQuarantine.for_group(
        group, {"shard-1": ("good", "bad")}
    )
    missing = FileNotFoundError(
        2, "No such file", str(storage.clip_dir("bad") / "clip.json")
    )
    assert first.record_if_local("shard-1", storage, "bad", "pair", missing)

    resumed = ClipQuarantine.for_group(
        group, {"shard-1": ("good",)}, storages={"shard-1": storage}
    )
    assert resumed.eligible == {"shard-1": ("good", "bad")}
    assert resumed.excluded_by_shard() == {"shard-1": ["bad"]}
    assert resumed.count == 1
    without_storages = ClipQuarantine.for_group(group, {"shard-1": ("good",)})
    assert without_storages.excluded_by_shard() == {"shard-1": ["bad"]}
    with pytest.raises(ValueError, match="eligible scope drifted"):
        ClipQuarantine.for_group(
            group, {"shard-1": ()}, storages={"shard-1": storage}
        )


def test_group_quarantine_restores_marker_without_reading_failure_log(
    tmp_path, monkeypatch
):
    storage = _Storage(tmp_path / "run")
    group = tmp_path / "group"
    first = ClipQuarantine.for_group(group, {"shard-1": ("good", "bad")})
    missing = FileNotFoundError(
        2, "No such file", str(storage.clip_dir("bad") / "clip.json")
    )
    assert first.record_if_local("shard-1", storage, "bad", "pair", missing)

    failure_path = storage.root / "failures.jsonl"
    failure_path.unlink()
    original_open = Path.open

    def guard_failure_log(path, *args, **kwargs):
        if path == failure_path:
            raise AssertionError("failures.jsonl must not be opened on resume")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guard_failure_log)
    resumed = ClipQuarantine.for_group(
        group, {"shard-1": ("good",)}, storages={"shard-1": storage}
    )

    assert resumed.eligible == {"shard-1": ("good", "bad")}
    assert resumed.excluded_by_shard() == {"shard-1": ["bad"]}
    assert resumed.count == 1


def test_group_quarantine_rejects_marker_outside_recorded_scope(tmp_path):
    storage = _Storage(tmp_path / "run")
    group = tmp_path / "group"
    first = ClipQuarantine.for_group(
        group, {"shard-1": ("good", "bad")}
    )
    missing = FileNotFoundError(
        2, "No such file", str(storage.clip_dir("bad") / "clip.json")
    )
    assert first.record_if_local("shard-1", storage, "bad", "pair", missing)
    marker_path = group / "composition" / "clip_quarantine.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["quarantined"].append(
        {
            "shard": "shard-1", "clip_uid": "not-eligible", "stage": "pair",
            "reason": "forged failure",
        }
    )
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid clip quarantine marker"):
        ClipQuarantine.for_group(
            group, {"shard-1": ("good",)}, storages={"shard-1": storage}
        )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    (
        ("schema", "wrong-schema", ValueError),
        ("quarantined", {}, TypeError),
    ),
)
def test_group_quarantine_keeps_marker_schema_and_type_validation(
    tmp_path, field, value, error
):
    group = tmp_path / "group"
    marker_path = group / "composition" / "clip_quarantine.json"
    marker_path.parent.mkdir(parents=True)
    marker = {
        "schema": "post_mask_clip_quarantine/1",
        "eligible_clip_uids_by_shard": {"shard-1": ["good"]},
        "quarantined": [],
    }
    marker[field] = value
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    with pytest.raises(error, match="invalid clip quarantine marker"):
        ClipQuarantine.for_group(group, {"shard-1": ("good",)})
