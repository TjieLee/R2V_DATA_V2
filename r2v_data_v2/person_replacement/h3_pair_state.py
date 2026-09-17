"""Pair-local durable commit markers. No cursor, model imports or production writes."""

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path

from r2v_data_v2.manifest import iter_source_records


def now():
    return datetime.now(UTC).isoformat()


def atomic_bytes(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def sync_directory(directory):
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, payload):
    atomic_bytes(path, (json.dumps(payload, ensure_ascii=False, indent=2)+"\n").encode())


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def select_shard(source, output, pair_id, pair_size):
    if pair_id < 0 or pair_size < 1:
        raise ValueError("pair-id >= 0 and pair-size >= 1 required")
    start = pair_id * pair_size
    cases = []
    for index, row in enumerate(islice(iter_source_records(source), start, start+pair_size), start):
        digest = hashlib.sha256(json.dumps(row,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        case_id = f"{index:09d}-{digest[:16]}"
        cases.append({"case_id":case_id, "source_index":index, "row":row, "row_sha256":digest,
                      "directory":str(output/f"shard-{pair_id:06d}"/case_id)})
    return cases


def partition(cases, worker):
    if worker not in (0,1):
        raise ValueError("Exactly two Qwen partitions")
    return cases[worker::2]


def phase(case):
    directory = Path(case["directory"])
    if (directory/"generation/manifest.json").is_file():
        return "done"
    if (directory/"preparation/prepared.json").is_file():
        return "generate"
    return "prepare"


def failure_count(case, stage):
    return len(list((Path(case["directory"])/"failures").glob(f"{stage}_attempt_*.json")))


def retry_limit(case, stage, maximum, *, retry_failed):
    failures = failure_count(case,stage)
    return failures + maximum if retry_failed and failures >= maximum else maximum


def eligible(case, stage, limit):
    return phase(case) == stage and failure_count(case,stage) < limit


def begin_attempt(case, stage, worker):
    directory = Path(case["directory"])/"attempts"
    directory.mkdir(parents=True, exist_ok=True)
    numbers = [int(p.stem.rsplit("_",1)[1]) for p in directory.glob(f"{stage}_attempt_*.json")]
    number = max(numbers,default=0)+1
    record = {"case_id":case["case_id"], "stage":stage, "attempt":number,
              "timestamp":now(), "worker":worker}
    atomic_json(directory/f"{stage}_attempt_{number:03d}.json",record)
    return record


def fail_attempt(case, attempt, exc):
    path = Path(case["directory"])/"failures"/f"{attempt['stage']}_attempt_{attempt['attempt']:03d}.json"
    if path.exists():
        raise FileExistsError(f"Failure history is immutable: {path}")
    atomic_json(path,{**attempt, "failed_at":now(), "error_type":type(exc).__name__, "error":str(exc)})


def publish_prepared(case, payload, texts):
    directory = Path(case["directory"])/"preparation"
    marker = directory/"prepared.json"
    if phase(case) != "prepare":
        raise FileExistsError("Case is already prepared/done")
    if payload["case_id"] != case["case_id"] or not payload["prompt"].strip():
        raise ValueError("Invalid preparation")
    for name, value in texts.items():
        if not isinstance(value,str) or not value.strip():
            raise ValueError(f"Missing preparation field: {name}")
        atomic_bytes(directory/f"{name}.txt",(value+"\n").encode())
    atomic_json(marker,payload)  # commit marker LAST


def publish_generated(case, temporary, manifest, validate):
    if phase(case) == "done":
        raise FileExistsError("Successful case must never be overwritten")
    validate(temporary)
    directory = Path(case["directory"])/"generation"
    directory.mkdir(parents=True,exist_ok=True)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary,directory/"raw.mp4")
    sync_directory(directory)
    atomic_json(directory/"manifest.json",manifest)  # commit marker LAST


def inventory(cases, limits):
    result = {"done":0, "prepared":0, "unprepared":0, "exhausted_prepare":0, "exhausted_generate":0}
    for case in cases:
        stage = phase(case)
        result[{"done":"done","prepare":"unprepared","generate":"prepared"}[stage]] += 1
        if stage != "done" and not eligible(case,stage,limits[case["case_id"]][stage]):
            result[f"exhausted_{stage}"] += 1
    return result
