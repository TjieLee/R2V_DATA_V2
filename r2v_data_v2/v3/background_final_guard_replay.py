from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from r2v_data_v2.reconciliation import write_json_atomic
from r2v_data_v2.v3 import config as config_module
from r2v_data_v2.v3.background_final_guard import (
    FinalBackgroundJudge,
    FinalBackgroundJudgeFailure,
    QwenFinalBackgroundJudge,
    load_final_background_image,
)
from r2v_data_v2.v3.config import QwenServiceConfig, V3Config
from r2v_data_v2.v3.storage import RunStorage


def _validated_output_path(output_path: Path, run_root: Path) -> Path:
    output = output_path.expanduser().resolve(strict=False)
    root = run_root.expanduser().resolve(strict=True)
    allowed = config_module.ALLOWED_WRITABLE_ROOT.resolve(strict=False)
    if allowed not in output.parents:
        raise ValueError("replay output must be inside the writable data root")
    if output == root or root in output.parents:
        raise ValueError("replay output must be outside the source run_root")
    if output.exists() and output.is_dir():
        raise ValueError("replay output must be a file path")
    return output


def _write_jsonl_atomic(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _judge_config(
    config: V3Config,
    *,
    base_url: str | None,
    model: str | None,
    api_key: str | None,
) -> QwenServiceConfig:
    service = (
        config.qwen.background_final_judge
        or config.qwen.background_remove_judge
    )
    if service is None:
        raise ValueError("replay requires a configured Qwen background judge")
    return replace(
        service,
        base_url=service.base_url if base_url is None else base_url,
        model=service.model if model is None else model,
        api_key=service.api_key if api_key is None else api_key,
    )


def run_background_final_guard_replay(
    config: V3Config,
    *,
    run_root: Path,
    output_path: Path,
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    judge: FinalBackgroundJudge | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, object]:
    root = run_root.expanduser().resolve(strict=True)
    output = _validated_output_path(output_path, root)
    summary_path = Path(f"{output}.summary.json")
    service = _judge_config(
        config,
        base_url=base_url,
        model=model,
        api_key=api_key,
    )
    replay_config = replace(
        config,
        run_root=root,
        pair=replace(config.pair, background_final_guard_mode="qwen_v1"),
        qwen=replace(config.qwen, background_final_judge=service),
    )
    replay_config.validate()
    storage = RunStorage(replay_config)
    storage.read_run()
    active_judge = judge
    owned_judge: QwenFinalBackgroundJudge | None = None
    records: list[dict[str, object]] = []
    started = clock()
    try:
        for clip in storage.iter_clips():
            if (
                clip.annotation is None
                or clip.annotation.status != "ready"
                or clip.annotation.background is None
                or clip.pairing is None
                or clip.pairing.status != "ready"
                or clip.pairing.background_token is None
            ):
                continue
            background = clip.references.background
            if background is None or background.status not in {
                "clean_raw",
                "ready_removed",
            }:
                raise ValueError("bound replay background is not ready")
            image = load_final_background_image(
                storage,
                clip_uid=clip.clip_uid,
                background=background,
            )
            if active_judge is None:
                owned_judge = QwenFinalBackgroundJudge(service)
                active_judge = owned_judge
            case_started = clock()
            try:
                attempt = active_judge.review(
                    image=image,
                    background_phrase=clip.annotation.background.phrase,
                    background_grounding_prompt=(
                        clip.annotation.background.grounding_prompt
                    ),
                    background_status=background.status,
                )
            except FinalBackgroundJudgeFailure as exc:
                records.append(
                    {
                        "clip_uid": clip.clip_uid,
                        "background_status": background.status,
                        "background_phrase": clip.annotation.background.phrase,
                        "status": "failed",
                        "verdict": None,
                        "reason": str(exc),
                        "duration_seconds": clock() - case_started,
                    }
                )
                continue
            records.append(
                {
                    "clip_uid": clip.clip_uid,
                    "background_status": background.status,
                    "background_phrase": clip.annotation.background.phrase,
                    "status": "succeeded",
                    **attempt.review.model_dump(mode="json"),
                    "duration_seconds": clock() - case_started,
                }
            )
    finally:
        if owned_judge is not None:
            owned_judge.close()

    summary: dict[str, object] = {
        "model": service.model,
        "candidate_background_count": len(records),
        "accepted": sum(record.get("verdict") == "accept" for record in records),
        "rejected": sum(record.get("verdict") == "reject" for record in records),
        "failed": sum(record["status"] == "failed" for record in records),
        "clean_raw_count": sum(
            record["background_status"] == "clean_raw" for record in records
        ),
        "ready_removed_count": sum(
            record["background_status"] == "ready_removed" for record in records
        ),
        "total_seconds": clock() - started,
    }
    _write_jsonl_atomic(output, records)
    write_json_atomic(summary_path, summary)
    return summary
