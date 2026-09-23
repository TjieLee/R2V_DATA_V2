import json
from types import SimpleNamespace

import pytest

from tools import plan_h3_t2va_mimo_ab as ab


def test_plan_uses_one_ordered_case_manifest_and_existing_serve_builder(tmp_path):
    case = tmp_path / "cases.json"
    case.write_text(json.dumps({"clip_uids": ["b", "a"]}))
    plan = ab.build_plan(
        audio_production_root=tmp_path / "audio",
        audio_shadow_run_id="resolved",
        case_manifest=case,
        base_url="http://127.0.0.1:8092/v1",
        model="mimo-v2.6-flash-rl",
        media_root=tmp_path,
        sglang_bin="/runtime/sglang",
        checkpoint="/models/v26",
        run_prefix="v26-random20",
        allow_unverified=True,
    )
    assert plan["clip_uids"] == ["b", "a"]
    assert (
        plan["endpoint"][plan["endpoint"].index("--speculative-algorithm") + 1]
        == "EAGLE"
    )
    for name in ("multi_t2va", "multi_ta2va", "single_t2va", "single_ta2va"):
        assert "mimo-v2.6-flash-rl" in plan[name]
    for name in ("multi_t2va", "single_t2va"):
        command = plan[name]
        assert command[command.index("--case-manifest") + 1] == str(case)
        assert command[command.index("--audio-shadow-run-id") + 1] == "resolved"
        assert "--existing-audio-cases" in command
        assert all(
            flag not in command
            for flag in (
                "--shot-manifest", "--clips-root", "--source-videos-root",
                "--shot-index-root",
            )
        )
    assert plan["multi_t2va"][-2:] == ["--call-mode", "multi"]
    assert plan["single_t2va"][-2:] == ["--call-mode", "single"]
    assert plan["multi_ta2va"][3].endswith("/v26-random20-multi-t2va")
    assert plan["single_ta2va"][3].endswith("/v26-random20-single-t2va")
    assert "--allow-unverified" in plan["multi_ta2va"]
    assert "--allow-unverified" in plan["single_ta2va"]


def test_report_preserves_case_order_and_sums_actual_calls(tmp_path, monkeypatch):
    case = tmp_path / "cases.json"
    case.write_text(json.dumps({"clip_uids": ["b", "a"]}))
    jobs = [SimpleNamespace(clip_uid=uid) for uid in ("b", "a")]
    selection = SimpleNamespace(
        case_manifest_path=str(case.resolve()),
        case_manifest_sha256=ab.sha256_file(case),
    )
    multi = SimpleNamespace(
        clip_uids=["b", "a"],
        jobs=jobs,
        shot_selection=selection,
        source_hashes={"upstream": "one"},
        audio_production_root="/audio",
        audio_shadow_run_id="resolved",
        inventory_fingerprint="multi",
        t2va_run_id="multi",
        backend=SimpleNamespace(model="mimo-v2.6-flash-rl", schema_version=".7"),
    )
    single = SimpleNamespace(
        **{
            **multi.__dict__,
            "inventory_fingerprint": "single",
            "t2va_run_id": "single",
            "backend": SimpleNamespace(
                model="mimo-v2.6-flash-rl", schema_version=".9", call_mode="single"
            ),
        }
    )
    records = {
        "multi": [
            SimpleNamespace(clip_uid="b", status="ready", model_call_count=2),
            SimpleNamespace(clip_uid="a", status="failed", model_call_count=1),
        ],
        "single": [
            SimpleNamespace(clip_uid="b", status="ready", model_call_count=1),
            SimpleNamespace(clip_uid="a", status="failed", model_call_count=1),
        ],
    }
    products = {
        "multi": [
            SimpleNamespace(
                clip_uid="b",
                variant="full_audio_reuse",
                status="ready",
                model_call_count=0,
            ),
            SimpleNamespace(
                clip_uid="b",
                variant="target_speech_reuse",
                status="ready",
                model_call_count=1,
            ),
        ],
        "single": [
            SimpleNamespace(
                clip_uid="b",
                variant="full_audio_reuse",
                status="ready",
                model_call_count=0,
            ),
            SimpleNamespace(
                clip_uid="b",
                variant="target_speech_reuse",
                status="ready",
                model_call_count=0,
            ),
        ],
    }
    monkeypatch.setattr(
        ab,
        "_load_t2va",
        lambda root: (multi if root.name == "multi" else single, records[root.name]),
    )
    monkeypatch.setattr(
        ab,
        "_load_ta2va",
        lambda root, inventory, t2va_root: products[root.name.split("-")[0]],
    )
    result = ab.build_report(
        case_manifest=case,
        multi_t2va_root=tmp_path / "multi",
        multi_ta2va_root=tmp_path / "multi-ta",
        single_t2va_root=tmp_path / "single",
        single_ta2va_root=tmp_path / "single-ta",
    )
    assert [row["clip_uid"] for row in result["clips"]] == ["b", "a"]
    assert result["clips"][0]["multi"]["t2va_model_call_count"] == 2
    assert result["clips"][0]["multi"]["ta2va_model_call_count"] == 1
    assert result["clips"][0]["single"]["t2va_model_call_count"] == 1
    assert result["clips"][0]["single"]["ta2va_model_call_count"] == 0
    assert result["totals"]["multi"]["model_call_count"] == 4
    assert result["totals"]["single"]["model_call_count"] == 2
    assert result["clips"][1]["multi"]["core_path"] is None
    assert result["clips"][0]["multi"]["ta2va_prompt_paths"]
    assert "score" not in result
    single.jobs = [SimpleNamespace(clip_uid="different")]
    with pytest.raises(ValueError, match="upstream"):
        ab.build_report(
            case_manifest=case,
            multi_t2va_root=tmp_path / "multi",
            multi_ta2va_root=tmp_path / "multi-ta",
            single_t2va_root=tmp_path / "single",
            single_ta2va_root=tmp_path / "single-ta",
        )


def test_report_reads_existing_v25_baseline_without_rerun(tmp_path, monkeypatch):
    case = tmp_path / "cases.json"
    case.write_text(json.dumps({"clip_uids": ["b", "a"]}))
    baseline = tmp_path / "v25"
    baseline.mkdir()
    (baseline / "records.jsonl").write_text(
        json.dumps({"clip_uid": "a", "status": "failed", "raw_responses": []})
        + "\n"
        + json.dumps({"clip_uid": "b", "status": "ready", "raw_responses": ["raw"]})
        + "\n"
    )
    (baseline / "prompts").mkdir()
    (baseline / "prompts/b.txt").write_text("frozen output")
    selection = SimpleNamespace(
        case_manifest_path=str(case.resolve()),
        case_manifest_sha256=ab.sha256_file(case),
    )
    inventory = SimpleNamespace(
        clip_uids=["b", "a"],
        jobs=[SimpleNamespace(clip_uid=uid) for uid in ("b", "a")],
        shot_selection=selection,
        source_hashes={"published": "hash"},
        audio_production_root="/audio",
        audio_shadow_run_id="resolved",
        backend=SimpleNamespace(model="v26", call_mode="multi"),
    )

    def load(root):
        mode = root.name
        backend = SimpleNamespace(model="v26", call_mode=mode)
        return SimpleNamespace(**{**inventory.__dict__, "backend": backend}), [
            SimpleNamespace(clip_uid=uid, status="ready", model_call_count=1)
            for uid in ("b", "a")
        ]

    monkeypatch.setattr(ab, "_load_t2va", load)
    monkeypatch.setattr(ab, "_load_ta2va", lambda *args: [])
    report = ab.build_report(
        case_manifest=case,
        multi_t2va_root=tmp_path / "multi",
        multi_ta2va_root=tmp_path / "multi-ta",
        single_t2va_root=tmp_path / "single",
        single_ta2va_root=tmp_path / "single-ta",
        v25_baseline_root=baseline,
    )
    assert [row["clip_uid"] for row in report["clips"]] == ["b", "a"]
    assert [row["v25"]["status"] for row in report["clips"]] == ["ready", "failed"]
    assert report["clips"][0]["v25"]["model_output_path"] == str(
        baseline / "records.jsonl"
    )
    assert report["clips"][0]["v25"]["prompt_path"] == str(
        baseline / "prompts/b.txt"
    )
    assert report["clips"][1]["v25"]["prompt_path"] is None
    assert report["clips"][0]["v26_multi"] == report["clips"][0]["multi"]
    assert report["clips"][0]["v26_single"] == report["clips"][0]["single"]


def test_report_rejects_different_ordered_population(tmp_path, monkeypatch):
    case = tmp_path / "cases.json"
    case.write_text(json.dumps({"clip_uids": ["a", "b"]}))
    selection = SimpleNamespace(
        case_manifest_path=str(case.resolve()),
        case_manifest_sha256=ab.sha256_file(case),
    )

    def load(root):
        uids = ["a", "b"] if root.name == "multi" else ["b", "a"]
        return SimpleNamespace(
            clip_uids=uids,
            jobs=[],
            shot_selection=selection,
            source_hashes={},
            audio_production_root="/audio",
            audio_shadow_run_id="resolved",
            inventory_fingerprint=root.name,
            backend=SimpleNamespace(model="v26", schema_version=".7"),
        ), []

    monkeypatch.setattr(ab, "_load_t2va", load)
    monkeypatch.setattr(ab, "_load_ta2va", lambda *args: [])
    with pytest.raises(ValueError, match="ordered|case manifest"):
        ab.build_report(
            case_manifest=case,
            multi_t2va_root=tmp_path / "multi",
            multi_ta2va_root=tmp_path / "multi-ta",
            single_t2va_root=tmp_path / "single",
            single_ta2va_root=tmp_path / "single-ta",
        )


def test_ab_report_cli_routes_explicit_roots(tmp_path, monkeypatch):
    monkeypatch.setattr(ab, "build_report", lambda **kwargs: kwargs)
    args = ["report", "--case-manifest", str(tmp_path / "cases.json")]
    for name in (
        "multi-t2va-root", "multi-ta2va-root",
        "single-t2va-root", "single-ta2va-root",
    ):
        args.extend((f"--{name}", str(tmp_path / name)))
    parsed = ab.main(args)
    assert parsed["case_manifest"] == tmp_path / "cases.json"
    assert parsed["single_ta2va_root"] == tmp_path / "single-ta2va-root"
