from pathlib import Path

from tests import test_h3_auk_speech_shadow as fixtures

ffmpeg = fixtures.ffmpeg
setup = fixtures.setup


def test_auk_replay_publishes_full_inventory(setup, tmp_path, ffmpeg):
    from r2v_data_v2.h3 import auk_speech_shadow as auk
    from r2v_data_v2.h3 import t2va_full_stems as stems
    from tests.test_h3_auk_speech_shadow import FakeBackend

    attempted = []

    def execute(stage_root, jobs, **kwargs):
        backend = FakeBackend(setup.model_configuration)
        result = {}
        for item in jobs:
            attempted.append(item["job_id"])
            raw = stage_root / (item["job_id"] + ".wav")
            raw.parent.mkdir(parents=True, exist_ok=True)
            response = backend.generate(auk.AukJob.model_validate(item["source"]), raw)
            result[item["job_id"]] = {
                "status": "ready",
                "result": {"raw_path": str(raw), "response": response},
            }
        return result

    stems.run_auk(
        setup,
        tmp_path / "work",
        ["0", "1"],
        eligible={"a", "c"},
        ffmpeg=ffmpeg,
        execute=execute,
    )
    _, records, _ = auk.load_auk_shadow(
        auk.auk_stage_root(Path(setup.audio_production_root), setup.shadow_run_id)
    )
    assert attempted == [uid for uid in setup.clip_uids if uid in {"a", "c"}]
    assert [r.clip_uid for r in records] == setup.clip_uids
    assert {r.clip_uid: r.status for r in records} == {
        "a": "ready",
        "b": "failed",
        "c": "ready",
    }
