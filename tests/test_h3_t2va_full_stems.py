from pathlib import Path

from tests import test_h3_auk_speech_shadow as fixtures

ffmpeg = fixtures.ffmpeg
setup = fixtures.setup


def test_sam_model_loaded_once_and_inventory_bound_per_request(tmp_path, monkeypatch):
    from r2v_data_v2.h3 import sam_audio_stem_shadow as sam
    from r2v_data_v2.h3 import t2va_full_stems as stems
    from tests.test_h3_sam_audio_stem_shadow import _canonical_fixture, _configuration

    manifest, _, _ = _canonical_fixture(tmp_path)
    config = _configuration(tmp_path)
    inventory = sam.build_sam_audio_stem_inventory(
        canonical_audio_manifest_path=manifest, model_configuration=config
    )
    loads = []

    class Backend:
        def __init__(self, configuration):
            self.configuration = configuration

        def _load(self):
            loads.append(self.configuration)

    monkeypatch.setattr(sam, "OfficialSAMAudioBackend", Backend)
    configuration = {"model": config.model_dump(mode="json")}
    with stems.SAMWorker(configuration) as worker:
        backend = worker.backend
        for name in ("a", "b", "c"):
            path = tmp_path / f"{name}.json"
            path.write_text(inventory.model_dump_json())
            worker.bind_request({**configuration, "inventory_path": str(path)})
            assert worker.inventory == inventory
            assert worker.backend is backend
        assert loads == [config]


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
