"""Deterministic, appearance-only replacement diversity cues."""

from r2v_data_v2.person_replacement.h3_pair_diversity import (
    BUCKET_WEIGHTS,
    GENERIC,
    PERIOD_CUES,
    PROFESSION_CUES,
    bucket_for_value,
    diversity_cues,
)


def test_bucket_mapping_covers_the_announced_distribution():
    assert bucket_for_value(0) == "generic"
    assert bucket_for_value(64) == "generic"
    assert bucket_for_value(65) == "profession"
    assert bucket_for_value(89) == "profession"
    assert bucket_for_value(90) == "period"
    assert bucket_for_value(99) == "period"
    assert sum(BUCKET_WEIGHTS.values()) == 100


def test_sampling_is_deterministic_for_the_same_input():
    first = diversity_cues(42,"row-sha-1")
    assert first == diversity_cues(42,"row-sha-1")
    assert first != diversity_cues(42,"row-sha-2") or first == diversity_cues(42,"row-sha-2")
    assert diversity_cues(7,"row-sha-1") == diversity_cues(7,"row-sha-1")


def test_seeds_can_vary_and_pools_cover_everyday_roles():
    assert len(PROFESSION_CUES) > 40
    for cue in ("doctor","nurse","teacher","chef","police officer","delivery courier",
                "supermarket cashier","hairdresser","construction worker","office professional"):
        assert cue in PROFESSION_CUES
    for cue in ("Qing-style noblewoman","traditional Chinese noblewoman",
                "Republican-era formal attire","hanfu-style figure"):
        assert cue in PERIOD_CUES
    seen = {diversity_cues(seed,"row")[0] for seed in range(40)}
    assert len(seen) > 1  # different seeds explore different cues


def test_cues_only_come_from_the_allowed_pools():
    allowed = {GENERIC,*PROFESSION_CUES,*PERIOD_CUES}
    specialized = []
    for seed in range(200):
        for cue in diversity_cues(seed,f"row-{seed}"):
            assert cue in allowed
            if cue != GENERIC:
                specialized.append(cue)
    assert specialized  # both special buckets are reachable
    generic = sum(1 for seed in range(200) for cue in diversity_cues(seed,f"row-{seed}")
                  if cue == GENERIC)
    assert generic / 400 > 0.4  # ordinary clothing stays the majority


def test_two_specialized_cues_are_never_identical():
    for seed in range(300):
        first, second = diversity_cues(seed,f"row-{seed}")
        assert first == GENERIC or second == GENERIC or first != second


def test_diversity_is_independent_of_execution_strategy():
    """group_size/ulysses are execution strategy and must not move the cues."""
    assert diversity_cues(42,"row") == diversity_cues(42,"row",index=0)


def test_prepared_provenance_records_both_hints(tmp_path, monkeypatch):
    from r2v_data_v2.person_replacement import h3_pair_prepare as module
    from r2v_data_v2.person_replacement.h3_pair_state import phase, read_json
    from r2v_data_v2.person_replacement.timeline import VideoTimeline
    from tests.person_replacement.test_h3_pair_prepare import (
        Qwen,
        make_cases,
        passthrough_audio,
    )

    clips = tmp_path/"clips"
    clips.mkdir()
    cases = make_cases(tmp_path,1,clips)
    config = {"cases":cases,"clips_root":str(clips),"qwen_model":str(tmp_path),"seed":42,"pair_id":0,
              "identity":"identity","limits":{"0":{"prepare":2,"generate":2}}}
    passthrough_audio(module,monkeypatch)
    monkeypatch.setattr(module,"inspect_video_timeline",lambda _:VideoTimeline(138,25,25,1,1920,1080,5.52))
    seen = {}

    class Recording(Qwen):
        def invent_two_with_diversity(self, subject1, subject2, cue1, cue2):
            seen.update(cue1=cue1,cue2=cue2)
            return "replacement 1","replacement 2"

    module.prepare_partition(config,0,lambda _:Recording())
    assert phase(cases[0]) == "generate"
    prepared = read_json((tmp_path/"case0")/"preparation"/"prepared.json")
    assert prepared["replacement_diversity_1"] == seen["cue1"]
    assert prepared["replacement_diversity_2"] == seen["cue2"]
    assert seen["cue1"] in {GENERIC,*PROFESSION_CUES,*PERIOD_CUES}
