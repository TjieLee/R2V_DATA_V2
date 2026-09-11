import json
import os
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from r2v_data_v2.h3 import ta2va_shadow as ta
from r2v_data_v2.h3.audio_reuse import read_canonical_stem_pcm16, read_reuse_asset_pcm16
from r2v_data_v2.h3.mimo25_backend import SPEAKER_PROFILE_SYSTEM_PROMPT
from r2v_data_v2.h3.qwen38_h3_recaption import (
    RecaptionAudioContract,
    _canonical_audio_definition,
    _canonical_audio_retention,
)
from r2v_data_v2.h3.schemas import SchemaModel
from r2v_data_v2.h3.t2va_shadow import render_t2va_prompt
from r2v_data_v2.h3.ta2va_qa import build_ta2va_qa
from tests import test_h3_t2va_shadow as shared

job = shared.job
setup = shared.setup
ffmpeg = shared.ffmpeg
finalized = shared.finalized


class ProfileClient:
    def __init__(self, *, fail=False, identity=False, wrong=False):
        self.calls = []
        self.fail = fail
        self.identity = identity
        self.wrong = wrong
        self.chat = SimpleNamespace(completions=self)

    def create(self, **request):
        self.calls.append(request)
        if self.fail:
            raise RuntimeError("synthetic profile failure")
        text = request["messages"][1]["content"][0]["text"]
        targets = json.loads(text.split("\nRESPONSE SCHEMA:")[0].split("\n", 1)[1])
        profiles = [
            {
                "speaker_group": "S99" if self.wrong else t["speaker_group"],
                "voice_characteristics": "A teacher's voice."
                if self.identity
                else "A low resonant register with measured cadence and clear articulation.",
            }
            for t in targets
        ]
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content=json.dumps({"speaker_voice_profiles": profiles}),
                        reasoning_content=None,
                    ),
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=20,
                completion_tokens=20,
                total_tokens=40,
                prompt_tokens_details={"audio_tokens": 20},
                completion_tokens_details={"reasoning_tokens": 0},
            ),
        )


@pytest.fixture
def playable_manifest(monkeypatch, ffmpeg):
    original = shared.shot_manifest

    def write(tmp_path):
        manifest = original(tmp_path)
        for line in manifest.read_text().splitlines():
            path = json.loads(line)["video_path"]
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc=size=640x360:rate=4:duration=1",
                    "-pix_fmt",
                    "yuv420p",
                    path,
                ],
                check=True,
                capture_output=True,
            )
        return manifest

    monkeypatch.setattr(shared, "shot_manifest", write)


@pytest.fixture
def source(playable_manifest, finalized, tmp_path, request):
    inventory = shared.build(finalized, tmp_path)
    drafts = []
    for job in inventory.jobs:
        if job.upstream_failure:
            continue
        draft = shared.draft_for(job)
        if getattr(request, "param", None) == "unsafe":
            draft.speaker_assignments[0].primary_speaker_ownership_resolved = False
        drafts.append(draft.model_dump_json())
    client = shared.Client(
        drafts,
        audio_response=json.dumps(
            {
                "overall_soundscape": "Light footsteps are audible.",
                "non_diegetic_music": "A gentle piano melody continues.",
            }
        ),
    )
    shared.t2va.run_t2va_shadow(
        inventory, shared.T2VAMimoBackend(shared.config(tmp_path), client=client)
    )
    return shared.t2va.t2va_root(finalized[0], "t2va-test")


def run(source, tmp_path, name="ta", **client_args):
    client = ProfileClient(**client_args)
    backend = ta.TA2VAProfileBackend(shared.config(tmp_path), client=client)
    root = ta.run_ta2va_shadow(source, name, backend, allow_unverified=True)
    return root, ta.read_rows(root / "records.jsonl", ta.TA2VAProduct), client


@pytest.mark.parametrize(
    "summary",
    [
        "<Audio 1> is supplied.",
        "<Picture 1>",
        "<Subject 1>",
        "<Video 1>",
        "(S1) speaks.",
        "S1 speaks.",
        "<d>Hello</d>",
        "summary: People converse.",
        "[reference generation] People converse.",
        "An audio reuse example.",
    ],
)
def test_internal_summary_rejects_pipeline_syntax(job, summary):
    draft = shared.draft_for(job).model_dump()
    draft["summary"] = summary
    with pytest.raises(ValueError):
        shared.t2va.T2VAMimoDraft.model_validate(draft)


def test_summary_asr_hidden_from_public_renderer(job):
    draft = shared.draft_for(job)
    core = shared.materialize(job, draft)
    public = render_t2va_prompt(core)
    assert draft.summary not in public
    assert public.startswith("integrated_multimodal_description:")
    assert "\noverall_soundscape:" in public and "\nnon_diegetic_music:" in public
    assert "\nsummary:" not in public
    draft.summary = job.speech_facts[0].text
    with pytest.raises(ValueError, match="copy authoritative ASR"):
        shared.materialize(job, draft)


class Sample(SchemaModel):
    segment_id: str
    source_start_sample: int
    source_end_sample: int
    source_sample_rate_hz: int
    start_time: float
    end_time: float
    source_audio_path: str
    source_audio_sha256: str


def samples(job, rate=32000):
    return [
        Sample(
            segment_id=f.segment_id,
            start_time=f.start_time,
            end_time=f.end_time,
            source_start_sample=round(f.start_time * rate),
            source_end_sample=round(f.end_time * rate),
            source_sample_rate_hz=rate,
            source_audio_path=job.audio_evidence.speech_path,
            source_audio_sha256=job.audio_evidence.speech_sha256,
        )
        for f in job.speech_facts
    ]


@pytest.mark.parametrize("rate", [16000, 32000, 44100])
def test_sample_mapping_and_frozen_sx(job, rate):
    core = shared.materialize(job, shared.draft_for(job))
    rows = samples(job, rate)
    groups, warnings = ta.sample_ranges(job, core, rows, 160000, 160000)
    assert list(groups) == ["S1", "S2"]
    assert not warnings
    for row, group in zip(rows, groups.values(), strict=True):
        assert group[0].target_start_sample == round(
            Fraction(row.source_start_sample * 32000, rate)
        )
        assert group[0].target_end_sample == round(
            Fraction(row.source_end_sample * 32000, rate)
        )


def test_same_sx_keeps_multiple_separated_intervals(job):
    job.speech_facts[1].source_speaker_cluster = job.speech_facts[
        0
    ].source_speaker_cluster
    core = shared.materialize(job, shared.draft_for(job))
    groups, _ = ta.sample_ranges(job, core, samples(job), 160000, 160000)
    assert list(groups) == ["S1"]
    assert [(r.target_start_sample, r.target_end_sample) for r in groups["S1"]] == [
        (32000, 48000),
        (64000, 80000),
    ]
    pcm = np.full((160000, 2), 300, dtype=np.int16)
    result = ta.speech_track(pcm, 160000, groups["S1"])
    assert np.all(result[32000:48000] == 300)
    assert np.all(result[64000:80000] == 300)
    assert not np.any(result[:32000])
    assert not np.any(result[48000:64000])
    assert not np.any(result[80000:])


@pytest.mark.parametrize(
    "resolved,composition,present,relation,kind,reusable",
    [
        (True, "single_speaker", False, "none", None, True),
        (False, "single_speaker", False, "none", None, False),
        (True, "same_speaker_nonlexical", True, "same_speaker", "laughter", True),
        (True, "same_speaker_nonlexical", True, "same_speaker", "speech", False),
        (True, "secondary_non_speech_vocalization", True, "different_speaker", "laughter", False),
        (True, "overlapping_secondary_speech", True, "different_speaker", "speech", False),
        (True, "sequential_multi_speaker_speech", True, "different_speaker", "speech", False),
        (True, "uncertain", False, "none", None, False),
    ],
)
def test_frozen_ownership_blocks_entire_sx(
    job, resolved, composition, present, relation, kind, reusable
):
    from r2v_data_v2.h3.speaker_ownership import speaker_ownership_reasons

    job.speech_facts[1].source_speaker_cluster = job.speech_facts[0].source_speaker_cluster
    draft = shared.draft_for(job)
    assignment = draft.speaker_assignments[0]
    values = assignment.model_dump()
    values.update(
        primary_speaker_ownership_resolved=resolved,
        vocal_composition=composition,
        secondary_vocal_activity={"present": present, "speaker_relation": relation, "kind": kind},
    )
    draft.speaker_assignments[0] = type(assignment).model_validate(values)
    before = render_t2va_prompt(shared.materialize(job, shared.draft_for(job)))
    core = shared.materialize(job, draft)
    assert render_t2va_prompt(core) == before
    assert [a.speaker_id for a in core.speaker_assignments] == ["S1", "S1"]
    assert "entity_id" not in values and "subject_label" not in values
    groups, warnings = ta.sample_ranges(job, core, samples(job), 160000, 160000)
    assert bool(groups) is reusable
    reasons = speaker_ownership_reasons(draft.speaker_assignments[0])
    assert warnings == [f"S1:{job.speech_facts[0].segment_id}:{r}" for r in reasons]
    if not reusable:
        assert "S1" not in groups  # The otherwise-safe second segment cannot rescue this Sx.


@pytest.mark.parametrize("source", ["unsafe"], indirect=True)
def test_unsafe_sx_no_speech_product_and_summary_exclusions(source, tmp_path):
    root, products, client = run(source, tmp_path)
    assert not client.calls
    assert all(p.variant == "full_audio_reuse" and p.status == "ready" for p in products)
    summary = json.loads((root / "summary.json").read_text())
    assert summary["model_call_count"] == 0
    assert len(summary["reuse_exclusions"]) == 2
    assert all(w.endswith(":unresolved")
               for rows in summary["reuse_exclusions"].values() for w in rows)
    assert sum(summary["warning_counts"].values()) == 2


@pytest.mark.parametrize("music", ["N/A", "n/a", "Unknown", "unknown", "", "  ", " N/a "])
def test_music_absence_matches_ra2va(music):
    assert not ta._music_present(music)
    assert not ta.select_speakers({"S1": []}, ta._music_present(music))[1]


def test_positive_music_gate():
    assert ta._music_present(" A piano melody. ")
    assert ta.select_speakers({"S1": []}, ta._music_present("A piano melody."))[1]


def test_unsafe_nontranscribed_segment_also_blocks_sx(job):
    job.speech_facts[0].text = job.speech_facts[0].language = None
    job.speech_facts[1].source_speaker_cluster = job.speech_facts[0].source_speaker_cluster
    draft = shared.draft_for(job)
    draft.speaker_assignments[0].vocal_composition = "uncertain"
    core = shared.materialize(job, draft)
    groups, warnings = ta.sample_ranges(job, core, samples(job), 160000, 160000)
    assert not groups
    assert warnings == [f"S1:{job.speech_facts[0].segment_id}:non_single_speaker"]


def test_ownership_evidence_required_and_not_public(job):
    from r2v_data_v2.h3.t2va_mimo_backend import T2VA_SYSTEM_PROMPT

    model = shared.t2va.T2VASpeakerAssignment
    fields = {
        "primary_speaker_ownership_resolved", "vocal_composition", "secondary_vocal_activity",
    }
    assert fields <= set(model.model_json_schema()["required"])
    values = shared.draft_for(job).speaker_assignments[0].model_dump()
    for field in fields:
        with pytest.raises(ValueError):
            model.model_validate({k: v for k, v in values.items() if k != field})
        assert field in T2VA_SYSTEM_PROMPT
    assert "eligibility evidence only" in T2VA_SYSTEM_PROMPT
    assert "do not change Sx, speech presentation, boundaries or ASR" in T2VA_SYSTEM_PROMPT
    core = shared.materialize(job, shared.draft_for(job))
    public = render_t2va_prompt(core)
    assert all(field not in public for field in fields)
    assert core.schema_version == "r2v.h3.no_reference_av_core.6"


def test_two_sx_tracks_are_separate(job):
    core = shared.materialize(job, shared.draft_for(job))
    groups, _ = ta.sample_ranges(job, core, samples(job), 160000, 160000)
    pcm = np.full((160000, 2), 321, dtype=np.int16)
    tracks = {sx: ta.speech_track(pcm, 160000, ranges) for sx, ranges in groups.items()}
    assert np.all(tracks["S1"][32000:48000] == 321)
    assert not np.any(tracks["S1"][64000:80000])
    assert np.all(tracks["S2"][64000:80000] == 321)
    assert not np.any(tracks["S2"][32000:48000])


def test_no_transcribed_speech_has_no_reuse_track(job):
    for f in job.speech_facts:
        f.text = f.language = None
    core = shared.materialize(job, shared.draft_for(job))
    groups, _ = ta.sample_ranges(job, core, samples(job), 160000, 160000)
    assert groups == {}


def test_cross_speaker_overlap_not_copied(job):
    job.speech_facts[1].start_time = 1.25
    core = shared.materialize(job, shared.draft_for(job))
    groups, warnings = ta.sample_ranges(job, core, samples(job), 160000, 160000)
    assert groups == {}
    assert any("cross_speaker_overlap" in w for w in warnings)


@pytest.mark.parametrize(
    "count,music,expected",
    [(1, False, False), (2, True, True), (3, True, False), (4, True, False)],
)
def test_capacity(count, music, expected):
    groups = {f"S{i}": [] for i in range(1, count + 1)}
    selected, included, warnings = ta.select_speakers(groups, music)
    assert selected == list(groups)[:3]
    assert included == expected
    assert ("music:omitted_audio_capacity" in warnings) == (music and count >= 3)
    assert ("S4:omitted_audio_capacity" in warnings) == (count == 4)


def audio(kind="speaker_speech_reuse", index=1, sx="S1", traits=None):
    return RecaptionAudioContract(
        audio_index=index,
        audio_label=f"<Audio {index}>",
        kind=kind,
        path="/test.flac",
        sha256="a" * 64,
        speaker_id=sx if kind == "speaker_speech_reuse" else None,
        voice_characteristics=traits,
        retention_marker="fully_copy"
        if kind == "full_audio_reuse"
        else "partially_copy",
    )


def test_frozen_ra2va_null_profile_strings_unchanged():
    ref = audio()
    assert _canonical_audio_definition(ref) == (
        "<Audio 1> is the synchronized speech track for (S1), containing the supplied "
        "speaker's speech at its original timeline positions."
    )
    assert _canonical_audio_retention(ref) == (
        "<Audio 1>: partially_copy - the supplied speech signal is copied directly at its "
        "original timeline positions while the target's other audio layers remain separate."
    )
    profiled = audio(traits="a dry, low voice")
    assert _canonical_audio_definition(profiled).endswith("featuring a dry, low voice.")
    assert _canonical_audio_retention(profiled) == _canonical_audio_retention(ref)


@pytest.mark.parametrize("value", [
    "A Chinese speaker.", "A teacher's voice.", "The speaker named Li.",
    "A confident personality.", "<Subject 1>", "(S1)", "", None,
])
def test_profiles_reject_frozen_identity_claims_and_null(value):
    with pytest.raises(ValueError):
        ta.TA2VAProfile(speaker_group="S1", voice_characteristics=value)


@pytest.mark.parametrize("variant", ["full_audio_reuse", "target_speech_reuse"])
def test_six_sections_preserve_prose_and_exact_dialogue(job, variant):
    core = shared.materialize(job, shared.draft_for(job))
    core.non_diegetic_music = "A piano melody."
    refs = (
        [audio("full_audio_reuse")]
        if variant == "full_audio_reuse"
        else [
            audio(traits="a low voice"),
            audio(index=2, sx="S2", traits="a bright voice"),
            audio("music_reuse", index=3),
        ]
    )
    result = ta.render_product(core, variant, refs)
    headers = [line for line in result.splitlines() if line.endswith(":")]
    assert headers == [
        "subject_definitions:",
        "summary:",
        "retention_analysis:",
        "detailed_description:",
        "overall_soundscape:",
        "non_diegetic_music:",
    ]
    assert not any(label in result for label in ("<Picture", "<Subject", "<Video"))
    assert "[reference generation + audio reuse] " + core.summary in result
    caption = result.split("detailed_description:\n", 1)[1].split(
        "\n\noverall_soundscape:", 1
    )[0]
    if variant == "full_audio_reuse":
        assert caption == core.integrated_multimodal_description
        assert "fully_copy" in result
    else:
        for i in (1, 2):
            relation = f", with the synchronized speech signal copied directly from <Audio {i}>,"
            assert caption.count(relation) == 1
            caption = caption.replace(relation, "", 1)
        assert caption == core.integrated_multimodal_description
        assert (
            "The audience-only score uses the synchronized music signal copied directly from <Audio 3>. A piano melody."
            in result
        )
    shared.t2va.validate_t2va_dialogue(
        core.speech_facts, core.speaker_assignments, caption
    )
    assert core.overall_soundscape in result


@pytest.mark.parametrize("transport", ["sglang", "xiaomi"])
def test_profile_fixed_targets_one_audio_only_call(job, tmp_path, transport):
    from dataclasses import replace

    client = ProfileClient()
    backend = ta.TA2VAProfileBackend(
        replace(shared.config(tmp_path), transport=transport), client=client
    )
    targets = [
        {"speaker_group": sx, "speaker_id": sx, "segments": []} for sx in ("S1", "S2")
    ]
    profiles, raw = backend.profile(
        targets,
        [
            (
                {"speaker_group": sx, "speaker_id": sx},
                Path(job.audio_evidence.speech_path),
            )
            for sx in ("S1", "S2")
        ],
    )
    assert [p.speaker_group for p in profiles] == ["S1", "S2"]
    assert raw["model_call_count"] == 1 and not raw["error"]
    request = client.calls[0]
    assert request["messages"][0]["content"] == SPEAKER_PROFILE_SYSTEM_PROMPT
    assert [m["type"] for m in request["messages"][1]["content"]] == [
        "text",
        "text",
        "audio_url",
        "text",
        "audio_url",
    ]
    assert request["extra_body"]["use_audio_in_video"] is False


@pytest.mark.parametrize("failure", ["fail", "identity", "wrong"])
def test_profile_failure_only_speech_variant(source, tmp_path, failure):
    root, products, client = run(source, tmp_path, **{failure: True})
    assert all(p.status == "ready" for p in products if p.variant == "full_audio_reuse")
    assert all(
        p.status == "failed" for p in products if p.variant == "target_speech_reuse"
    )
    summary = json.loads((root / "summary.json").read_text())
    assert summary["model_call_count"] == len(client.calls) == 2
    assert summary["ready_count"] == summary["failed_count"] == 2


def test_real_lineage_pcm_products_and_qa(source, tmp_path):
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    root, products, client = run(source, tmp_path)
    assert len(products) == 4 and all(p.status == "ready" for p in products)
    assert len(client.calls) == 2
    for product in products:
        if product.variant == "full_audio_reuse":
            assert Path(product.audio_references[0].path) in before
            assert not product.assets and product.model_call_count == 0
            continue
        assert [r.kind for r in product.audio_references] == [
            "speaker_speech_reuse",
            "music_reuse",
        ]
        for asset in product.assets:
            pcm = read_reuse_asset_pcm16(Path(asset.output_path), asset.output_sha256)
            stem = read_canonical_stem_pcm16(
                Path(asset.source_path), asset.source_sha256
            )
            assert len(pcm) == asset.frame_count == 32000
            expected = np.zeros((32000, 2), dtype=np.int16)
            if asset.kind == "speaker_speech_reuse":
                for r in asset.ranges:
                    expected[r.target_start_sample : r.target_end_sample] = stem[
                        r.target_start_sample : r.target_end_sample
                    ]
            else:
                expected[: min(len(stem), 32000)] = stem[:32000]
            np.testing.assert_array_equal(pcm, expected)
        raw = json.loads(
            (root / "raw_profiles" / f"{product.clip_uid}.json").read_text()
        )
        for snippet in raw["snippets"]:
            pcm = read_reuse_asset_pcm16(Path(snippet["path"]), snippet["sha256"])
            label = snippet["label"]
            assert len(pcm) == label["target_end_sample"] - label["target_start_sample"]
            asset = next(a for a in product.assets if a.speaker_id == label["speaker_id"])
            source_pcm = read_canonical_stem_pcm16(Path(asset.source_path), asset.source_sha256)
            np.testing.assert_array_equal(
                pcm, source_pcm[label["target_start_sample"]:label["target_end_sample"]]
            )
    page = build_ta2va_qa(
        root, media_root=tmp_path, media_base_url="http://127.0.0.1:8765"
    )
    data = json.loads(page.with_name("data.json").read_text())
    assert len(data["cases"]) == 2
    for case in data["cases"]:
        assert case["video_url"].startswith("http://127.0.0.1:8765/")
        assert {v["variant"] for v in case["variants"]} == {
            "full_audio_reuse",
            "target_speech_reuse",
        }
        assert all(v["prompt"] for v in case["variants"])
    assert all(p.read_bytes() == contents for p, contents in before.items())
    with pytest.raises(FileExistsError):
        run(source, tmp_path)
    Path(products[-1].audio_references[0].path).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="artifact hash differs"):
        build_ta2va_qa(root, overwrite=True)


def test_changed_source_fails_before_profile(source, tmp_path):
    inventory = json.loads((source / "inventory.json").read_text())
    path = Path(
        next(
            j["audio_evidence"]["speech_path"]
            for j in inventory["jobs"]
            if j["audio_evidence"]
        )
    )
    path.write_bytes(path.read_bytes() + b"changed")
    client = ProfileClient()
    with pytest.raises(ValueError):
        ta.run_ta2va_shadow(
            source,
            "stale",
            ta.TA2VAProfileBackend(shared.config(tmp_path), client=client),
            allow_unverified=True,
        )
    assert not client.calls


def test_qa_browser_playback(source, tmp_path):
    node = os.environ.get("NODE_BINARY") or shutil.which("node")
    playwright = os.environ.get("QA_PLAYWRIGHT_MODULE")
    if not node or not playwright:
        pytest.skip("local Playwright unavailable")
    root, _, _ = run(source, tmp_path)
    page = build_ta2va_qa(root)
    script = tmp_path / "browser.cjs"
    script.write_text(r"""
const assert=require("assert"),{chromium}=require(process.argv[2]);
(async()=>{
 const browser=await chromium.launch({headless:true,channel:process.env.QA_BROWSER_CHANNEL});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1000}});
  const errors=[];page.on("pageerror",e=>errors.push(e.message));
  await page.route(/^https?:/,r=>{throw Error("No network: "+r.request().url());});
  await page.goto(process.argv[3]);
  assert.strictEqual(await page.locator("#error").textContent(),"");
  assert.strictEqual(await page.locator("section.variant").count(),2);
  assert.strictEqual(await page.locator("audio").count(),4);
  for(const selector of ["#video","#full",".ref audio"]){
   const media=page.locator(selector).first();
   await media.evaluate(async m=>{m.muted=true;await m.play();});
   await page.waitForFunction(s=>document.querySelector(s).currentTime>0,selector);
   await media.evaluate(m=>m.pause());
  }
  await page.screenshot({path:process.argv[4]+"/ta2va-desktop.png",fullPage:true});
  await page.setViewportSize({width:390,height:844});
  await page.screenshot({path:process.argv[4]+"/ta2va-mobile.png",fullPage:true});
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
  assert.deepStrictEqual(errors,[]);
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1);});
""")
    result = subprocess.run(
        [node, str(script), playwright, page.as_uri(), str(tmp_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
