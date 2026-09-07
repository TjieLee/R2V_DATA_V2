from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from r2v_data_v2.h3 import audio_shadow_qa as qa
from r2v_data_v2.h3.jea_final_renderer import FinalH3SampleV2
from r2v_data_v2.h3.mimo25_av_reconcile import _inventory, _job
from r2v_data_v2.h3.mimo25_backend import (
    MimoAuxAudioDescriptionCall,
    MimoAVAnnotationDraft,
    MimoBackendConfig,
    MimoBackendFailure,
    MimoCompletionDiagnostic,
    MimoMediaResolver,
    MimoUsage,
)
from r2v_data_v2.h3.mimo25_stem_shadow import (
    MIMO25_STEM_RECONCILE_STAGE,
    build_stem_reconcile_jobs,
    run_mimo25_stem_reconcile_shadow,
)
from r2v_data_v2.h3.sam_audio_stem_shadow import (
    load_stem_shadow,
    sha256_file,
    validate_stem_diarization_lineage,
)
from tests.test_h3_mimo25_av_shadow import _annotation, _sample
from tests.test_h3_sam_audio_stem_shadow import (
    _mimo_inventory_for_clip_order,
    _MixedDiarization,
    _run_three_clip_shadow_to_asr,
)
from tools.build_h3_audio_shadow_qa import main


class _Reconcile:
    def __init__(self, tmp_path):
        self.provenance = MimoBackendConfig(
            media_resolver=MimoMediaResolver(mode="base64", media_root=tmp_path),
            api_key="fake", transport="sglang",
        ).provenance()

    def describe_auxiliary_audio(self, path):
        return MimoAuxAudioDescriptionCall(
            description="A quiet sound.", raw_response='{"description":"A quiet sound."}',
            diagnostic=MimoCompletionDiagnostic(
                input_modality="auxiliary_audio_only", usage=MimoUsage(), http_attempt_count=1,
            ),
        )

    def reconcile(self, job, **kwargs):
        if job.clip_uid == "clip-a":
            raise MimoBackendFailure(
                code="synthetic_failed", reason="failed <script>not markup</script>",
                model_call_count=1,
                raw_responses=(_annotation().model_dump_json(),),
            )
        # Pure fixture output; no client or model is called.
        annotation = MimoAVAnnotationDraft.model_validate_json(
            _annotation().model_dump_json().replace("segment_1", "segment_0001")
        )
        values = annotation.model_dump(mode="json")
        if not job.segments:
            values["visual_observation"]["segment_views"] = []
            values["audio_observation"]["segment_decisions"] = []
            values["audio_observation"]["speaker_voice_profiles"] = []
            values["av_grounding"]["segment_groundings"] = []
            values["h3_semantics"]["shot1_caption"] = "The seated person remains still."
        annotation = MimoAVAnnotationDraft.model_validate(values)
        return SimpleNamespace(annotation=annotation, raw_responses=(), diagnostics=(), model_call_count=1)


def _fixture(tmp_path, monkeypatch, *, mixed=False, variants=False):
    root = tmp_path / "production"
    root.mkdir()
    separation, diarization, asr, inventory, order = _run_three_clip_shadow_to_asr(
        root, shadow_run_id="random10-v1",
        diarization_backend=_MixedDiarization() if mixed else None,
    )
    shadow = separation.parent
    base = _mimo_inventory_for_clip_order(tmp_path / "base", order)
    jobs = []
    for old, stem in zip(base.jobs, inventory.jobs, strict=True):
        values = old.model_dump(mode="json", exclude={"request_fingerprint"})
        values.update(
            target_video_path=stem.target_video_path,
            target_video_sha256=stem.target_video_sha256,
            target_full_audio_path=stem.source_audio_path,
            target_full_audio_sha256=stem.source_audio_sha256,
        )
        if variants:
            values["source_h3_sample_ids"].append(f"{old.clip_uid}/canonical")
        for ref in values["reference_images"]:
            Image.new("RGB", (160, 200), (98, 150, 140)).save(ref["image_artifact_path"])
            ref["image_sha256"] = sha256_file(Path(ref["image_artifact_path"]))
        jobs.append(_job(values))
    values = base.model_dump(mode="json", exclude={"inventory_fingerprint"})
    values["jobs"] = [job.model_dump(mode="json") for job in jobs]
    sample_root = tmp_path / "sample-template"
    sample_root.mkdir()
    template = _sample(sample_root)
    samples_path = root / "h3/samples.jsonl"
    samples_path.parent.mkdir()
    samples = []
    for job in jobs:
        sample_values = template.model_dump(mode="json")
        sample_values.update(sample_id=job.source_h3_sample_ids[0], clip_uid=job.clip_uid,
                             target_video=job.target_video_path,
                             target_full_audio_path=job.target_full_audio_path,
                             target_full_audio_sha256=job.target_full_audio_sha256)
        sample_values["visual_references"][0]["image_artifact_path"] = job.reference_images[0].image_artifact_path
        sample_values["subject_voices"][0]["target_occurrence_id"] = f"{job.clip_uid}/e1"
        samples.append(FinalH3SampleV2.model_validate(sample_values))
        if variants:
            samples.append(FinalH3SampleV2.model_validate({
                **sample_values, "sample_id": f"{job.clip_uid}/canonical",
                "pair_type": "canonical", "subject_voices": [],
            }))
    samples_path.write_text("".join(sample.model_dump_json() + "\n" for sample in samples))
    values["source_h3_samples_sha256"] = sha256_file(samples_path)
    base = _inventory(values)
    monkeypatch.setattr(qa, "build_mimo25_inventory", lambda **kwargs: base)
    assert not (shadow / "mimo_stem_facts").exists()
    provenance, _, _ = validate_stem_diarization_lineage(diarization)
    jobs = build_stem_reconcile_jobs(
        base_inventory=qa.usable_stem_reconcile_inventory(base, provenance),
        stem_diarization_root=diarization,
        stem_asr_root=asr, route="music_first",
    )
    run_mimo25_stem_reconcile_shadow(
        jobs=jobs, stem_records=load_stem_shadow(separation)[1],
        backend=_Reconcile(tmp_path), output_root=shadow / MIMO25_STEM_RECONCILE_STAGE,
        source_clip_uids=order, route="music_first",
        diarization_failed_clips=provenance.diarization_failed_clips,
        allow_unverified=True,
    )
    visual, runs = tmp_path / "visual", tmp_path / "runs"
    visual.mkdir()
    runs.mkdir()
    kwargs = {
        "visual_production_root": visual, "visual_runs_root": runs,
        "audio_production_root": root, "case_manifest": root / "case.json",
        "shadow_run_id": "random10-v1",
    }
    return kwargs, shadow


def _snapshot(root):
    return {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_builder_ready_failed_order_media_and_sources_unchanged(tmp_path, monkeypatch):
    kwargs, shadow = _fixture(tmp_path, monkeypatch)
    materialized = []
    original = qa._materialize_sample

    def capture(sample, job, record, **kwargs):
        result = original(sample, job, record, **kwargs)
        materialized.append((sample, job, record, result[1]))
        return result

    monkeypatch.setattr(qa, "_materialize_sample", capture)
    before = _snapshot(tmp_path)
    result = qa.build_audio_shadow_qa(**kwargs)
    assert result["model_called"] is False
    output = shadow / "qa"
    assert result["output_root"] == str(output)
    data = json.loads((output / "data.json").read_text())
    reconcile_summary = json.loads((shadow / MIMO25_STEM_RECONCILE_STAGE / "summary.json").read_text())
    assert reconcile_summary["schema_version"] == "r2v.h3.mimo25_stem_reconcile_summary.12"
    assert reconcile_summary["current_mimo_versions_modified"] is True
    assert data["clip_uids"] == ["clip-z", "clip-a", "clip-m"]
    assert [clip["reconcile"]["status"] for clip in data["clips"]] == ["ready", "failed", "ready"]
    failed = data["clips"][1]["reconcile"]
    assert failed["failure_code"] == "synthetic_failed"
    assert failed["model_call_count"] == 3
    assert data["clips"][1]["direct_h3"]["style_opening"]
    assert data["clips"][1]["direct_h3"]["shot1_caption"]
    assert data["clips"][0]["reconcile"]["annotation"]["audio_observation"]
    assert [item[1].clip_uid for item in materialized] == ["clip-z", "clip-m"]
    for clip, call in zip((data["clips"][0], data["clips"][2]), materialized, strict=True):
        final = clip["final_h3"]
        assert final["status"] == "ready"
        assert final["materializer_version"] == "h3_mimo25_materializer_v23"
        assert final["text"] == call[3] == original(*call[:3])[1]
        assert final["variants"][0]["text"] == final["text"]
        assert "[[" not in final["text"]
        assert clip["direct_h3"]["shot1_caption"]
        assert clip["production_h3"]
        assert "MiMo DIRECT H3" in (output / "review.html").read_text()
        assert "<Picture 1>" in final["text"]
        sections = ["subject_definitions", "summary", "retention_analysis",
                    "detailed_description", "overall_soundscape", "non_diegetic_music"]
        positions = [final["text"].index(section + ":\n") for section in sections]
        assert positions == sorted(positions)
        assert clip["direct_h3"]["shot1_caption"] in final["text"]
        assert final["text"].count("[Shot 1]") == 1
    assert data["clips"][1]["final_h3"]["status"] == "unavailable"
    assert data["clips"][1]["final_h3"]["text"] is None
    assert data["clips"][1]["final_h3"]["reason"] == "AV reconcile failed: failed <script>not markup</script>"
    assert data["qa_labels"] == list(qa.QA_LABELS)
    assert all(path.read_bytes() == content for path, content in before.items())
    for url, source in data["media"].items():
        link = output / url
        assert link.is_symlink()
        assert link.resolve() == Path(source["source_path"])
        assert link.read_bytes() == Path(source["source_path"]).read_bytes()
    html = (output / "review.html").read_text()
    assert 'fetch("data.json", {cache: "no-store"})' in html
    assert "Export QA JSON" in html and "Import QA JSON" in html
    assert "localStorage.clear" not in html
    assert "innerHTML" not in html
    assert "frozen_production_evidence" in data["clips"][0]["speaker_binding"]
    assert "stem_segment_evidence" in data["clips"][0]["speaker_binding"]
    assert "stem_facts" not in data["clips"][0]
    assert data["clips"][1]["direct_h3"]["overall_soundscape"]
    assert data["clips"][1]["direct_h3"]["non_diegetic_music"]
    assert not (shadow / "mimo_stem_facts").exists()
    first_data = (output / "data.json").read_bytes()
    qa.build_audio_shadow_qa(**kwargs, overwrite=True)
    assert (output / "data.json").read_bytes() == first_data


def test_upstream_failed_and_empty_clips_remain_visible(tmp_path, monkeypatch):
    kwargs, shadow = _fixture(tmp_path, monkeypatch, mixed=True)
    qa.build_audio_shadow_qa(**kwargs)
    clips = json.loads((shadow / "qa/data.json").read_text())["clips"]
    assert [clip["diarization"]["clip_result"]["status"] for clip in clips] == ["ready", "failed", "empty"]
    assert clips[1]["reconcile"]["status"] == "upstream_failed"
    assert clips[1]["reconcile"]["model_call_count"] == 0
    assert "stem_facts" not in clips[1]
    assert clips[1]["asr"]["segments"] == []
    assert clips[1]["final_h3"]["status"] == "unavailable"
    assert clips[1]["final_h3"]["text"] is None
    assert clips[1]["final_h3"]["reason"] == "upstream_failed"
    assert clips[2]["asr"]["segments"] == []


def test_final_text_changes_review_fingerprints(tmp_path, monkeypatch):
    kwargs, shadow = _fixture(tmp_path, monkeypatch)
    qa.build_audio_shadow_qa(**kwargs)
    before = json.loads((shadow / "qa/data.json").read_text())
    original = qa._materialize_sample

    def changed_text(*args, **kwargs):
        corrected, text, warnings = original(*args, **kwargs)
        return corrected, text + "\nsynthetic materializer change", warnings

    monkeypatch.setattr(qa, "_materialize_sample", changed_text)
    qa.build_audio_shadow_qa(**kwargs, overwrite=True)
    after = json.loads((shadow / "qa/data.json").read_text())
    assert before["dataset_fingerprint"] != after["dataset_fingerprint"]
    for old, new in zip(before["clips"], after["clips"], strict=True):
        assert (old["record_fingerprint"] == new["record_fingerprint"]) == (
            old["reconcile"]["status"] == "failed"
        )


def test_final_h3_keeps_conditioning_variants_separate(tmp_path, monkeypatch):
    kwargs, shadow = _fixture(tmp_path, monkeypatch, variants=True)
    qa.build_audio_shadow_qa(**kwargs)
    final = json.loads((shadow / "qa/data.json").read_text())["clips"][0]["final_h3"]
    assert [row["sample_id"] for row in final["variants"]] == [
        "clip-z/canonical", "clip-z/in_pair",
    ]
    canonical, voice = final["variants"]
    assert final["text"] == canonical["text"]
    assert "[reference generation]" in canonical["text"]
    assert "<Audio 1>" not in canonical["text"]
    assert "[reference generation + audio reference]" in voice["text"]
    assert voice["text"].split("detailed_description:", 1)[1] == canonical["text"].split("detailed_description:", 1)[1]


def test_output_safety_and_atomic_failure_preserve_existing_qa(tmp_path, monkeypatch):
    kwargs, shadow = _fixture(tmp_path, monkeypatch)
    for protected in (
        kwargs["audio_production_root"], kwargs["visual_production_root"],
        kwargs["visual_runs_root"], shadow, shadow / "asr",
        shadow.parent / "another-run/qa",
    ):
        with pytest.raises(ValueError, match="overlap"):
            qa.build_audio_shadow_qa(**kwargs, output_root=protected, overwrite=True)
    qa.build_audio_shadow_qa(**kwargs)
    with pytest.raises(FileExistsError):
        qa.build_audio_shadow_qa(**kwargs)
    external = tmp_path / "unknown-output"
    external.mkdir()
    (external / "sentinel").write_text("keep")
    with pytest.raises(ValueError, match="not owned"):
        qa.build_audio_shadow_qa(**kwargs, output_root=external, overwrite=True)
    assert (external / "sentinel").read_text() == "keep"
    before = _snapshot(shadow / "qa")
    original_copy = qa.shutil.copyfile

    def fail_template(source, destination):
        if Path(source).suffix == ".html":
            raise RuntimeError("synthetic publication failure")
        return original_copy(source, destination)

    monkeypatch.setattr(qa.shutil, "copyfile", fail_template)
    with pytest.raises(RuntimeError, match="publication"):
        qa.build_audio_shadow_qa(**kwargs, overwrite=True)
    assert all(path.read_bytes() == content for path, content in before.items())
    assert not list(shadow.glob(".qa.tmp-*"))


def test_stale_reconcile_fingerprint_rejected_before_publication(tmp_path, monkeypatch):
    kwargs, shadow = _fixture(tmp_path, monkeypatch)
    path = shadow / MIMO25_STEM_RECONCILE_STAGE / "records.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[0]["source_job_fingerprint"] = "f" * 64
    values = {k: v for k, v in records[0].items() if k != "record_fingerprint"}
    records[0]["record_fingerprint"] = qa._fingerprint(values)
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    with pytest.raises(ValueError, match="stale source provenance"):
        qa.build_audio_shadow_qa(**kwargs)
    assert not (shadow / "qa").exists()


def test_cli_and_manifest_order_fail_closed(tmp_path, monkeypatch):
    kwargs, shadow = _fixture(tmp_path, monkeypatch)
    argv = [part for key, value in kwargs.items() for part in ("--" + key.replace("_", "-"), str(value))]
    result = main(argv)
    assert result["clip_count"] == 3
    case = kwargs["case_manifest"]
    value = json.loads(case.read_text())
    value["clip_uids"].reverse()
    case.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="separation order"):
        main([*argv, "--overwrite"])
    assert (shadow / "qa/review.html").is_file()


@pytest.mark.parametrize("blocked, expected", [
    ("multi", "multi_speaker_segment_requires_turn_refinement"),
    ("placeholder", "direct_internal_syntax"),
])
def test_ready_annotation_with_blocked_materialization_is_unavailable(tmp_path, monkeypatch, blocked, expected):
    original = _Reconcile.reconcile

    def reconcile(self, job, **kwargs):
        result = original(self, job, **kwargs)
        values = result.annotation.model_dump(mode="json")
        if blocked == "multi":
            for decision in values["audio_observation"]["segment_decisions"]:
                decision.update(
                    vocal_composition="sequential_multi_speaker_speech",
                    resolution="needs_acoustic_refinement",
                    secondary_vocal_activity={"present": True, "speaker_relation": "different_speaker", "kind": "speech"},
                )
            values["audio_observation"]["speaker_voice_profiles"] = []
            for grounding in values["av_grounding"]["segment_groundings"]:
                grounding.update(binding_status="no_reliable_entity", entity_id=None,
                                 speech_presentation="uncertain", confidence="low",
                                 evidence_codes=["insufficient_evidence"])
        else:
            values["h3_semantics"]["shot1_caption"] += " [[unknown]]"
        return SimpleNamespace(annotation=MimoAVAnnotationDraft.model_validate(values),
                               raw_responses=(), diagnostics=(), model_call_count=1)

    monkeypatch.setattr(_Reconcile, "reconcile", reconcile)
    kwargs, shadow = _fixture(tmp_path, monkeypatch, variants=True)
    before = _snapshot(tmp_path)
    qa.build_audio_shadow_qa(**kwargs)
    data = json.loads((shadow / "qa/data.json").read_text())
    assert data["schema_version"] == "r2v.h3.audio_shadow_qa.5"
    for clip in (data["clips"][0], data["clips"][2]):
        assert clip["reconcile"]["status"] == "ready"
        final = clip["final_h3"]
        assert clip["direct_h3"]["shot1_caption"]
        assert final["status"] == "unavailable" and final["text"] is None
        assert final["reason"] == "materialization_contract_failed"
        for variant in final["variants"]:
            assert variant["text"] is None
            assert expected in {issue["code"] for issue in variant["issues"]}
    assert all(path.read_bytes() == content for path, content in before.items())


def test_generated_javascript_and_qa_roundtrip(tmp_path, monkeypatch):
    node = os.environ.get("NODE_BINARY") or shutil.which("node")
    if not node:
        pytest.skip("Node.js unavailable for generated JavaScript execution")
    kwargs, shadow = _fixture(tmp_path, monkeypatch)
    qa.build_audio_shadow_qa(**kwargs)
    html = (shadow / "qa/review.html").read_text()
    script = re.search(r"<script>(.*?)</script>", html, re.DOTALL).group(1)
    js = tmp_path / "generated.js"
    js.write_text(script)
    subprocess.run([node, "--check", str(js)], check=True, capture_output=True, text=True)
    # Execute generated QA functions with a minimal DOM/storage, no network or model.
    prelude = r'''
const fs = require("fs");
const dataset = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const elements = new Map(), saved = new Map();
const inputs = dataset.qa_labels.map(label => ({dataset:{qaLabel:label}, checked:false}));
function el(id) {
  if (!elements.has(id)) elements.set(id, {value: "", hidden: false, textContent: ""});
  return elements.get(id);
}
const document = {
  getElementById: el,
  querySelector: selector => inputs.find(input => selector.includes('"' + input.dataset.qaLabel + '"')),
  querySelectorAll: () => inputs,
  addEventListener: () => {}
};
const localStorage = {
  getItem: key => saved.get(key) ?? null,
  setItem: (key, value) => saved.set(key, value)
};
const fetch = () => new Promise(() => {});
'''
    checks = r'''
data = dataset; index = 1;
$("notes").value = "review <script> text";
render = function() { updateProgress(); };
function change(label, checked) {
  const input = inputs.find(input => input.dataset.qaLabel === label);
  input.checked = checked; labelChanged(input);
}
function expectLabels(labels) {
  if (JSON.stringify(stateFor(data.clips[index]).labels) !== JSON.stringify(labels))
    throw Error("selection was not immediately persisted");
  if (JSON.stringify(exportPayload().annotations[0].labels) !== JSON.stringify(labels))
    throw Error("export selection differs");
}
change("better", true); expectLabels(["better"]);
change("same", true); expectLabels(["same"]);
change("same", false); expectLabels([]);
change("dialogue_wrong", true); expectLabels(["dialogue_wrong"]);
change("audio_wrong", true); expectLabels(["dialogue_wrong", "audio_wrong"]);
change("dialogue_wrong", false); change("audio_wrong", false); expectLabels([]);
change("dialogue_wrong", true); change("audio_wrong", true);
change("dialogue_wrong", false); expectLabels(["audio_wrong"]);
const first = exportPayload();
if (first.annotations[0].clip_uid !== data.clips[1].clip_uid) throw Error("wrong clip");
if (first.annotations[0].labels.join(",") !== "audio_wrong") throw Error("DOM state was not exported");
if (stateFor(data.clips[1]).notes !== $("notes").value) throw Error("notes not restored");
if ($("progress").textContent !== "Labeled 1 / 3") throw Error("progress wrong");
importPayload(first);
if (JSON.stringify(exportPayload()) !== JSON.stringify(first)) throw Error("roundtrip unstable");
const old = JSON.stringify(first);
first.dataset_fingerprint = "stale";
let rejected = false; try { importPayload(first); } catch (e) { rejected = true; }
if (!rejected) throw Error("stale import accepted");
const invalid = JSON.parse(old); invalid.annotations[0].labels = ["invented"];
rejected = false; try { importPayload(invalid); } catch (e) { rejected = true; }
if (!rejected) throw Error("unknown label accepted");
const contradictory = JSON.parse(old); contradictory.annotations[0].labels = ["better", "same"];
const beforeImport = JSON.stringify([...saved]);
rejected = false; try { importPayload(contradictory); } catch (e) {
  rejected = e.message === "Invalid, duplicate, or stale QA annotation.";
}
if (!rejected || JSON.stringify([...saved]) !== beforeImport) throw Error("contradictory import accepted or rewritten");
const storageKey = key(data.clips[index]), previousState = saved.get(storageKey);
saved.set(storageKey, JSON.stringify({labels:["better", "same"], notes:"stale"}));
rejected = false; try { stateFor(data.clips[index]); } catch (e) {
  rejected = e.message.startsWith("Invalid saved QA state");
}
if (!rejected) throw Error("contradictory saved state accepted");
saved.set(storageKey, previousState);
inputs.find(input => input.dataset.qaLabel === "better").checked = true;
inputs.find(input => input.dataset.qaLabel === "same").checked = true;
rejected = false; try { exportPayload(); } catch (e) { rejected = true; }
if (!rejected || saved.get(storageKey) !== previousState) throw Error("contradictory DOM exported or persisted");
inputs.find(input => input.dataset.qaLabel === "better").checked = false;
inputs.find(input => input.dataset.qaLabel === "same").checked = false;
const duplicate = JSON.parse(old); duplicate.annotations.push(duplicate.annotations[0]);
rejected = false; try { importPayload(duplicate); } catch (e) { rejected = true; }
if (!rejected) throw Error("duplicate accepted");
const staleRecord = JSON.parse(old); staleRecord.annotations[0].record_fingerprint = "stale";
rejected = false; try { importPayload(staleRecord); } catch (e) { rejected = true; }
if (!rejected) throw Error("stale record accepted");
index = 0; $("notes").value = ""; saveCurrent();
if (exportPayload().annotations.map(x => x.clip_uid).join(",") !== "clip-z,clip-a")
  throw Error("export did not retain authoritative order");
index = 0; $("final-variant").value = "0";
data.clips[0].final_h3.variants[0] = {status:"unavailable", text:null,
  issues:[{code:"multi_speaker_segment_requires_turn_refinement", message:"retain ASR"}]};
renderFinalVariant();
if ($("final-unavailable").hidden || !$("copy-final").disabled ||
    $("final-text").textContent !== "" ||
    !$("final-unavailable").textContent.includes("multi_speaker_segment_requires_turn_refinement"))
  throw Error("blocked materialization was not displayed");
'''
    harness = tmp_path / "test.cjs"
    harness.write_text(prelude + "\n" + script + "\n" + checks)
    subprocess.run(
        [node, str(harness), str(shadow / "qa/data.json")],
        check=True, capture_output=True, text=True,
    )


def test_synthetic_browser_review_desktop_mobile(tmp_path, monkeypatch):
    node = os.environ.get("NODE_BINARY") or shutil.which("node")
    playwright = os.environ.get("QA_PLAYWRIGHT_MODULE")
    if not node or not playwright:
        pytest.skip("optional preinstalled Playwright not configured")
    kwargs, shadow = _fixture(tmp_path, monkeypatch, variants=True)
    qa.build_audio_shadow_qa(**kwargs)
    script = tmp_path / "browser.cjs"
    script.write_text(r'''
const fs = require("fs"), path = require("path"), assert = require("assert");
const {chromium} = require(process.argv[2]);
(async () => {
  const browser = await chromium.launch({headless: true, channel:process.env.QA_BROWSER_CHANNEL});
  try {
    const page = await browser.newPage({viewport:{width:1440,height:1000}});
    const errors = []; page.on("pageerror", error => errors.push(error.message));
    const root = process.argv[3];
    await page.route("**/*", async route => {
      const url = new URL(route.request().url());
      if (url.hostname !== "qa.invalid") throw Error("Unexpected network request");
      const filename = path.join(root, url.pathname);
      if (!fs.existsSync(filename)) return route.fulfill({status:404, body:"fixture only"});
      const type = filename.endsWith(".html") ? "text/html" : filename.endsWith(".json") ? "application/json" : filename.endsWith(".wav") ? "audio/wav" : filename.endsWith(".png") ? "image/png" : "application/octet-stream";
      return route.fulfill({body: fs.readFileSync(filename), contentType:type});
    });
    await page.goto("http://qa.invalid/review.html");
    await page.locator("#main").waitFor({state:"visible"});
    await page.waitForFunction(() => [...document.querySelectorAll("#references img")].every(image => image.naturalWidth > 0));
    assert.strictEqual(await page.locator("#clip-id").textContent(), "clip-z");
    const dataset = JSON.parse(fs.readFileSync(path.join(root, "data.json"), "utf8"));
    const expectedFinal = dataset.clips[0].final_h3.text;
    assert.strictEqual(await page.locator("#final-text").textContent(), expectedFinal);
    assert(await page.locator("#final-text").isVisible());
    assert.strictEqual(await page.locator("#final-h3").evaluate(el => el.closest("details")), null);
    assert.strictEqual(await page.locator("#materializer-version").textContent(), "h3_mimo25_materializer_v23");
    await page.locator("#final-variant").selectOption("1");
    assert.strictEqual(await page.locator("#final-text").textContent(), dataset.clips[0].final_h3.variants[1].text);
    await page.locator("#final-variant").selectOption("0");
    const label = name => page.locator('input[data-qa-label="' + name + '"]');
    async function assertSelection(expected) {
      assert.deepStrictEqual(await page.locator("input[data-qa-label]:checked").evaluateAll(
        inputs => inputs.map(input => input.dataset.qaLabel)), expected);
      assert.deepStrictEqual(await page.evaluate(() => stateFor(data.clips[index]).labels), expected);
    }
    await label("better").check();
    await label("same").check(); await assertSelection(["same"]);
    await label("same").uncheck(); await assertSelection([]);
    await label("dialogue_wrong").check(); await assertSelection(["dialogue_wrong"]);
    await label("audio_wrong").check(); await assertSelection(["dialogue_wrong", "audio_wrong"]);
    await label("dialogue_wrong").uncheck(); await label("audio_wrong").uncheck(); await assertSelection([]);
    await page.locator('input[data-qa-label="audio_wrong"]').check();
    await page.locator("#notes").fill("synthetic human note");
    await page.locator("#next").click();
    assert.strictEqual(await page.locator("#status").textContent(), "failed");
    assert(await page.locator("#final-unavailable").isVisible());
    assert.match(await page.locator("#final-unavailable").textContent(), /Final H3 Prompt unavailable.*Reconcile failed: synthetic_failed/s);
    assert.strictEqual(await page.locator("#final-text").textContent(), "");
    assert(await page.locator("#copy-final").isDisabled());
    assert.match(await page.locator("#reconcile-body").textContent(), /synthetic_failed/);
    await page.locator("#previous").click();
    assert.strictEqual(await page.locator("#final-text").textContent(), expectedFinal);
    assert(await page.locator("#final-text").isVisible());
    assert(!await page.locator("#final-unavailable").isVisible());
    assert.strictEqual(await page.locator("#notes").inputValue(), "synthetic human note");
    const downloaded = page.waitForEvent("download");
    await page.locator("#export").click();
    const download = await downloaded;
    const payload = JSON.parse(fs.readFileSync(await download.path(), "utf8"));
    assert.deepStrictEqual(payload.annotations[0].labels, ["audio_wrong"]);
    await page.evaluate(() => localStorage.removeItem(key(data.clips[0])));
    await page.reload(); await page.locator("#main").waitFor({state:"visible"});
    await page.locator("#import-file").setInputFiles({name:"qa.json", mimeType:"application/json", buffer:Buffer.from(JSON.stringify(payload))});
    await page.waitForFunction(() => document.getElementById("notes").value === "synthetic human note");
    const contradictory = JSON.parse(JSON.stringify(payload));
    contradictory.annotations[0].labels = ["better", "same"];
    await page.locator("#import-file").setInputFiles({name:"invalid.json", mimeType:"application/json", buffer:Buffer.from(JSON.stringify(contradictory))});
    await page.waitForFunction(() => document.getElementById("message").textContent === "Invalid, duplicate, or stale QA annotation.");
    await assertSelection(["audio_wrong"]);
    await page.screenshot({path:path.join(process.argv[4], "qa-desktop.png"), fullPage:true});
    await page.setViewportSize({width:390,height:844});
    await page.screenshot({path:path.join(process.argv[4], "qa-mobile.png"), fullPage:true});
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    await page.evaluate(() => localStorage.setItem(key(data.clips[0]),
      JSON.stringify({labels:["better", "same"], notes:"injected"})));
    await page.reload();
    await page.waitForFunction(() => document.getElementById("message").textContent.startsWith("Invalid saved QA state"));
    assert.strictEqual(await page.locator("#main").isVisible(), false);
    assert.deepStrictEqual(errors, []);
  } finally { await browser.close(); }
})().catch(error => {console.error(error);process.exitCode=1;});
''')
    result = subprocess.run(
        [node, str(script), playwright, str(shadow / "qa"), str(tmp_path)],
        check=False, capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stderr
