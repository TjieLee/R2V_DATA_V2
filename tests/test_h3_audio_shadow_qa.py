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
from r2v_data_v2.h3.mimo25_av_reconcile import _inventory, _job
from r2v_data_v2.h3.mimo25_backend import MimoAVAnnotationDraft, MimoBackendFailure
from r2v_data_v2.h3.mimo25_stem_shadow import (
    build_stem_reconcile_jobs,
    load_stem_fact_records,
    run_mimo25_stem_facts_shadow,
    run_mimo25_stem_reconcile_shadow,
)
from r2v_data_v2.h3.sam_audio_stem_shadow import sha256_file
from tests.test_h3_mimo25_av_shadow import _annotation
from tests.test_h3_sam_audio_stem_shadow import (
    _FactsBackend,
    _FailingReconcileBackend,
    _mimo_inventory_for_clip_order,
    _MixedDiarization,
    _run_three_clip_shadow_to_asr,
    _ViewBackend,
)
from tools.build_h3_audio_shadow_qa import main


class _Reconcile(_FailingReconcileBackend):
    def reconcile(self, job, **kwargs):
        if job.clip_uid == "clip-a":
            raise MimoBackendFailure(
                code="synthetic_failed", reason="failed <script>not markup</script>",
                model_call_count=2,
            )
        # Pure fixture output; no client or model is called.
        annotation = MimoAVAnnotationDraft.model_validate_json(
            _annotation().model_dump_json().replace("segment_1", "segment_0001")
        )
        return SimpleNamespace(annotation=annotation, diagnostics=(), model_call_count=1)


def _fixture(tmp_path, monkeypatch, *, mixed=False):
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
        for ref in values["reference_images"]:
            Image.new("RGB", (160, 200), (98, 150, 140)).save(ref["image_artifact_path"])
            ref["image_sha256"] = sha256_file(Path(ref["image_artifact_path"]))
        jobs.append(_job(values))
    values = base.model_dump(mode="json", exclude={"inventory_fingerprint"})
    values["jobs"] = [job.model_dump(mode="json") for job in jobs]
    base = _inventory(values)
    monkeypatch.setattr(qa, "build_mimo25_inventory", lambda **kwargs: base)
    facts_summary = run_mimo25_stem_facts_shadow(
        stem_root=separation, route="music_first", backend=_FactsBackend(),
        view_backend=_ViewBackend(), output_root=shadow / "mimo_stem_facts",
        allow_unverified=True,
    )
    jobs = build_stem_reconcile_jobs(
        base_inventory=base, stem_diarization_root=diarization,
        stem_asr_root=asr, route="music_first",
    )
    run_mimo25_stem_reconcile_shadow(
        jobs=jobs, stem_facts=load_stem_fact_records(shadow / "mimo_stem_facts"),
        backend=_Reconcile(tmp_path), output_root=shadow / "mimo_reconcile",
        source_clip_uids=order, route="music_first",
        diarization_failed_clips=facts_summary.diarization_failed_clips,
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
    before = _snapshot(tmp_path)
    result = qa.build_audio_shadow_qa(**kwargs)
    assert result["model_called"] is False
    output = shadow / "qa"
    assert result["output_root"] == str(output)
    data = json.loads((output / "data.json").read_text())
    assert data["clip_uids"] == ["clip-z", "clip-a", "clip-m"]
    assert [clip["reconcile"]["status"] for clip in data["clips"]] == ["ready", "failed", "ready"]
    failed = data["clips"][1]["reconcile"]
    assert failed["failure_code"] == "synthetic_failed"
    assert failed["model_call_count"] == 2
    assert data["clips"][0]["reconcile"]["annotation"]["audio_observation"]
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
    assert "raw_response" not in data["clips"][0]["stem_facts"]["diagnostics"][0]
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
    assert clips[1]["stem_facts"] is None
    assert clips[1]["asr"]["segments"] == []
    assert clips[2]["asr"]["segments"] == []


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
    path = shadow / "mimo_reconcile/records.jsonl"
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
change("good", true); expectLabels(["good"]);
change("asr_issue", true); expectLabels(["asr_issue"]);
change("stem_issue", true); expectLabels(["asr_issue", "stem_issue"]);
change("good", true); expectLabels(["good"]);
change("good", false); expectLabels([]);
change("asr_issue", true); change("stem_issue", true);
change("asr_issue", false); expectLabels(["stem_issue"]);
const first = exportPayload();
if (first.annotations[0].clip_uid !== data.clips[1].clip_uid) throw Error("wrong clip");
if (first.annotations[0].labels.join(",") !== "stem_issue") throw Error("DOM state was not exported");
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
const contradictory = JSON.parse(old); contradictory.annotations[0].labels = ["good", "asr_issue"];
const beforeImport = JSON.stringify([...saved]);
rejected = false; try { importPayload(contradictory); } catch (e) {
  rejected = e.message === "Invalid, duplicate, or stale QA annotation.";
}
if (!rejected || JSON.stringify([...saved]) !== beforeImport) throw Error("contradictory import accepted or rewritten");
const storageKey = key(data.clips[index]), previousState = saved.get(storageKey);
saved.set(storageKey, JSON.stringify({labels:["good", "asr_issue"], notes:"stale"}));
rejected = false; try { stateFor(data.clips[index]); } catch (e) {
  rejected = e.message.startsWith("Invalid saved QA state");
}
if (!rejected) throw Error("contradictory saved state accepted");
saved.set(storageKey, previousState);
inputs.find(input => input.dataset.qaLabel === "good").checked = true;
rejected = false; try { exportPayload(); } catch (e) { rejected = true; }
if (!rejected || saved.get(storageKey) !== previousState) throw Error("contradictory DOM exported or persisted");
inputs.find(input => input.dataset.qaLabel === "good").checked = false;
const duplicate = JSON.parse(old); duplicate.annotations.push(duplicate.annotations[0]);
rejected = false; try { importPayload(duplicate); } catch (e) { rejected = true; }
if (!rejected) throw Error("duplicate accepted");
const staleRecord = JSON.parse(old); staleRecord.annotations[0].record_fingerprint = "stale";
rejected = false; try { importPayload(staleRecord); } catch (e) { rejected = true; }
if (!rejected) throw Error("stale record accepted");
index = 0; $("notes").value = ""; saveCurrent();
if (exportPayload().annotations.map(x => x.clip_uid).join(",") !== "clip-z,clip-a")
  throw Error("export did not retain authoritative order");
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
    kwargs, shadow = _fixture(tmp_path, monkeypatch)
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
    const label = name => page.locator('input[data-qa-label="' + name + '"]');
    async function assertSelection(expected) {
      assert.deepStrictEqual(await page.locator("input[data-qa-label]:checked").evaluateAll(
        inputs => inputs.map(input => input.dataset.qaLabel)), expected);
      assert.deepStrictEqual(await page.evaluate(() => stateFor(data.clips[index]).labels), expected);
    }
    await label("good").check();
    await label("asr_issue").check(); await assertSelection(["asr_issue"]);
    await label("stem_issue").check(); await assertSelection(["asr_issue", "stem_issue"]);
    await label("good").check(); await assertSelection(["good"]);
    await label("good").uncheck(); await assertSelection([]);
    await page.locator('input[data-qa-label="stem_issue"]').check();
    await page.locator("#notes").fill("synthetic human note");
    await page.locator("#next").click();
    assert.strictEqual(await page.locator("#status").textContent(), "failed");
    assert.match(await page.locator("#reconcile-body").textContent(), /synthetic_failed/);
    await page.locator("#previous").click();
    assert.strictEqual(await page.locator("#notes").inputValue(), "synthetic human note");
    const downloaded = page.waitForEvent("download");
    await page.locator("#export").click();
    const download = await downloaded;
    const payload = JSON.parse(fs.readFileSync(await download.path(), "utf8"));
    assert.deepStrictEqual(payload.annotations[0].labels, ["stem_issue"]);
    await page.evaluate(() => localStorage.removeItem(key(data.clips[0])));
    await page.reload(); await page.locator("#main").waitFor({state:"visible"});
    await page.locator("#import-file").setInputFiles({name:"qa.json", mimeType:"application/json", buffer:Buffer.from(JSON.stringify(payload))});
    await page.waitForFunction(() => document.getElementById("notes").value === "synthetic human note");
    const contradictory = JSON.parse(JSON.stringify(payload));
    contradictory.annotations[0].labels = ["good", "asr_issue"];
    await page.locator("#import-file").setInputFiles({name:"invalid.json", mimeType:"application/json", buffer:Buffer.from(JSON.stringify(contradictory))});
    await page.waitForFunction(() => document.getElementById("message").textContent === "Invalid, duplicate, or stale QA annotation.");
    await assertSelection(["stem_issue"]);
    await page.screenshot({path:path.join(process.argv[4], "qa-desktop.png"), fullPage:true});
    await page.setViewportSize({width:390,height:844});
    await page.screenshot({path:path.join(process.argv[4], "qa-mobile.png"), fullPage:true});
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    await page.evaluate(() => localStorage.setItem(key(data.clips[0]),
      JSON.stringify({labels:["good", "asr_issue"], notes:"injected"})));
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
