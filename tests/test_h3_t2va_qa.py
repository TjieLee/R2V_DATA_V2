import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from r2v_data_v2.h3 import t2va_shadow as t2va
from r2v_data_v2.h3.t2va_mimo_backend import T2VAMimoBackend
from r2v_data_v2.h3.t2va_qa import build_t2va_qa
from tests import test_h3_t2va_shadow as shared
from tests.test_h3_t2va_shadow import Client, config, draft_for

job = shared.job


@pytest.fixture
def published(job, tmp_path, monkeypatch):
    ffmpeg = os.environ.get("FFMPEG_BINARY") or shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("CPU ffmpeg fixture unavailable")
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=640x360:rate=4:duration=5",
            "-pix_fmt",
            "yuv420p",
            job.target_video_path,
        ],
        check=True,
        capture_output=True,
    )
    job.target_video_sha256 = t2va.sha256_file(Path(job.target_video_path))
    job.clip_display_path = "</script><script>window.injected=true</script>"
    other = job.model_copy(update={"clip_uid": "second", "speech_facts": []})
    production = tmp_path / "production"
    production.mkdir()
    values = {
        "schema_version": "r2v.h3.t2va_inventory.2",
        "shot_selection": {
            "schema_version": "r2v.h3.t2va_shot_selection.1",
            "shot_manifest_path": "/synthetic/shots.jsonl",
            "shot_manifest_sha256": "a" * 64,
            "clips_root": str(tmp_path),
            "source_videos_root": str(tmp_path),
            "selection_mode": "all",
            "sample_seed": None,
            "case_manifest_path": None,
            "case_manifest_sha256": None,
            "valid_row_count": 2,
            "excluded_rows": [],
            "shots": [
                {
                    "clip_uid": j.clip_uid,
                    "source_index": i,
                    "source_row_sha256": "b" * 64,
                    "source_video_id": "synthetic",
                    "shot_index": i,
                    "clip_display_path": j.clip_display_path,
                    "video_path": j.target_video_path,
                    "video_sha256": j.target_video_sha256,
                    "duration_seconds": j.target_duration_seconds,
                }
                for i, j in enumerate([job, other])
            ],
        },
        "audio_production_root": str(production),
        "audio_shadow_run_id": "synthetic",
        "t2va_run_id": "qa-test",
        "source_hashes": {"/synthetic/shots.jsonl": "a" * 64},
        "selection_mode": "all",
        "sample_seed": None,
        "clip_uids": ["clip", "second"],
        "jobs": [j.model_dump(mode="json") for j in [job, other]],
        "backend": config(tmp_path).provenance().model_dump(mode="json"),
    }
    inventory = t2va.T2VAInventory(
        **values, inventory_fingerprint=t2va.fingerprint(values)
    )
    monkeypatch.setattr(t2va, "_check_sources", lambda _: None)
    t2va.run_t2va_shadow(
        inventory,
        T2VAMimoBackend(
            config(tmp_path),
            client=Client([draft_for(j).model_dump_json() for j in inventory.jobs]),
        ),
    )
    return t2va.t2va_root(production, "qa-test")


def test_static_payload_safe_and_inputs_immutable(published):
    before = {p: p.read_bytes() for p in published.rglob("*") if p.is_file()}
    page = build_t2va_qa(published)
    data = page.with_name("data.json").read_bytes()
    payload = json.loads(data)
    assert payload["schema_version"] == "r2v.h3.t2va_qa.2"
    speech = payload["cases"][0]["speech"][0]
    assert speech["segment_id"] == "segment_1"
    assert speech["source_speaker_cluster"] == "cluster_1"
    assert speech["assignment"]["speaker_id"] == "S1"
    assert speech["assignment"]["speech_presentation"] == "onscreen_spoken"
    assert speech["model_lead_in"] == "A person says"
    assert speech["text"] == "你好。"
    assert speech["rendered_speech"] == "A person says (S1) <d>[Chinese] 你好。</d>"
    assert "<script>window.injected=true</script>" not in page.read_text()
    assert "\\u003c/script" in page.read_text()
    build_t2va_qa(published, overwrite=True)
    assert page.with_name("data.json").read_bytes() == data
    assert all(p.read_bytes() == value for p, value in before.items())
    with pytest.raises(FileExistsError):
        build_t2va_qa(published)


def test_qa_rejects_stale_core_and_media_escape(published, tmp_path):
    with pytest.raises(ValueError):
        build_t2va_qa(
            published, media_root=published, media_base_url="https://media.invalid/"
        )
    with pytest.raises(ValueError, match="HTTP"):
        build_t2va_qa(
            published, media_root=tmp_path, media_base_url="javascript:alert(1)"
        )
    core = published / "core/clip.json"
    core.write_text(core.read_text() + "\n")
    with pytest.raises(ValueError, match="provenance differs"):
        build_t2va_qa(published)


def test_static_browser_desktop_mobile(published, tmp_path):
    node = os.environ.get("NODE_BINARY") or shutil.which("node")
    playwright = os.environ.get("QA_PLAYWRIGHT_MODULE")
    if not node or not playwright:
        pytest.skip("optional local Playwright unavailable")
    page = build_t2va_qa(published)
    script = tmp_path / "qa.cjs"
    script.write_text(r"""
const assert=require('assert');
const {chromium}=require(process.argv[2]);
(async()=>{
 const browser=await chromium.launch({headless:true,channel:process.env.QA_BROWSER_CHANNEL});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1000}});
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.route(/^https?:/,route=>{throw Error('No network permitted: '+route.request().url());});
  await page.goto(process.argv[3]);
  assert.strictEqual(await page.locator('#uid').textContent(),'clip');
  assert.strictEqual(await page.locator('#error').textContent(),'');
  assert.strictEqual(await page.evaluate(()=>window.injected),undefined);
  assert.strictEqual(await page.locator('#speech tr').count(),2);
  assert.strictEqual(await page.locator('#speech tr').first().locator('td').count(),7);
  assert.strictEqual(await page.locator('#speech tr').first().locator('td').nth(5).textContent(),'A person says');
  assert.strictEqual(await page.locator('#speech tr').first().locator('td').nth(6).textContent(),'A person says (S1) <d>[Chinese] 你好。</d>');
  assert((await page.locator('#prompt').textContent()).startsWith('integrated_multimodal_description:'));
  await page.waitForFunction(()=>document.querySelector('video').readyState>=2);
  await page.evaluate(()=>{const v=document.querySelector('video');v.muted=true;return v.play();});
  await page.waitForFunction(()=>document.querySelector('video').currentTime>0);
  await page.evaluate(()=>document.querySelector('video').pause());
  await page.screenshot({path:process.argv[4]+'/t2va-desktop.png',fullPage:true});
  await page.locator('#next').click();
  assert.strictEqual(await page.locator('#uid').textContent(),'second');
  assert.strictEqual(await page.locator('#speech tr').count(),0);
  assert.strictEqual(await page.locator('#next').isDisabled(),true);
  await page.locator('#prev').click();
  await page.setViewportSize({width:390,height:844});
  await page.screenshot({path:process.argv[4]+'/t2va-mobile.png',fullPage:true});
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
  assert.deepStrictEqual(errors,[]);
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1);});
""")
    screenshots = Path(os.environ.get("T2VA_SCREENSHOT_DIR", str(tmp_path)))
    screenshots.mkdir(exist_ok=True, parents=True)
    subprocess.run(
        [node, str(script), playwright, page.as_uri(), str(screenshots)],
        check=True,
        text=True,
        capture_output=True,
    )


def test_qa_summary_reconciles(published):
    page = build_t2va_qa(published)
    data = json.loads(page.with_name("data.json").read_text())
    assert data["summary"]["record_count"] == len(data["cases"]) == 2
    assert data["summary"]["ready_count"] == 2
    assert data["summary"]["model_call_count"] == 2
