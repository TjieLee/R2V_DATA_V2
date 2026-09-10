from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("soundfile")

from r2v_data_v2.h3 import audio_reuse_materializer as reuse_product
from r2v_data_v2.h3 import frame_conditioning_projection as frame
from r2v_data_v2.h3.audio_reuse_finalizer import finalize_audio_reuse_shadow
from r2v_data_v2.h3.audio_reuse_materializer import _hash
from r2v_data_v2.h3.audio_shadow_qa_variants import finalize_registry
from tests import test_h3_audio_reuse_finalizer as final_fixture
from tests import test_h3_sam_audio_stem_shadow as sam_fixture
from tools.materialize_h3_frame_conditioned_products import main


@pytest.fixture
def ffmpeg():
    binary = os.environ.get("FFMPEG_BINARY") or shutil.which("ffmpeg")
    if not binary:
        pytest.skip("FFmpeg required for exact decoded-frame regression")
    return binary


def _fixture(tmp_path, monkeypatch, ffmpeg):
    original = sam_fixture._canonical_fixture
    pixels = [np.full((24, 32, 3), color, dtype=np.uint8) for color in ((201, 12, 31), (20, 211, 42), (32, 51, 223))]

    def canonical(root):
        manifest, record, video = original(root)
        inputs = root / "synthetic-frames"
        inputs.mkdir()
        for i, array in enumerate(pixels):
            Image.fromarray(array).save(inputs / f"{i}.png")
        subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-y", "-framerate", "3", "-i", str(inputs / "%d.png"),
                        "-frames:v", "3", "-c:v", "libx264rgb", "-crf", "0", "-preset", "ultrafast", str(video)],
                       check=True, capture_output=True)
        record.target_video_sha256 = frame.sha256_file(video)
        manifest.write_text(record.model_dump_json() + "\n")
        return manifest, record, video

    monkeypatch.setattr(sam_fixture, "_canonical_fixture", canonical)
    args, shadow = final_fixture._fixture(tmp_path, monkeypatch)
    finalize_audio_reuse_shadow(**args)
    return args, shadow, pixels


def _prompt(suffix=""):
    return (
        "subject_definitions:\n<Picture 3> old standalone row\n"
        "<Subject 1> is the person in <Picture 3> and <Picture 2>.\n<Audio 1> voice of (S1).\n"
        "\nsummary:\n[reference generation" + suffix + "] A person from <Picture 4> stands."
        "\n\nretention_analysis:\n<Picture 4>: old standalone row\n"
        "<Subject 1> (appears in [Shot 1]): fully_preserved - From <Picture 5>.\n"
        "<Audio 1>: partially_copy - Keep the speech."
        "\n\ndetailed_description:\nA static shot.\n[Shot 1] <Subject 1> from <Picture 2> (S1) says, <d>[Chinese] 你好。</d>"
        "\n\noverall_soundscape:\nRoom tone near <Picture 6>."
        "\n\nnon_diegetic_music:\nMusic around <Picture 7>."
    )


@pytest.mark.parametrize("mode", frame.DERIVED_MODES)
@pytest.mark.parametrize("suffix", ["", " + audio reference", " + audio reuse", " + audio reference + audio reuse"])
def test_exact_text_projection_and_field_ownership(mode, suffix):
    source = _prompt(suffix)
    result = frame.project_prompt(source, mode, 8.125)
    sections = frame.split_sections(result)
    assert list(sections) == list(frame.SECTIONS)
    assert result.startswith("subject_definitions:\n<Picture 1> is the ")
    assert sections["summary"].startswith("[" + frame.VISUAL_TASKS[mode] + suffix + "] For the target video,")
    assert "For the target video" not in sections["subject_definitions"]
    assert sections["summary"].endswith("A person from <Picture 1> stands.")
    assert "old standalone row" not in result
    assert "<Subject 1> is the person in <Picture 1> and <Picture 1>." in result
    assert "<Subject 1> (appears in [Shot 1]): fully_preserved - From <Picture 1>." in result
    assert "<Audio 1>: partially_copy - Keep the speech." in sections["retention_analysis"]
    assert re.findall(r"<d>.*?</d>", result) == re.findall(r"<d>.*?</d>", source)
    assert result.count("(S1)") == source.count("(S1)")
    assert sections["overall_soundscape"] == "Room tone near <Picture 1>."
    assert sections["non_diegetic_music"] == "Music around <Picture 1>."
    labels = set(re.findall(r"<Picture \d+>", result))
    assert labels == ({"<Picture 1>", "<Picture 2>"} if mode == "first_last_frame" else {"<Picture 1>"})
    roles = [(1, "first", "opening"), (2, "last", "ending")] if mode == "first_last_frame" else [(1, "first", "opening")] if mode == "first_frame" else [(1, "last", "ending")]
    for i, role, edge in roles:
        assert f"<Picture {i}> is the {role} frame of [Shot 1], serving as the exact {edge}-frame reference for the target video." in sections["subject_definitions"]
        assert f"<Picture {i}> ([Shot 1] {role} frame): fully_preserved - <Picture {i}> is fully preserved as the exact {edge} frame of the target video." in sections["retention_analysis"]
    if mode == "first_last_frame":
        assert "For the target video, <Picture 1> (from [Shot 1]) is fully referenced at 0.00 seconds, and <Picture 2> (from [Shot 1]) is fully referenced at 8.12 seconds." in sections["summary"]
    else:
        time = "0.00" if mode == "first_frame" else "8.12"
        assert f"For the target video, at {time} seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced." in sections["summary"]
    original_detail = frame.split_sections(source)["detailed_description"].replace("<Picture 2>", "<Picture 1>")
    ending = f"The target video ends on the exact visual state established by <Picture {2 if mode == 'first_last_frame' else 1}>."
    assert sections["detailed_description"] == original_detail + (" " + ending if mode != "first_frame" else "")


@pytest.mark.parametrize("change", [
    lambda p: "alignment\n" + p,
    lambda p: p.replace("[Shot 1] <Subject", "[Shot 2] <Subject"),
    lambda p: p.replace("[Shot 1] <Subject", "[Shot 1] [Shot 3] <Subject"),
    lambda p: p.replace("[reference generation]", "[keyframe completion]"),
    lambda p: p.replace("summary:\n", "summary:\n\nsummary:\n"),
    lambda p: p.replace("你好。", "<Picture 9>"),
    lambda p: p.replace("Keep the speech.", "Keep <Picture 4>."),
])
def test_ambiguous_or_protected_structure_fails_closed(change):
    with pytest.raises(ValueError):
        frame.project_prompt(change(_prompt()), "first_frame", 8)


def test_end_to_end_exact_frames_3n_audio_and_qa_4n(tmp_path, monkeypatch, ffmpeg):
    args, shadow, pixels = _fixture(tmp_path, monkeypatch, ffmpeg)
    before = {p: frame.sha256_file(p) for p in tmp_path.rglob("*") if p.is_file()}
    calls = []
    original = frame.extract_frames

    def capture(job, *a, **kw):
        calls.append(job.clip_uid)
        return original(job, *a, **kw)

    monkeypatch.setattr(frame, "extract_frames", capture)
    result = main(["--audio-production-root", str(args["audio_production_root"]),
                   "--shadow-run-id", args["shadow_run_id"], "--ffmpeg", ffmpeg])
    assert (result["source_ready_product_count"], result["derived_product_count"]) == (6, 18)
    assert calls == ["clip-z", "clip-m"]
    assert result["model_call_count"] == 0 and result["production_artifacts_modified"] is False
    root = shadow / frame.FRAME_STAGE
    assert not (root / "frames/clip-a").exists()
    for uid in calls:
        np.testing.assert_array_equal(np.asarray(Image.open(root / "frames" / uid / "first.png")), pixels[0])
        np.testing.assert_array_equal(np.asarray(Image.open(root / "frames" / uid / "last.png")), pixels[-1])
    projected, _ = frame.load_frame_conditioned_products(root, shadow)
    sources, _ = frame.load_projection_sources(shadow)
    by_id = {s.product.sample_id: s for s in sources}
    for record in projected:
        source = by_id[record.source_product_sample_id]
        assert record.audio_references == source.product.audio_references
        assert [r.model_dump(mode="json") for r in record.audio_references] == [r.model_dump(mode="json") for r in source.product.audio_references]
        assert record.corrected_speech_segments == source.product.corrected_speech_segments
        assert record.warnings == source.product.warnings
        for subject, original_subject in zip(record.subjects, source.subjects, strict=True):
            assert subject.source_picture_labels == ["<Picture 1>"]
            assert subject.model_dump(exclude={"source_picture_labels"}) == original_subject.model_dump(exclude={"source_picture_labels"})
    assert before == {p: frame.sha256_file(p) for p in before}
    with pytest.raises(FileExistsError):
        frame.materialize_frame_conditioned_products(audio_production_root=args["audio_production_root"], shadow_run_id=args["shadow_run_id"], ffmpeg=ffmpeg)
    final_fixture.qa_fixture.qa.build_audio_shadow_qa(**{k: v for k, v in args.items() if k != "allow_unverified"})
    data = json.loads((shadow / "qa/data.json").read_text())
    count = 0
    for clip in data["clips"]:
        variants = clip["final_h3"]["variants"]
        family = [v for v in variants if v["review_family"] == "final_4way"]
        count += len(family)
        assert variants[:len(family)] == family
        for start in range(0, len(family), 4):
            group = family[start:start + 4]
            assert [v["visual_reference_mode"] for v in group] == list(frame.VISUAL_TASKS)
            assert len({v["source_product_sample_id"] for v in group}) == 1
            for derived in group[1:]:
                assert derived["references"]["audios"] == group[0]["references"]["audios"]
                assert all(s["source_picture_labels"] == ["<Picture 1>"] for s in derived["references"]["subjects"])
                assert all(p["frame_role"] in {"first_frame", "last_frame"} for p in derived["references"]["pictures"])
    assert count == 24


@pytest.mark.parametrize("target", ["source_product", "prepared", "video", "frame", "projected", "audio"])
def test_stale_provenance_or_derived_tampering_fails(tmp_path, monkeypatch, ffmpeg, target):
    args, shadow, _ = _fixture(tmp_path, monkeypatch, ffmpeg)
    root = shadow / frame.FRAME_STAGE
    frame.materialize_frame_conditioned_products(audio_production_root=args["audio_production_root"], shadow_run_id=args["shadow_run_id"], ffmpeg=ffmpeg)
    path = {"source_product": shadow / frame.PRODUCTS_STAGE / "records.jsonl",
            "prepared": shadow / frame.PREPARED_STAGE / "records.jsonl",
            "video": args["audio_production_root"] / "target.mp4",
            "frame": root / "frames/clip-z/last.png"}.get(target)
    if path:
        with path.open("ab") as stream:
            stream.write(b"changed")
    else:
        rows = [json.loads(line) for line in (root / "records.jsonl").read_text().splitlines()]
        row = next(r for r in rows if r["audio_references"])
        if target == "audio":
            row["audio_references"] = []
        else:
            row["rendered_h3_prompt"] = row["rendered_h3_prompt"].replace("exact opening", "invented opening")
        row["record_fingerprint"] = _hash({k: v for k, v in row.items() if k != "record_fingerprint"})
        (root / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        summary = json.loads((root / "summary.json").read_text())
        summary["records_sha256"] = frame.sha256_file(root / "records.jsonl")
        (root / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        frame.load_frame_conditioned_products(root, shadow)


@pytest.mark.parametrize("target", ["video", "frame"])
def test_input_mutation_during_extraction_cannot_publish(tmp_path, monkeypatch, ffmpeg, target):
    args, shadow, _ = _fixture(tmp_path, monkeypatch, ffmpeg)
    original = frame.extract_frames

    def mutate(job, *a, **kw):
        result = original(job, *a, **kw)
        path = Path(job.target_video_path) if target == "video" else a[0] / "last.png"
        with path.open("ab") as stream:
            stream.write(b"changed")
        return result

    monkeypatch.setattr(frame, "extract_frames", mutate)
    with pytest.raises(ValueError, match="changed|hash"):
        frame.materialize_frame_conditioned_products(audio_production_root=args["audio_production_root"], shadow_run_id=args["shadow_run_id"], ffmpeg=ffmpeg)
    assert not (shadow / frame.FRAME_STAGE).exists()
    assert not list(shadow.glob(".frame-projection-*"))


def test_qa_rejects_symlinked_frame_even_with_identical_hash(tmp_path, monkeypatch, ffmpeg):
    args, shadow, _ = _fixture(tmp_path, monkeypatch, ffmpeg)
    frame.materialize_frame_conditioned_products(audio_production_root=args["audio_production_root"],
        shadow_run_id=args["shadow_run_id"], ffmpeg=ffmpeg)
    root = shadow / frame.FRAME_STAGE
    image = root / "frames/clip-z/last.png"
    original = image.with_name("original.png")
    image.rename(original)
    image.symlink_to(original)
    with pytest.raises(ValueError, match="symlink"):
        frame.load_frame_conditioned_products(root, shadow)


def test_registry_never_deduplicates_across_modes_or_source_products():
    variants = [{"sample_id": f"{sid}/{mode}", "source_product_sample_id": sid, "visual_reference_mode": mode,
                 "conditioning_variant": "visual_only", "text": "same", "status": "ready", "references": {},
                 "publication_status": "published", "source_kind": "audio_reuse_product" if mode == "reference" else "frame_conditioned_product"}
                for sid in ("clip/z", "clip/a") for mode in reversed(frame.VISUAL_TASKS)]
    result, _ = finalize_registry(variants, _hash)
    assert len(result) == 8
    assert [v["visual_reference_mode"] for v in result] == list(frame.VISUAL_TASKS) * 2
    assert Counter(v["source_product_sample_id"] for v in result) == {"clip/z": 4, "clip/a": 4}


@pytest.mark.parametrize("music_only", [False, True])
def test_phantom_speech_products_never_enter_projection_or_qa(tmp_path, monkeypatch, ffmpeg, music_only):
    original = reuse_product.select_reuse_audio

    def select(*a, **kw):
        refs, warnings = original(*a, **kw)
        refs = [r for r in refs if music_only and r.contract.kind == "music_reuse"]
        return [r.model_copy(update={"contract": r.contract.model_copy(update={
            "audio_index": 1, "audio_label": "<Audio 1>"})}) for r in refs], warnings

    monkeypatch.setattr(reuse_product, "select_reuse_audio", select)
    args, shadow, _ = _fixture(tmp_path, monkeypatch, ffmpeg)
    sources, _ = frame.load_projection_sources(shadow)
    assert sources and all(s.product.conditioning_variant != "target_speech_reuse" for s in sources)
    summary = frame.materialize_frame_conditioned_products(audio_production_root=args["audio_production_root"],
        shadow_run_id=args["shadow_run_id"], ffmpeg=ffmpeg)
    assert summary.derived_product_count == 3 * len(sources)
    final_fixture.qa_fixture.qa.build_audio_shadow_qa(**{k: v for k, v in args.items() if k != "allow_unverified"})
    data = json.loads((shadow / "qa/data.json").read_text())
    families = {}
    for clip in data["clips"]:
        for variant in clip["final_h3"]["variants"]:
            if variant["review_family"] == "final_4way":
                assert variant["conditioning_variant"] != "target_speech_reuse"
                families.setdefault(variant["source_product_sample_id"], []).append(variant["visual_reference_mode"])
    assert len(families) == len(sources)
    assert all(modes == list(frame.VISUAL_TASKS) for modes in families.values())


def test_source_must_match_frozen_deterministic_materialization(tmp_path, monkeypatch, ffmpeg):
    _, shadow, _ = _fixture(tmp_path, monkeypatch, ffmpeg)
    path = shadow / frame.PRODUCTS_STAGE / "records.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    row = next(r for r in rows if r["status"] == "ready")
    row["rendered_h3_prompt"] = row["rendered_h3_prompt"].replace("subject_definitions:\n", "subject_definitions:\nInvented prose.\n")
    row["record_fingerprint"] = _hash({k: v for k, v in row.items() if k != "record_fingerprint"})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="deterministic reconstruction"):
        frame.load_projection_sources(shadow)


@pytest.mark.parametrize("where", ["outside", "source", "symlink"])
def test_projection_output_ownership(tmp_path, monkeypatch, ffmpeg, where):
    args, shadow, _ = _fixture(tmp_path, monkeypatch, ffmpeg)
    output = tmp_path / "outside" if where == "outside" else shadow / frame.PRODUCTS_STAGE
    if where == "symlink":
        output = shadow / "redirected"
        output.symlink_to(tmp_path / "absent", target_is_directory=True)
    with pytest.raises((ValueError, FileExistsError)):
        frame.materialize_frame_conditioned_products(audio_production_root=args["audio_production_root"],
            shadow_run_id=args["shadow_run_id"], ffmpeg=ffmpeg, output_root=output)


def test_frame_modes_offline_browser_desktop_mobile(tmp_path, monkeypatch, ffmpeg):
    node = os.environ.get("NODE_BINARY") or shutil.which("node")
    playwright = os.environ.get("QA_PLAYWRIGHT_MODULE")
    if not node or not playwright:
        pytest.skip("optional Playwright not configured")
    args, shadow, _ = _fixture(tmp_path, monkeypatch, ffmpeg)
    frame.materialize_frame_conditioned_products(audio_production_root=args["audio_production_root"], shadow_run_id=args["shadow_run_id"], ffmpeg=ffmpeg)
    final_fixture.qa_fixture.qa.build_audio_shadow_qa(**{k: v for k, v in args.items() if k != "allow_unverified"})
    script = tmp_path / "frames-browser.cjs"
    script.write_text(r'''
const fs=require('fs'), path=require('path'), assert=require('assert');
const {chromium}=require(process.argv[2]);
(async()=>{
  const browser=await chromium.launch({headless:true,channel:process.env.QA_BROWSER_CHANNEL});
  try {
    const page=await browser.newPage({viewport:{width:1440,height:1000}});
    const errors=[]; page.on('pageerror',e=>errors.push(e.message));
    const root=process.argv[3];
    await page.route('**/*',async route=>{
      const u=new URL(route.request().url()); assert.strictEqual(u.hostname,'qa.invalid');
      const p=path.join(root,u.pathname);
      if(!fs.existsSync(p))return route.fulfill({status:404,body:'fixture only'});
      const ext=path.extname(p), types={'.html':'text/html','.json':'application/json','.png':'image/png','.flac':'audio/flac','.wav':'audio/wav','.mp4':'video/mp4'};
      return route.fulfill({body:fs.readFileSync(p),contentType:types[ext]||'application/octet-stream'});
    });
    await page.goto('http://qa.invalid/review.html');
    await page.locator('#main').waitFor({state:'visible'});
    const dataset=JSON.parse(fs.readFileSync(path.join(root,'data.json'),'utf8'));
    const variants=dataset.clips[0].final_h3.variants;
    const reference=variants.findIndex(v=>v.conditioning_variant==='target_speech_reuse'&&v.visual_reference_mode==='reference'&&v.review_family==='final_4way');
    assert(reference>=0);
    for(const [width,name] of [[1440,'desktop'],[390,'mobile']]) {
      await page.setViewportSize({width,height:width===390?844:1000});
      for(let i=0;i<4;i++) {
        await page.locator('#final-variant').selectOption(String(reference+i));
        const v=variants[reference+i];
        assert.match(await page.locator('#final-variant option:checked').textContent(),/^FINAL \|/);
        assert.strictEqual(await page.locator('#final-text').textContent(),v.text);
        assert.match(await page.locator('#visual-mode').textContent(),new RegExp(i===0?'reference':v.visual_task));
        assert.strictEqual(await page.locator('#references .ref-card').count(),i===3?2:1);
        assert.strictEqual(await page.locator('#audio-references audio').count(),3);
        await page.locator('#references').scrollIntoViewIfNeeded();
        await page.waitForFunction(()=>[...document.querySelectorAll('#references img')].every(im=>im.complete&&im.naturalWidth>0));
        if(i>0) {
          assert.match(await page.locator('#references').textContent(),/Target video:/);
          assert.match(await page.locator('#references').textContent(),i===2?/last frame/:/first frame/);
          const sizes=await page.locator('#references img').evaluateAll(imgs=>imgs.map(im=>[im.naturalWidth,im.naturalHeight]));
          sizes.forEach(size=>assert.deepStrictEqual(size,[32,24]));
          assert.match(await page.locator('#subject-graph').textContent(),/Source: <Picture 1>/);
        }
        await page.evaluate(()=>{window.oldPlayer=document.querySelector('#conditioning audio');window.pausedOld=false;const p=oldPlayer.pause.bind(oldPlayer);oldPlayer.pause=()=>{pausedOld=true;p();};});
        await page.screenshot({path:path.join(process.argv[4],`frame-${name}-${i}.png`),fullPage:true});
        assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
        await page.locator('#final-variant').selectOption(String(reference+(i+1)%4));
        assert(await page.evaluate(()=>pausedOld&&!oldPlayer.isConnected));
      }
    }
    assert.deepStrictEqual(errors,[]);
  } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
''')
    result = subprocess.run([node, str(script), playwright, str(shadow / "qa"), str(tmp_path)],
                            capture_output=True, text=True, timeout=90, check=False)
    assert result.returncode == 0, result.stderr
