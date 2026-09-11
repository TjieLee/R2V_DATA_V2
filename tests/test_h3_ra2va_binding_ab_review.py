import json
import os
import subprocess
from types import SimpleNamespace

import pytest
from PIL import Image

from r2v_data_v2.h3 import ra2va_binding_ab_review as ab
from r2v_data_v2.h3.visual_production_source import NormalizedVisualReference


def _record(uid, *, entity="e1", status="ready", annotation=True, extra=False):
    segment = {"segment_id": "segment_0001", "primary_speaker_group": "g1",
               "binding_status": "visible_entity" if entity else "offscreen",
               "entity_id": entity, "speech_presentation": "onscreen_spoken" if entity else "offscreen_spoken",
               "evidence_codes": ["av_correspondence"]}
    segments = [segment, {**segment, "segment_id": "segment_0002"}] if extra else [segment]
    raw = {
        "schema_version": "r2v.h3.mimo25_stem_reconcile.14", "clip_uid": uid, "status": status,
        "failure_code": "direct_dialogue_speaker_marker_missing" if status == "failed" else None,
        "failure_reason": "Frozen <script>not executable</script>" if status == "failed" else None,
        "failure_issues": [{"code": "marker_missing", "message": "Original diagnostic", "path": "dialogue_2"}] if status == "failed" else [],
        "model_call_count": 4,
        "annotation": {
            "av_grounding": {"segment_groundings": segments},
            "audio_observation": {"segment_decisions": [{"segment_id": s["segment_id"], "primary_speaker_group": "g1"} for s in segments]},
            "h3_semantics": {"summary": "Frozen summary", "shot1_caption":
                "<Subject 1> (S1) says, <d>[Chinese] Exact frozen words.</d> She continues, <d>[Chinese] More words.</d>"},
        } if annotation else None,
    }
    return {**raw, "record_fingerprint": ab._hash(raw)}


@pytest.mark.parametrize("group", ["g2", None])
def test_display_preserves_frozen_grounding_group(group):
    raw = _record("clip", status="failed")
    raw["annotation"]["av_grounding"]["segment_groundings"][0]["primary_speaker_group"] = group
    row = ab._side(ab._Record.model_validate(raw))["segments"]["segment_0001"]
    assert row["primary_speaker_group"] == group
    assert row["grounding_present"] and row["decision_present"]


def test_decision_only_does_not_supply_display_group():
    raw = _record("clip", status="failed")
    raw["annotation"]["av_grounding"]["segment_groundings"] = []
    row = ab._side(ab._Record.model_validate(raw))["segments"]["segment_0001"]
    assert row["primary_speaker_group"] is None
    assert not row["grounding_present"] and row["decision_present"]


def test_grounding_only_preserves_display_group():
    raw = _record("clip", status="failed")
    raw["annotation"]["audio_observation"]["segment_decisions"] = []
    row = ab._side(ab._Record.model_validate(raw))["segments"]["segment_0001"]
    assert row["primary_speaker_group"] == "g1"
    assert row["grounding_present"] and not row["decision_present"]


@pytest.fixture
def source(tmp_path, monkeypatch):
    visual, runs, legacy, new = [tmp_path / n for n in ("visual", "runs", "legacy", "new")]
    for root in (visual, runs, legacy, new):
        root.mkdir()
    image = visual / "reference.png"
    Image.new("RGB", (120, 140), (45, 140, 115)).save(image)
    video = visual / "target.mp4"
    video.write_bytes(b"synthetic MP4; no inference")
    (visual / "samples.jsonl").write_text("{}\n")
    order = [f"clip-{i:02}" for i in range(19, -1, -1)]
    old_rows = [_record(uid) for uid in order]
    new_rows = [_record(uid, entity="e2" if i == 1 else None if i == 2 else "e1",
                        extra=i == 3, status="failed" if i in (4, 5) else "ready", annotation=i != 5)
                for i, uid in enumerate(order)]
    for root, rows in ((legacy, old_rows[::-1]), (new, new_rows[5:]+new_rows[:5])):
        (root / "records.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    manifest = tmp_path / "cases.json"
    manifest.write_text(json.dumps({"clip_uids": order}))
    reference = NormalizedVisualReference(
        image_id="image_1", image_index=1, kind="subject", entity_id="e1", image_path="reference.png",
        artifact_path=str(image), source_frame_index=0, scope="full", synthetic=False,
    )
    monkeypatch.setattr(ab, "load_visual_production_inventory", lambda **kw: SimpleNamespace(canonical_clips=[
        SimpleNamespace(identity=SimpleNamespace(clip_uid=uid),
                        sample=SimpleNamespace(target_video=str(video), references=[reference])) for uid in order
    ]))
    from r2v_data_v2.h3.mimo25_backend import OpenAIMimo25Backend

    def forbidden(*a, **kw):
        raise AssertionError("review must never construct model backend")

    monkeypatch.setattr(OpenAIMimo25Backend, "__init__", forbidden)
    return {"visual_production_root": visual, "visual_runs_root": runs, "case_manifest": manifest,
            "legacy_mimo_root": legacy, "no_lrasd_mimo_root": new, "output_root": tmp_path / "review"}


def test_uncapped_frozen_visual_context(source, monkeypatch):
    from r2v_data_v2.h3 import qwen38_h3_recaption as recaption

    def forbidden(*args, **kwargs):
        raise AssertionError("review must not construct conditioning contracts")

    monkeypatch.setattr(recaption, "build_reference_contract", forbidden)
    monkeypatch.setattr(recaption, "RecaptionReferenceContract", forbidden)
    inventory = ab.load_visual_production_inventory()
    base = inventory.canonical_clips[0].sample.references[0]
    specs = [
        ("object", "e3"), ("subject", "e1"), ("group", "e2"),
        ("subject", "e1"), ("object", "e3"), ("group", "e2"),
        ("subject", "e1"), ("subject", "e1"),
        ("attribute", None), ("attribute", None),
        ("background", None), ("background", None),
    ]
    refs = [base.model_copy(update={
        "image_index": i, "image_id": f"image_{i}", "kind": kind, "entity_id": entity,
        "attribute_id": f"a{i}" if kind == "attribute" else None,
        "attribute_type": "hair" if kind == "attribute" else None,
        "owner_entity_id": "e1" if kind == "attribute" else None,
    }) for i, (kind, entity) in enumerate(specs, 1)]
    for clip in inventory.canonical_clips:
        clip.sample.references = refs
    monkeypatch.setattr(ab, "load_visual_production_inventory", lambda **kw: inventory)
    ab.build_review(**source)
    data = json.loads((source["output_root"] / "data.json").read_text())
    case = data["cases"][0]
    assert len(case["references"]) == 12
    assert [p["picture_label"] for p in case["references"]] == [f"<Picture {i}>" for i in range(1, 13)]
    subjects = case["subjects"]
    assert [s["entity_id"] for s in subjects[:3]] == ["e3", "e1", "e2"]
    assert subjects[0]["source_picture_labels"] == ["<Picture 1>", "<Picture 5>"]
    assert subjects[1]["source_picture_labels"] == ["<Picture 2>", "<Picture 4>", "<Picture 7>", "<Picture 8>"]
    assert [s["attribute_id"] for s in subjects[3:5]] == ["a9", "a10"]
    assert all(s["owner_entity_id"] == "e1" and s["attribute_type"] == "hair" for s in subjects[3:5])
    assert subjects[5]["kind"] == "background"
    assert subjects[5]["source_picture_labels"] == ["<Picture 11>", "<Picture 12>"]
    assert [s["subject_label"] for s in subjects] == [f"<Subject {i}>" for i in range(1, 7)]
    assert case["references"][8]["attribute_id"] == "a9"
    # Labels come from frozen indices, not enumeration or a first-nine selection.
    pictures, _ = ab._visual_context([refs[11], refs[1]], lambda path: path)
    assert [p["picture_label"] for p in pictures] == ["<Picture 2>", "<Picture 12>"]
    assert all(p["media"] == base.artifact_path for p in pictures)


def test_pairing_summary_failed_annotations_and_isolation(source, tmp_path):
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    summary = ab.build_review(**source)
    root = source["output_root"]
    data = json.loads((root / "data.json").read_text())
    assert [c["clip_uid"] for c in data["cases"]] == json.loads(source["case_manifest"].read_text())["clip_uids"]
    assert summary["clip_count"] == 20
    assert (summary["legacy_ready_count"], summary["no_lrasd_ready_count"], summary["no_lrasd_failed_count"]) == (20, 18, 2)
    assert summary["visible_entity_transitions"] == {"same_entity_id": 17, "changed_entity_id": 1}
    assert summary["transition_counts"]["visible_entity -> offscreen"] == 1
    assert summary["legacy_grounding_count"] == summary["no_lrasd_grounding_count"] == 20
    assert summary["no_lrasd_offscreen_count"] == summary["no_lrasd_null_entity_count"] == 1
    assert data["cases"][1]["aligned_segments"][0]["changed_fields"] == ["entity_id"]
    unmatched = data["cases"][3]
    assert unmatched["segment_inventory_diff"]
    assert unmatched["aligned_segments"][1]["old"] is None
    assert data["cases"][4]["new"]["annotation_present"]
    assert "<d>" in data["cases"][4]["new"]["shot1_caption"]
    assert "She continues" in data["cases"][4]["new"]["dialogue"][1]
    assert not data["cases"][5]["new"]["annotation_present"]
    assert data["cases"][5]["new"]["failure_issues"]
    assert set(data["labels"]) == {"SAME", "NEW_BETTER", "NEW_WORSE", "BOTH_WRONG", "FORMAT_ONLY"}
    for case in data["cases"]:
        for media in [case["target_video"], *(r["media"] for r in case["references"])]:
            assert media.startswith("media/") and (root / media).is_file()
            assert not (root / media).is_symlink()
    assert len(list((root / "media").iterdir())) == 2
    assert all(p.read_bytes() == contents for p, contents in before.items())
    assert summary["model_calls_invoked"] == 0 and not summary["production_artifacts_modified"]
    second = tmp_path / "review2"
    assert ab.build_review(**{**source, "output_root": second}) == summary
    assert (second / "data.json").read_bytes() == (root / "data.json").read_bytes()


def _freeze_projected_context(source, *, root_key="no_lrasd_mimo_root", selected=(1, 3), promotion=False):
    root = source[root_key]
    image = source["visual_production_root"] / "reference.png"
    video = source["visual_production_root"] / "target.mp4"
    images, subjects = [], []
    for i, original in enumerate(selected, 1):
        entity = i == 1 or (len(selected) == 4 and i == 2)
        owner = "e2" if i == 4 else "e1"
        images.append({
            "image_index": i, "picture_label": f"<Picture {i}>", "source_image_index": original,
            "source_image_id": f"image_{original}", "source_image_label": f"<Image {original}>",
            "kind": "subject" if entity else "attribute", "entity_id": f"e{i}" if entity else None,
            "owner_entity_id": None if entity else owner, "attribute_id": None if entity else f"a{original}",
            "attribute_type": None if entity else "face" if len(selected) == 2 else "clothing" if i == 3 else "hair",
            "image_artifact_path": str(image), "image_sha256": ab.sha256_file(image),
        })
        subjects.append({"subject_index": i, "subject_label": f"<Subject {i}>",
                         "kind": "entity" if entity else "attribute", "entity_id": images[-1]["entity_id"],
                         "attribute_id": images[-1]["attribute_id"], "owner_entity_id": images[-1]["owner_entity_id"],
                         "attribute_type": images[-1]["attribute_type"], "source_picture_labels": [f"<Picture {i}>"]})
    selection = {"selected_source_image_indexes": list(selected), "dropped_references": [{
        "source_image_index": 2 if len(selected) == 2 else 3,
        "kind": "attribute", "attribute_type": "hair", "owner_entity_id": "e1", "drop_reason": "hair_sampling_drop",
    }], "face_promotions": []}
    if promotion:
        selection["face_promotions"] = [{"source_image_index": selected[0], "entity_id": "e1", "replaced_source_image_indexes": [1]}]
        selection["dropped_references"].append({"source_image_index": 1, "kind": "subject", "entity_id": "e1", "drop_reason": "person_replaced_by_face"})
    rows = [json.loads(line) for line in (root / "records.jsonl").read_text().splitlines()]
    jobs = []
    for record in rows:
        job = {"clip_uid": record["clip_uid"], "target_video_path": str(video), "target_video_sha256": ab.sha256_file(video),
               "reference_images": images, "reference_subjects": subjects, "reference_selection": selection}
        job["request_fingerprint"] = ab._hash(job)
        jobs.append(job)
        record["source_job_fingerprint"] = job["request_fingerprint"]
        record["record_fingerprint"] = ab._hash({k: v for k, v in record.items() if k != "record_fingerprint"})
    (root / "records.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    path = root / "source_contract.json"
    path.write_text(json.dumps({"schema_version": "r2v.h3.mimo25_stem_source_contract.1", "jobs": jobs}))
    return path


@pytest.mark.parametrize("selected,promotion", [((1, 3), False), ((1, 2, 4, 5), False), ((3, 4), True)])
def test_actual_frozen_projection_not_recomputed(source, monkeypatch, selected, promotion):
    from r2v_data_v2.h3 import mimo25_av_reconcile as reconcile

    def forbidden(*args, **kwargs):
        raise AssertionError("never recompute reference selection")

    monkeypatch.setattr(reconcile, "build_mimo25_reference_inventory", forbidden)
    inventory = ab.load_visual_production_inventory()
    base = inventory.canonical_clips[0].sample.references[0]
    specs = [("subject", "e1", None), ("attribute", None, "hair"), ("attribute", None, "face")]
    if len(selected) == 4:
        specs = [("subject", "e1", None), ("subject", "e2", None),
                 ("attribute", None, "hair"), ("attribute", None, "clothing"), ("attribute", None, "hair")]
    elif promotion:
        specs.append(("attribute", None, "clothing"))
    refs = [base.model_copy(update={
        "image_index": i, "image_id": f"image_{i}", "kind": kind, "entity_id": entity,
        "attribute_id": f"a{i}" if attr else None, "attribute_type": attr,
        "owner_entity_id": ("e2" if i == 5 else "e1") if attr else None,
    }) for i, (kind, entity, attr) in enumerate(specs, 1)]
    for clip in inventory.canonical_clips:
        clip.sample.references = refs
    monkeypatch.setattr(ab, "load_visual_production_inventory", lambda **kw: inventory)
    path = _freeze_projected_context(source, selected=selected, promotion=promotion)
    before = path.read_bytes()
    ab.build_review(**source)
    data = json.loads((source["output_root"] / "data.json").read_text())
    case = data["cases"][0]
    actual = case["new_reference_context"]
    assert actual["available"] and not case["old_reference_context"]["available"]
    assert [r["source_image_index"] for r in actual["references"]] == list(selected)
    assert [r["picture_label"] for r in actual["references"]] == [f"<Picture {i}>" for i in range(1, len(selected)+1)]
    assert [s["subject_label"] for s in actual["subjects"]] == [f"<Subject {i}>" for i in range(1, len(selected)+1)]
    frozen = json.loads(path.read_text())["jobs"][0]
    assert actual["subjects"] == frozen["reference_subjects"]
    assert actual["reference_selection"] == frozen["reference_selection"]
    assert len(case["references"]) == len(specs)
    assert case["references"][-1]["source_image_index"] == len(specs)
    assert path.read_bytes() == before
    assert str(path) in data["source_hashes"]


def test_explicit_historical_contract_and_lineage_checks(source, tmp_path):
    path = _freeze_projected_context(source, root_key="legacy_mimo_root")
    external = tmp_path / "old-exact-contract.json"
    path.rename(external)
    ab.build_review(**source, legacy_source_contract=external)
    data = json.loads((source["output_root"] / "data.json").read_text())
    assert data["cases"][0]["old_reference_context"]["available"]
    payload = json.loads(external.read_text())
    payload["jobs"][0]["reference_selection"]["selected_source_image_indexes"] = [1, 9]
    external.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="fingerprint"):
        ab.build_review(**{**source, "output_root": tmp_path / "bad"}, legacy_source_contract=external)


@pytest.mark.parametrize("tamper", ["record", "media"])
def test_actual_reference_context_fails_closed(source, tamper):
    path = _freeze_projected_context(source)
    payload = json.loads(path.read_text())
    job = payload["jobs"][0]
    if tamper == "record":
        job["extra_frozen_field"] = "different job"
        job["request_fingerprint"] = ab._hash({k: v for k, v in job.items() if k != "request_fingerprint"})
        path.write_text(json.dumps(payload))
    else:
        (source["visual_production_root"] / "reference.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs from run record|image hash mismatch"):
        ab.build_review(**source)


@pytest.mark.parametrize("where", ["legacy_mimo_root", "no_lrasd_mimo_root", "visual_production_root", "visual_runs_root"])
def test_source_overlap_rejected(source, where):
    with pytest.raises(ValueError, match="overlaps"):
        ab.build_review(**{**source, "output_root": source[where] / "review"})


def test_existing_output_and_tampering_fail_closed(source):
    source["output_root"].mkdir()
    with pytest.raises(FileExistsError):
        ab.build_review(**source)
    source["output_root"].rmdir()
    path = source["legacy_mimo_root"] / "records.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["model_call_count"] = 9
    path.write_text("".join(json.dumps(r)+"\n" for r in rows))
    with pytest.raises(ValueError, match="fingerprint"):
        ab.build_review(**source)
    assert not source["output_root"].exists()


def test_browser_review_export_roundtrip(source, tmp_path):
    node, playwright = os.environ.get("NODE_BINARY"), os.environ.get("QA_PLAYWRIGHT_MODULE")
    if not node or not playwright:
        pytest.skip("optional local Playwright runtime unavailable")
    ffmpeg = os.environ.get("FFMPEG_BINARY")
    if not ffmpeg:
        pytest.skip("optional local FFmpeg runtime unavailable")
    subprocess.run([ffmpeg, "-y", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=12",
                    "-t", "1", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(source["visual_production_root"] / "target.mp4")],
                   check=True, capture_output=True)
    _freeze_projected_context(source, selected=(3, 4), promotion=True)
    ab.build_review(**source)
    script = tmp_path / "browser.cjs"
    script.write_text(r'''
const fs=require('fs'),path=require('path'),assert=require('assert');
const {chromium}=require(process.argv[2]);
(async()=>{const browser=await chromium.launch({headless:true,channel:process.env.QA_BROWSER_CHANNEL});
try{const page=await browser.newPage({viewport:{width:1400,height:1000},acceptDownloads:true});const errors=[];page.on('pageerror',e=>errors.push(e.message));
await page.route('**/*',async route=>{const u=new URL(route.request().url());assert.equal(u.hostname,'ab.invalid');const file=path.join(process.argv[3],u.pathname);if(!fs.existsSync(file))return route.fulfill({status:404});return route.fulfill({body:fs.readFileSync(file),contentType:file.endsWith('.html')?'text/html':file.endsWith('.json')?'application/json':file.endsWith('.png')?'image/png':'video/mp4'});});
await page.goto('http://ab.invalid/review.html');await page.locator('#review').waitFor({state:'visible'});
assert((await page.locator('#old-refs').innerText()).includes('OLD actual MiMo reference context: unavailable'));
const actual=await page.locator('#new-refs').innerText();assert(actual.includes('<Picture 2>'));assert(actual.includes('<Subject 2>'));assert(actual.includes('DROPPED: hair_sampling_drop'));assert(actual.includes('promoted to entity e1'));assert(actual.includes('replaced source person image indexes: [1]'));
assert((await page.locator('.media').innerText()).includes('SOURCE / frozen Visual context (not MiMo numbering)'));
await page.waitForFunction(()=>document.getElementById('video').readyState>=2&&[...document.querySelectorAll('#refs img')].every(i=>i.naturalWidth>0));
assert(await page.evaluate(()=>{const cells=document.querySelector('#grounding tr').children;return Math.abs(cells[1].getBoundingClientRect().width-cells[2].getBoundingClientRect().width)<2&&cells[2].getBoundingClientRect().height<400;}));
assert.equal(await page.locator('input[name=label]').count(),5);await page.locator('input[value=SAME]').check();await page.locator('#note').fill('Exact review note');
await page.reload();await page.locator('#review').waitFor({state:'visible'});assert(await page.locator('input[value=SAME]').isChecked());assert.equal(await page.locator('#note').inputValue(),'Exact review note');
await page.locator('#next').click();assert.equal(await page.locator('#grounding .changed').count(),2);await page.locator('input[value=NEW_WORSE]').check();
await page.locator('#next').click();await page.locator('#next').click();assert((await page.locator('#inventory').innerText()).includes('true'));assert((await page.locator('#grounding').innerText()).includes('Unmatched'));
await page.locator('#next').click();assert((await page.locator('#new').innerText()).includes('direct_dialogue_speaker_marker_missing'));assert((await page.locator('#new').innerText()).includes('She continues'));await page.locator('input[value=FORMAT_ONLY]').check();
await page.screenshot({path:process.argv[4]+'/ab-desktop.png',fullPage:true});
const download=page.waitForEvent('download');await page.locator('#export').click();const d=await download;const output=JSON.parse(fs.readFileSync(await d.path(),'utf8'));assert.equal(output.schema_version,'r2v.h3.ra2va_binding_ab_review.1');assert.equal(output.clip_count,20);assert.equal(output.reviews[0].label,'SAME');assert.equal(output.reviews[0].note,'Exact review note');assert.equal(output.reviews[1].label,'NEW_WORSE');assert.equal(output.reviews[4].label,'FORMAT_ONLY');assert.equal(output.reviews[2].label,null);assert(output.legacy_root&&output.no_lrasd_root);
await page.setViewportSize({width:390,height:844});await page.screenshot({path:process.argv[4]+'/ab-mobile.png',fullPage:true});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
assert.deepEqual(errors,[]);console.log('Browser export/persistence/desktop/mobile passed');}finally{await browser.close();}})().catch(e=>{console.error(e);process.exit(1)});
''')
    result = subprocess.run([node, str(script), playwright, str(source["output_root"]), str(tmp_path)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
