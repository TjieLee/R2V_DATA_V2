from __future__ import annotations

import hashlib
import json
import os
import shutil
import statistics
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal
from urllib.parse import quote

from pydantic import Field, model_validator

from r2v_data_v2.h3.audio_pairing import (
    H3_PAIR_POLICY_FACE_THRESHOLD,
    H3_PAIR_POLICY_VERSION,
    H3_PAIR_POLICY_VOICE_THRESHOLD,
    AudioPairingConfig,
    evaluate_pair_policy_v1,
    select_complete_donor_matching,
)
from r2v_data_v2.h3.audio_schemas import PairEvidence
from r2v_data_v2.h3.embedding_pilot import EmbeddingPilotOccurrence
from r2v_data_v2.h3.pair_policy_calibration import (
    PairPolicyEvidence,
    build_pair_policy_evidence,
)
from r2v_data_v2.h3.schemas import SchemaModel


class PairingPilotInPair(SchemaModel):
    pair_id: str
    clip_uid: str
    entity_occurrence_ids: list[str]
    primary_voice_paths: list[str]


class PairingPilotMapping(SchemaModel):
    accepted_pair_id: str
    subject_index: int = Field(gt=0)
    target_occurrence_id: str
    donor_occurrence_id: str
    target_clip_uid: str
    donor_clip_uid: str
    face_similarity: float = Field(ge=-1, le=1)
    voice_similarity: float = Field(ge=-1, le=1)
    target_face_path: str
    donor_face_path: str
    target_voice_path: str
    donor_voice_path: str
    pair_policy_version: Literal["h3_pair_policy_v1"] = H3_PAIR_POLICY_VERSION
    face_threshold: Literal[0.72] = H3_PAIR_POLICY_FACE_THRESHOLD
    voice_threshold: Literal[0.2] = H3_PAIR_POLICY_VOICE_THRESHOLD


class PairingPilotCrossPair(SchemaModel):
    pair_id: str
    target_clip_uid: str
    mappings: list[PairingPilotMapping]

    @model_validator(mode="after")
    def validate_pair(self) -> PairingPilotCrossPair:
        if not self.mappings:
            raise ValueError("pairing pilot cross-pair requires mappings")
        if any(item.target_clip_uid != self.target_clip_uid for item in self.mappings):
            raise ValueError("pairing pilot mappings must share the target clip")
        if [item.subject_index for item in self.mappings] != list(
            range(1, len(self.mappings) + 1)
        ):
            raise ValueError("pairing pilot mapping indices must be contiguous")
        return self


class PairingPilotEvidence(SchemaModel):
    target_occurrence_id: str
    donor_occurrence_id: str
    target_clip_uid: str
    donor_clip_uid: str
    pair_evidence: PairEvidence
    duplicate_source: bool
    donor_eligible: bool
    rejection_reason_codes: list[str]


class AcceptedPairHumanLabel(SchemaModel):
    accepted_pair_id: str
    target_occurrence_id: str
    donor_occurrence_id: str
    label: Literal["CORRECT", "WRONG", "UNCERTAIN"]
    face_similarity: float = Field(ge=-1, le=1)
    voice_similarity: float = Field(ge=-1, le=1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(values: list[SchemaModel]) -> str:
    return "".join(
        json.dumps(
            item.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for item in values
    )


def _load_occurrence_rows(root: Path) -> list[EmbeddingPilotOccurrence]:
    rows = [
        EmbeddingPilotOccurrence.model_validate_json(line)
        for line in (root / "occurrences.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows.sort(key=lambda item: item.entity_occurrence_id)
    ids = [item.entity_occurrence_id for item in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("pairing pilot embedding occurrences must be unique")
    for item in rows:
        for path_value, expected in (
            (item.visual_reference_path, item.visual_reference_sha256),
            (item.primary_voice_reference_path, item.primary_voice_reference_sha256),
        ):
            path = Path(path_value).expanduser().resolve(strict=True)
            if not path.is_file() or _sha256(path) != expected:
                raise ValueError("pairing pilot source artifact provenance mismatch")
    return rows


def _source_hashes(audio_root: Path, clip_uids: set[str]) -> dict[str, str]:
    output: dict[str, str] = {}
    for clip_uid in sorted(clip_uids):
        sidecar = json.loads(
            (audio_root / "clips" / clip_uid / "audio_binding.json").read_text(
                encoding="utf-8"
            )
        )
        if not isinstance(sidecar, dict) or sidecar.get("clip_uid") != clip_uid:
            raise ValueError("pairing pilot audio sidecar identity is invalid")
        if sidecar.get("status") != "ready":
            raise ValueError("pairing pilot requires matching ready audio sidecars")
        source_video_path = sidecar.get("source_video_path")
        if not isinstance(source_video_path, str) or not source_video_path.strip():
            raise ValueError("pairing pilot sidecar lacks source video provenance")
        video = Path(source_video_path).expanduser().resolve(strict=True)
        if not video.is_file():
            raise ValueError("pairing pilot source video is unavailable")
        output[clip_uid] = _sha256(video)
    return output


def _minimum(values: list[float | None]) -> float:
    available = [value for value in values if value is not None]
    return min(available) if available else 0.0


def _directional_evidence(
    *,
    target_id: str,
    donor_id: str,
    record: PairPolicyEvidence,
    config: AudioPairingConfig,
) -> PairEvidence:
    left_id = record.left_occurrence_id
    left_to_right = target_id == left_id
    return evaluate_pair_policy_v1(
        target_entity_occurrence_id=target_id,
        reference_entity_occurrence_id=donor_id,
        face_similarity=record.face_similarity,
        voice_similarity=record.voice_similarity,
        config=config,
        face_rank_left_to_right=(
            record.face_left_to_right_rank
            if left_to_right
            else record.face_right_to_left_rank
        ),
        face_rank_right_to_left=(
            record.face_right_to_left_rank
            if left_to_right
            else record.face_left_to_right_rank
        ),
        voice_rank_left_to_right=(
            record.voice_left_to_right_rank
            if left_to_right
            else record.voice_right_to_left_rank
        ),
        voice_rank_right_to_left=(
            record.voice_right_to_left_rank
            if left_to_right
            else record.voice_left_to_right_rank
        ),
        face_margin=_minimum(
            [
                record.face_top1_top2_margin_left,
                record.face_top1_top2_margin_right,
            ]
        ),
        voice_margin=_minimum(
            [
                record.voice_top1_top2_margin_left,
                record.voice_top1_top2_margin_right,
            ]
        ),
    )


def _review_url(path: str, output_root: Path) -> str:
    return quote(Path(os.path.relpath(Path(path), output_root)).as_posix(), safe="/..")


def _review_html(rows: list[PairingPilotMapping], output_root: Path) -> str:
    payload_rows = []
    for item in rows:
        payload = item.model_dump(mode="json")
        for key in (
            "target_face_path",
            "donor_face_path",
            "target_voice_path",
            "donor_voice_path",
        ):
            payload[f"{key}_url"] = _review_url(str(payload[key]), output_root)
        payload_rows.append(payload)
    payload = json.dumps(payload_rows, ensure_ascii=False, sort_keys=True).replace(
        "</", "<\\/"
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>H3 Accepted Cross-Pair Review</title><style>
body{{margin:0;font-family:system-ui,sans-serif;background:#f4f4f5;color:#18181b}}
header{{padding:16px 24px;background:white;border-bottom:1px solid #d4d4d8}}
main{{max-width:1120px;margin:20px auto;padding:0 20px}}
.faces{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
.face,.meta{{background:white;border:1px solid #d4d4d8;padding:12px}}
img{{width:100%;height:390px;object-fit:contain;background:#e4e4e7}}
audio{{width:100%;margin-top:10px}}.meta{{margin:14px 0;white-space:pre-wrap}}
.actions{{display:flex;gap:8px;flex-wrap:wrap}}
button{{padding:10px 15px;background:white;border:1px solid #71717a;cursor:pointer}}
button.active{{background:#18181b;color:white}}
@media(max-width:700px){{.faces{{grid-template-columns:1fr}}img{{height:300px}}}}
</style></head><body><header><strong>H3 Accepted Cross-Pair Review</strong>
<span id="progress"></span><p>Is this accepted cross-pair donor the same physical
person as the target, with a usable donor voice reference?</p></header><main>
<div class="faces"><div class="face"><strong id="target-id"></strong>
<img id="target-face"><audio id="target-audio" controls></audio></div>
<div class="face"><strong id="donor-id"></strong><img id="donor-face">
<audio id="donor-audio" controls></audio></div></div><div class="meta" id="meta"></div>
<div class="actions"><button data-label="CORRECT">1 CORRECT</button>
<button data-label="WRONG">2 WRONG</button><button data-label="UNCERTAIN">3 UNCERTAIN</button>
<button id="previous">Previous (k/left)</button><button id="next">Next (j/right)</button>
<button id="export">Export JSONL</button></div></main><script>
const cases={payload};const storageKey="r2v-h3-accepted-pair-review-v1";
const labels=JSON.parse(localStorage.getItem(storageKey)||"{{}}");let index=0;
function show(){{const x=cases[index];document.getElementById("progress").textContent=` ${{cases.length?index+1:0}}/${{cases.length}}`;if(!x)return;
document.getElementById("target-id").textContent=x.target_occurrence_id;
document.getElementById("donor-id").textContent=x.donor_occurrence_id;
document.getElementById("target-face").src=x.target_face_path_url;
document.getElementById("donor-face").src=x.donor_face_path_url;
document.getElementById("target-audio").src=x.target_voice_path_url;
document.getElementById("donor-audio").src=x.donor_voice_path_url;
document.getElementById("meta").textContent=`target_clip=${{x.target_clip_uid}} donor_clip=${{x.donor_clip_uid}}\nface=${{x.face_similarity.toFixed(6)}} threshold=${{x.face_threshold}}\nvoice=${{x.voice_similarity.toFixed(6)}} threshold=${{x.voice_threshold}}\npolicy=${{x.pair_policy_version}}`;
document.querySelectorAll("button[data-label]").forEach(b=>b.classList.toggle("active",labels[x.accepted_pair_id]===b.dataset.label));}}
function label(v){{if(!cases.length)return;labels[cases[index].accepted_pair_id]=v;localStorage.setItem(storageKey,JSON.stringify(labels));show()}}
function move(d){{if(cases.length){{index=(index+d+cases.length)%cases.length;show()}}}}
document.querySelectorAll("button[data-label]").forEach(b=>b.onclick=()=>label(b.dataset.label));
document.getElementById("previous").onclick=()=>move(-1);document.getElementById("next").onclick=()=>move(1);
document.getElementById("export").onclick=()=>{{const rows=cases.filter(x=>labels[x.accepted_pair_id]).map(x=>JSON.stringify({{
accepted_pair_id:x.accepted_pair_id,target_occurrence_id:x.target_occurrence_id,donor_occurrence_id:x.donor_occurrence_id,label:labels[x.accepted_pair_id],face_similarity:x.face_similarity,voice_similarity:x.voice_similarity}})).join("\\n")+"\\n";
const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([rows],{{type:"application/x-ndjson"}}));a.download="accepted_pair_review_labels.jsonl";a.click();URL.revokeObjectURL(a.href)}};
document.addEventListener("keydown",e=>{{if(e.key==="1")label("CORRECT");else if(e.key==="2")label("WRONG");else if(e.key==="3")label("UNCERTAIN");else if(e.key==="j"||e.key==="ArrowRight"||e.key==="ArrowDown")move(1);else if(e.key==="k"||e.key==="ArrowLeft"||e.key==="ArrowUp")move(-1)}});show();
</script></body></html>"""


def _publish(temporary: Path, destination: Path, overwrite: bool) -> None:
    if not destination.exists():
        temporary.replace(destination)
        return
    if not overwrite:
        raise FileExistsError(f"pairing pilot output already exists: {destination}")
    backup = destination.with_name(f".{destination.name}.backup-{uuid.uuid4().hex}")
    destination.replace(backup)
    published = False
    try:
        temporary.replace(destination)
        published = True
    finally:
        if published:
            shutil.rmtree(backup)
        elif backup.exists() and not destination.exists():
            backup.replace(destination)


def build_pairing_pilot(
    *,
    audio_pilot_root: Path,
    embedding_root: Path,
    output_root: Path,
    overwrite: bool = False,
) -> dict[str, object]:
    audio_root = audio_pilot_root.expanduser().resolve(strict=True)
    embedding = embedding_root.expanduser().resolve(strict=True)
    destination = output_root.expanduser().resolve(strict=False)
    if any(
        destination == source or source in destination.parents or destination in source.parents
        for source in (audio_root, embedding)
    ):
        raise ValueError("pairing pilot output must be separate from source artifacts")
    rows = _load_occurrence_rows(embedding)
    by_id = {item.entity_occurrence_id: item for item in rows}
    source_hash = _source_hashes(audio_root, {item.clip_uid for item in rows})
    _, complete, canonical = build_pair_policy_evidence(
        embedding_root=embedding,
        face_mining_root=None,
    )
    config = AudioPairingConfig()
    evidence_rows: list[PairingPilotEvidence] = []
    eligible_by_target: dict[str, list[PairEvidence]] = defaultdict(list)
    rejection_counts: Counter[str] = Counter()
    for record in canonical:
        for target_id, donor_id in (
            (record.left_occurrence_id, record.right_occurrence_id),
            (record.right_occurrence_id, record.left_occurrence_id),
        ):
            edge = _directional_evidence(
                target_id=target_id,
                donor_id=donor_id,
                record=record,
                config=config,
            )
            duplicate = source_hash[by_id[target_id].clip_uid] == source_hash[
                by_id[donor_id].clip_uid
            ]
            reasons = [
                code
                for part in (edge.same_person, edge.same_voice)
                if part.status != "accepted"
                for code in part.reason_codes
            ]
            if duplicate:
                reasons.append("duplicate_source_video")
            donor_eligible = not reasons
            rejection_counts.update(reasons)
            evidence_rows.append(
                PairingPilotEvidence(
                    target_occurrence_id=target_id,
                    donor_occurrence_id=donor_id,
                    target_clip_uid=by_id[target_id].clip_uid,
                    donor_clip_uid=by_id[donor_id].clip_uid,
                    pair_evidence=edge,
                    duplicate_source=duplicate,
                    donor_eligible=donor_eligible,
                    rejection_reason_codes=reasons,
                )
            )
            if donor_eligible:
                eligible_by_target[target_id].append(edge)
    evidence_rows.sort(key=lambda item: (item.target_occurrence_id, item.donor_occurrence_id))
    by_clip: dict[str, list[EmbeddingPilotOccurrence]] = defaultdict(list)
    for item in rows:
        by_clip[item.clip_uid].append(item)
    in_pairs = [
        PairingPilotInPair(
            pair_id=f"in_pair/{clip_uid}",
            clip_uid=clip_uid,
            entity_occurrence_ids=[item.entity_occurrence_id for item in items],
            primary_voice_paths=[item.primary_voice_reference_path for item in items],
        )
        for clip_uid, items in sorted(by_clip.items())
    ]
    cross_pairs: list[PairingPilotCrossPair] = []
    eligible_target_clips: list[str] = []
    incomplete_reasons: Counter[str] = Counter()
    for clip_uid, items in sorted(by_clip.items()):
        ordered = sorted(items, key=lambda item: item.entity_occurrence_id)
        if any(item.entity_occurrence_id not in complete for item in ordered):
            incomplete_reasons.update(["target_missing_required_embedding"])
            continue
        eligible_target_clips.append(clip_uid)
        candidate_sets = [
            sorted(
                eligible_by_target.get(item.entity_occurrence_id, []),
                key=lambda edge: (
                    -edge.same_person.face_similarity,
                    edge.reference_entity_occurrence_id,
                ),
            )
            for item in ordered
        ]
        matching = select_complete_donor_matching(candidate_sets)
        if matching is None:
            incomplete_reasons.update(["no_complete_strict_donor_mapping"])
            continue
        pair_id = f"cross_pair/{clip_uid}/1"
        mappings: list[PairingPilotMapping] = []
        for index, edge in enumerate(matching, start=1):
            target = by_id[edge.target_entity_occurrence_id]
            donor = by_id[edge.reference_entity_occurrence_id]
            target_complete = complete[target.entity_occurrence_id]
            donor_complete = complete[donor.entity_occurrence_id]
            mappings.append(
                PairingPilotMapping(
                    accepted_pair_id=f"{pair_id}/subject_{index}",
                    subject_index=index,
                    target_occurrence_id=target.entity_occurrence_id,
                    donor_occurrence_id=donor.entity_occurrence_id,
                    target_clip_uid=target.clip_uid,
                    donor_clip_uid=donor.clip_uid,
                    face_similarity=edge.same_person.face_similarity,
                    voice_similarity=edge.same_voice.voice_similarity,
                    target_face_path=str(target_complete.face_crop_path),
                    donor_face_path=str(donor_complete.face_crop_path),
                    target_voice_path=str(target_complete.primary_voice_path),
                    donor_voice_path=str(donor_complete.primary_voice_path),
                )
            )
        cross_pairs.append(
            PairingPilotCrossPair(
                pair_id=pair_id,
                target_clip_uid=clip_uid,
                mappings=mappings,
            )
        )
    review_rows = [item for pair in cross_pairs for item in pair.mappings]
    face_values = [item.face_similarity for item in review_rows]
    voice_values = [item.voice_similarity for item in review_rows]

    def distribution(values: list[float]) -> tuple[float | None, float | None, float | None]:
        return (
            (min(values), statistics.median(values), max(values))
            if values
            else (None, None, None)
        )

    face_min, face_median, face_max = distribution(face_values)
    voice_min, voice_median, voice_max = distribution(voice_values)
    summary: dict[str, object] = {
        "eligible_in_pair_occurrence_count": len(rows),
        "in_pair_count": len(in_pairs),
        "cross_pair_eligible_target_count": len(eligible_target_clips),
        "cross_pair_count": len(cross_pairs),
        "accepted_target_donor_mapping_count": len(review_rows),
        "target_without_cross_pair_count": len(eligible_target_clips) - len(cross_pairs),
        "accepted_pair_face_min": face_min,
        "accepted_pair_face_median": face_median,
        "accepted_pair_face_max": face_max,
        "accepted_pair_voice_min": voice_min,
        "accepted_pair_voice_median": voice_median,
        "accepted_pair_voice_max": voice_max,
        "rejection_reason_counts": dict(sorted(rejection_counts.items())),
        "incomplete_target_reason_counts": dict(sorted(incomplete_reasons.items())),
        "thresholds_calibrated": True,
        "pair_policy_version": H3_PAIR_POLICY_VERSION,
        "face_threshold": H3_PAIR_POLICY_FACE_THRESHOLD,
        "voice_threshold": H3_PAIR_POLICY_VOICE_THRESHOLD,
        "rank_gate_enabled": False,
        "margin_gate_enabled": False,
        "text_gate_enabled": False,
        "transitive_clustering_performed": False,
        "parent_quota_applied": False,
        "human_calibration_labels_used_as_identity_truth": False,
    }
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    try:
        temporary.mkdir()
        (temporary / "in_pairs.jsonl").write_text(_jsonl(in_pairs), encoding="utf-8")
        (temporary / "cross_pairs.jsonl").write_text(
            _jsonl(cross_pairs), encoding="utf-8"
        )
        (temporary / "pair_evidence.jsonl").write_text(
            _jsonl(evidence_rows), encoding="utf-8"
        )
        (temporary / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "review.html").write_text(
            _review_html(review_rows, destination), encoding="utf-8"
        )
        _publish(temporary, destination, overwrite)
        return summary
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def report_accepted_pair_review(
    *,
    pairing_pilot_root: Path,
    labels_path: Path,
    output_path: Path,
) -> dict[str, object]:
    root = pairing_pilot_root.expanduser().resolve(strict=True)
    pairs = [
        PairingPilotCrossPair.model_validate_json(line)
        for line in (root / "cross_pairs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    accepted = {item.accepted_pair_id: item for pair in pairs for item in pair.mappings}
    labels = [
        AcceptedPairHumanLabel.model_validate_json(line)
        for line in labels_path.expanduser().resolve(strict=True)
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    label_ids = [item.accepted_pair_id for item in labels]
    if len(label_ids) != len(set(label_ids)):
        raise ValueError("accepted-pair review labels contain duplicate IDs")
    for label in labels:
        expected = accepted.get(label.accepted_pair_id)
        if expected is None:
            raise ValueError("accepted-pair review label references an unknown pair")
        if (
            label.target_occurrence_id != expected.target_occurrence_id
            or label.donor_occurrence_id != expected.donor_occurrence_id
            or abs(label.face_similarity - expected.face_similarity) > 1e-6
            or abs(label.voice_similarity - expected.voice_similarity) > 1e-6
        ):
            raise ValueError("accepted-pair review label diagnostics do not match")
    counts = Counter(item.label for item in labels)
    denominator = counts["CORRECT"] + counts["WRONG"]
    report: dict[str, object] = {
        "accepted_cross_pair_count": len(accepted),
        "reviewed_count": len(labels),
        "correct_count": counts["CORRECT"],
        "wrong_count": counts["WRONG"],
        "uncertain_count": counts["UNCERTAIN"],
        "empirical_precision": (
            counts["CORRECT"] / denominator if denominator else None
        ),
        "wrong_pairs": [
            item.model_dump(mode="json") for item in labels if item.label == "WRONG"
        ],
        "uncertain_pairs": [
            item.model_dump(mode="json")
            for item in labels
            if item.label == "UNCERTAIN"
        ],
        "pair_policy_version": H3_PAIR_POLICY_VERSION,
        "thresholds_modified": False,
    }
    destination = output_path.expanduser().resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return report
