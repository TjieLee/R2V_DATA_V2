from __future__ import annotations

import json
import uuid
from pathlib import Path

from r2v_data_v2.h3.face_identity_mining import (
    FaceIdentityCandidate,
    FaceMiningOccurrence,
)


def _read_jsonl(
    path: Path,
    model: type[FaceIdentityCandidate | FaceMiningOccurrence],
):
    return [
        model.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _resolve_crop(root: Path, occurrence: FaceMiningOccurrence) -> str:
    if occurrence.face.status != "available" or occurrence.face.crop_asset is None:
        raise ValueError("face review candidate requires an available crop")
    relative = Path(occurrence.face.crop_asset.path)
    if relative.is_absolute():
        raise ValueError("face review crop path must be relative")
    resolved = (root / relative).resolve(strict=True)
    resolved.relative_to(root)
    if not resolved.is_file():
        raise FileNotFoundError("face review crop is not a file")
    return relative.as_posix()


def _review_rows(root: Path) -> list[dict[str, object]]:
    occurrences = _read_jsonl(root / "occurrences.jsonl", FaceMiningOccurrence)
    candidates = _read_jsonl(root / "face_candidates.jsonl", FaceIdentityCandidate)
    by_id = {item.entity_occurrence_id: item for item in occurrences}
    same_parent = [item for item in candidates if item.candidate_pool == "same_parent"]
    same_parent.sort(
        key=lambda item: (
            not item.mutual_top_k,
            min(item.left_to_right_rank, item.right_to_left_rank),
            max(item.left_to_right_rank, item.right_to_left_rank),
            -item.face_similarity,
            item.left_occurrence_id,
            item.right_occurrence_id,
        )
    )
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for candidate in same_parent:
        pair = (candidate.left_occurrence_id, candidate.right_occurrence_id)
        if pair in seen:
            raise ValueError("face review input contains a duplicate unordered pair")
        seen.add(pair)
        left = by_id.get(candidate.left_occurrence_id)
        right = by_id.get(candidate.right_occurrence_id)
        if left is None or right is None:
            raise ValueError("face review candidate references an unknown occurrence")
        rows.append(
            {
                "left_occurrence_id": candidate.left_occurrence_id,
                "right_occurrence_id": candidate.right_occurrence_id,
                "face_similarity": candidate.face_similarity,
                "left_to_right_rank": candidate.left_to_right_rank,
                "right_to_left_rank": candidate.right_to_left_rank,
                "mutual_top_k": candidate.mutual_top_k,
                "parent_video_id": candidate.parent_video_id,
                "left_clip_suffix": candidate.left_clip_suffix,
                "right_clip_suffix": candidate.right_clip_suffix,
                "left_crop_path": _resolve_crop(root, left),
                "right_crop_path": _resolve_crop(root, right),
                "left_existing_v3_cross_pair_provenance": (
                    left.existing_v3_cross_pair_provenance
                ),
                "right_existing_v3_cross_pair_provenance": (
                    right.existing_v3_cross_pair_provenance
                ),
                "left_visual_integrity_provenance": (
                    left.visual_integrity_provenance
                ),
                "right_visual_integrity_provenance": (
                    right.visual_integrity_provenance
                ),
            }
        )
    return rows


def _render_html(rows: list[dict[str, object]]) -> str:
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True).replace(
        "</", "<\\/"
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>H3 Face Identity Review</title>
<style>
body {{ margin: 0; font-family: system-ui, sans-serif; background: #f3f4f6; color: #171717; }}
header {{ padding: 16px 24px; background: #fff; border-bottom: 1px solid #d4d4d4; }}
main {{ max-width: 1120px; margin: 20px auto; padding: 0 20px; }}
.images {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
.image {{ background: #fff; border: 1px solid #d4d4d4; padding: 10px; }}
img {{ display: block; width: 100%; height: 440px; object-fit: contain; background: #e5e7eb; }}
.meta {{ margin: 14px 0; padding: 14px; background: #fff; border: 1px solid #d4d4d4; }}
.actions {{ display: flex; gap: 10px; flex-wrap: wrap; }}
button {{ padding: 10px 16px; border: 1px solid #737373; background: #fff; cursor: pointer; }}
button.active {{ background: #171717; color: #fff; }}
pre {{ white-space: pre-wrap; overflow-wrap: anywhere; font-size: 12px; }}
@media (max-width: 700px) {{ .images {{ grid-template-columns: 1fr; }} img {{ height: 320px; }} }}
</style>
</head>
<body>
<header>
  <strong>H3 Face Identity Review</strong>
  <span id="progress"></span>
  <p>Are these two occurrences the same physical person?</p>
</header>
<main>
  <div class="images">
    <div class="image"><strong id="left-id"></strong><img id="left-image" alt="Left face crop"></div>
    <div class="image"><strong id="right-id"></strong><img id="right-image" alt="Right face crop"></div>
  </div>
  <div class="meta" id="metrics"></div>
  <details class="meta"><summary>Visual provenance</summary><pre id="provenance"></pre></details>
  <div class="actions">
    <button data-label="same">1 SAME</button>
    <button data-label="different">2 DIFFERENT</button>
    <button data-label="uncertain">3 UNCERTAIN</button>
    <button id="previous">Previous (k/left)</button>
    <button id="next">Next (j/right)</button>
    <button id="export">Export JSONL</button>
  </div>
</main>
<script>
const cases = {payload};
const storageKey = "r2v-h3-face-identity-labels-v1";
const labels = JSON.parse(localStorage.getItem(storageKey) || "{{}}");
let index = 0;
function pairKey(item) {{ return `${{item.left_occurrence_id}}|${{item.right_occurrence_id}}`; }}
function show() {{
  const item = cases[index];
  document.getElementById("progress").textContent = ` ${{cases.length ? index + 1 : 0}}/${{cases.length}}`;
  if (!item) return;
  document.getElementById("left-id").textContent = item.left_occurrence_id;
  document.getElementById("right-id").textContent = item.right_occurrence_id;
  document.getElementById("left-image").src = item.left_crop_path;
  document.getElementById("right-image").src = item.right_crop_path;
  document.getElementById("metrics").textContent =
    `parent=${{item.parent_video_id}} | suffixes=${{item.left_clip_suffix}},${{item.right_clip_suffix}} | ` +
    `cosine=${{item.face_similarity.toFixed(6)}} | ranks=${{item.left_to_right_rank}}/${{item.right_to_left_rank}} | ` +
    `mutual_top_k=${{item.mutual_top_k}}`;
  document.getElementById("provenance").textContent = JSON.stringify({{
    left_cross_pair: item.left_existing_v3_cross_pair_provenance,
    right_cross_pair: item.right_existing_v3_cross_pair_provenance,
    left_integrity: item.left_visual_integrity_provenance,
    right_integrity: item.right_visual_integrity_provenance,
  }}, null, 2);
  document.querySelectorAll("button[data-label]").forEach((button) =>
    button.classList.toggle("active", button.dataset.label === labels[pairKey(item)]));
}}
function label(value) {{
  if (!cases.length) return;
  labels[pairKey(cases[index])] = value;
  localStorage.setItem(storageKey, JSON.stringify(labels));
  show();
}}
function move(delta) {{ if (cases.length) {{ index = (index + delta + cases.length) % cases.length; show(); }} }}
document.querySelectorAll("button[data-label]").forEach((button) =>
  button.addEventListener("click", () => label(button.dataset.label)));
document.getElementById("previous").addEventListener("click", () => move(-1));
document.getElementById("next").addEventListener("click", () => move(1));
document.getElementById("export").addEventListener("click", () => {{
  const rows = cases.filter((item) => labels[pairKey(item)]).map((item) => JSON.stringify({{
    left_occurrence_id: item.left_occurrence_id,
    right_occurrence_id: item.right_occurrence_id,
    same_person_label: labels[pairKey(item)],
    face_similarity: item.face_similarity,
    left_to_right_rank: item.left_to_right_rank,
    right_to_left_rank: item.right_to_left_rank,
    mutual_top_k: item.mutual_top_k,
    parent_video_id: item.parent_video_id,
  }})).join("\\n") + "\\n";
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([rows], {{type: "application/x-ndjson"}}));
  link.download = "face_identity_labels.jsonl";
  link.click();
  URL.revokeObjectURL(link.href);
}});
document.addEventListener("keydown", (event) => {{
  if (event.key === "1") label("same");
  else if (event.key === "2") label("different");
  else if (event.key === "3") label("uncertain");
  else if (event.key === "j" || event.key === "ArrowRight" || event.key === "ArrowDown") move(1);
  else if (event.key === "k" || event.key === "ArrowLeft" || event.key === "ArrowUp") move(-1);
}});
show();
</script>
</body>
</html>
"""


def build_face_identity_review(face_mining_root: Path) -> Path:
    root = face_mining_root.expanduser().resolve(strict=True)
    if not (root / "summary.json").is_file():
        raise ValueError("face mining root is incomplete")
    destination = root / "review.html"
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(_render_html(_review_rows(root)), encoding="utf-8")
    temporary.replace(destination)
    return destination
