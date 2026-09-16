"""Small sequential pilot. Existing Visual/H3 pipelines are never invoked."""

from itertools import islice
from pathlib import Path

from r2v_data_v2.manifest import iter_source_records
from r2v_data_v2.naming import clip_uid
from r2v_data_v2.reconciliation import write_json_atomic
from r2v_data_v2.v3.frames import OpenCvFrameDecoder
from r2v_data_v2.v3.production_source import JeaVideoMotionAdapter

from .bernini import serialize_case


def build_prompt(source: str, replacement: str) -> str:
    return (
        f"Replace {source} with {replacement}.\n\n"
        "This is an appearance-only person replacement. Keep the original performance and motion "
        "exactly unchanged. Do not invent, reinterpret, simplify, or adjust the person's action or "
        "pose to fit the replacement appearance.\n\n"
        "For every frame, preserve the original body pose sequence, body orientation, limb positions, "
        "hand placement, hand-body and hand-object contact states, foot placement, head pose, facial "
        "expression, gaze direction, mouth state, spatial position, scale, occlusion relationships, "
        "and timing of all movements. The replacement person must follow the source person's motion "
        "frame by frame. Preserve all contacts exactly: if a hand is inside a pocket, resting on the "
        "body, holding an object, or touching another surface, keep the same contact state and placement.\n\n"
        "Only change the person's identity and visible appearance. Keep the background, foreground "
        "objects, camera motion, framing, crop, lighting, shadows, reflections, scene geometry, "
        "composition, and all non-human regions unchanged. Do not move, add, remove, or redesign any "
        "scene element."
    )


def validate_output_root(path: Path) -> Path:
    output = path.expanduser().resolve()
    for root in ("/mnt/workspace/public/dataset", "/mnt/workspace/public/pretrained",
                 "/mnt/workspace/liutao/X_human_data"):
        protected = Path(root).resolve()
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError("output_root overlaps a read-only source tree")
    server = Path("/mnt/workspace")
    owned = server / "litengjie/data"
    if server in output.parents and owned not in output.parents:
        raise ValueError("server output_root must be below the user's writable data root")
    return output


def select_cases(input_jsonl: Path, clips_root: Path, output_root: Path, *, limit: int) -> list[dict]:
    if input_jsonl.name == "collected_face_2.jsonl":
        raise ValueError("collected_face_2.jsonl is not enabled for this pilot")
    if input_jsonl.suffix.lower() != ".jsonl" or limit < 1:
        raise ValueError("require JSONL input and a positive --limit")
    source = input_jsonl.expanduser().resolve(strict=True)
    clips = clips_root.expanduser().resolve(strict=True)
    if not clips.is_dir():
        raise ValueError("--clips-root must name the actual processed clip directory")
    output = validate_output_root(output_root)
    if output == clips or clips in output.parents or output in clips.parents or output in source.parents:
        raise ValueError("output_root must not overlap source inputs")
    # Existing adapter with explicit roots, as in the existing T2VA consumer.
    # Only resolve_clip_path is called: source_video_path is provenance, NEVER input.
    resolver = JeaVideoMotionAdapter(clips_root=clips, source_videos_root=clips)
    cases = []
    seen = set()
    for index, raw in enumerate(islice(iter_source_records(source), limit)):
        video, _ = resolver.resolve_clip_path(raw)
        if video.suffix.lower() != ".mp4":
            raise ValueError("processed target clip must be MP4")
        if video in seen:
            raise ValueError(f"duplicate target clip in pilot selection: {video}")
        seen.add(video)
        case_id = f"{index:06d}-{clip_uid(video)}"
        cases.append({
            "case_id": case_id, "source_index": index,
            "source_video_id": raw.get("source_video_id"),
            "source_video_path": raw.get("source_video_path"),
            "target_video_path": str(video), "output_dir": str(output / case_id),
        })
    if not cases:
        raise ValueError("input JSONL contains no pilot cases")
    return cases


def extract_frame_zero(video: Path, destination: Path):
    frame = OpenCvFrameDecoder().decode_indices(video, [0])[0]
    frame.image.save(destination, format="JPEG", quality=95)


def validate_video(video: Path):
    if not video.is_file() or video.stat().st_size == 0:
        raise ValueError(f"Bernini produced no video: {video}")
    OpenCvFrameDecoder().decode_indices(video, [0])


def run_cases(cases, *, qwen, bernini, seed: int) -> list[Path]:
    bernini.validate()  # Fail before Qwen load if launcher/config is missing.
    for case in cases:
        output = validate_output_root(Path(case["output_dir"]))
        if output.exists():
            raise FileExistsError(f"pilot does not overwrite existing case: {output}")
    prepared = []
    try:
        for case in cases:
            output = Path(case["output_dir"])
            output.mkdir(parents=True, exist_ok=False)
            video = Path(case["target_video_path"])
            reference = output / "reference_frame0.jpg"
            extract_frame_zero(video, reference)
            source = qwen.describe(video)
            replacement = qwen.invent(source)
            prompt = build_prompt(source, replacement)
            for name, text in (("source_person", source), ("replacement_person", replacement),
                               ("bernini_prompt", prompt)):
                (output / f"{name}.txt").write_text(text + "\n", encoding="utf-8")
            generated = output / "replacement.mp4"
            case_path = output / "bernini_case.json"
            write_json_atomic(case_path, serialize_case(video, generated, prompt))
            prepared.append((case_path, {
                **case, "reference_image_path": str(reference), "input_video_path": str(generated),
                "source_person_description": source, "replacement_person_description": replacement,
                "bernini_prompt": prompt, "qwen_model_path": str(qwen.model_path),
                "bernini_model_path": str(bernini.config), "seed": seed,
            }))
    finally:
        qwen.close()
    # Avoid resident Qwen competing with the official multi-GPU Bernini subprocess.
    manifests = []
    for case_path, manifest in prepared:
        bernini.run(case_path, seed)
        validate_video(Path(manifest["input_video_path"]))
        path = case_path.parent / "manifest.json"
        write_json_atomic(path, manifest)
        manifests.append(path)
    return manifests
