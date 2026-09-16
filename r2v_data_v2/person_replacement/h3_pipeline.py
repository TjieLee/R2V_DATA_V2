"""Independent sequential H3 candidate pilot; never writes production samples."""

import json
import time
from dataclasses import asdict
from pathlib import Path

from r2v_data_v2.reconciliation import write_json_atomic

from .h3_prompt import build_h3_ref2va_replacement_prompt
from .h3_ref2va import match_aspect_ratio, output_size, plan_h3_timeline
from .pipeline import extract_frame_zero, validate_output_root
from .timeline import inspect_video_timeline


def read_descriptions(manifest: Path, video: Path) -> tuple[str, str]:
    data = json.loads(manifest.read_text())
    if Path(data["target_video_path"]).resolve() != video.resolve():
        raise ValueError("Description manifest original source does not match this case")
    values = data["source_person_description"], data["replacement_person_description"]
    if not all(isinstance(value, str) and value.strip() for value in values):
        raise ValueError("Description manifest must contain two nonempty descriptions")
    return values


def run_cases(cases, *, qwen, backends, seed, descriptions_from=None):
    preflight_errors = {}
    for backend in backends:
        try:
            backend.validate()
        except Exception as exc:  # noqa: BLE001 -- independent backend preflight
            preflight_errors[backend.name] = exc
    if len(preflight_errors) == len(backends):
        qwen.close()
        return {"manifests":[], "failures":[
            {"case_id":case["case_id"], "backend":name, "error_type":type(exc).__name__, "error":str(exc)}
            for case in cases for name,exc in preflight_errors.items()]}
    prepared = []
    try:
        for case in cases:
            directory = validate_output_root(Path(case["output_dir"]))
            if directory.exists():
                raise FileExistsError(f"H3 pilot never overwrites a case: {directory}")
            video = Path(case["target_video_path"])
            source = inspect_video_timeline(video)
            plan = plan_h3_timeline(source)
            aspect = match_aspect_ratio(source)
            directory.mkdir(parents=True, exist_ok=False)
            reference = directory / "reference_frame0.jpg"
            extract_frame_zero(video, reference)
            if descriptions_from is None:
                description = qwen.describe(video)
                replacement = qwen.invent(description)
            else:
                description, replacement = read_descriptions(descriptions_from, video)
            prompt = build_h3_ref2va_replacement_prompt(description, replacement)
            for name, text in (("source_person", description), ("replacement_person", replacement), ("h3_prompt", prompt)):
                (directory / f"{name}.txt").write_text(text + "\n", encoding="utf-8")
            write_json_atomic(directory / "source_timeline.json", asdict(source))
            prepared.append((case, directory, video, reference, description, replacement, prompt, plan, aspect))
    finally:
        qwen.close()
    report = {"manifests": [], "failures": []}
    for case, directory, video, reference, description, replacement, prompt, plan, aspect in prepared:
        for backend in backends:
            target = directory / backend.name
            target.mkdir()
            try:
                if backend.name in preflight_errors:
                    raise preflight_errors[backend.name]
                started = time.monotonic()
                metadata = backend.run(video, prompt, plan, aspect, target, seed)
                elapsed = time.monotonic()-started
                raw = target / "raw.mp4"
                output = inspect_video_timeline(raw)
                if output.frame_count != plan.native_frame_count or output.rate != 24:
                    raise ValueError("H3 generated output does not match planned native frame count / 24 FPS")
                if abs(output.duration_seconds - plan.native_frame_count/24) > 1/24:
                    raise ValueError("H3 generated duration does not match native timeline")
                if (output.width, output.height) != output_size(aspect):
                    raise ValueError("H3 output resolution differs from shared 1.0 MP contract")
                manifest = {
                    **metadata, "case_id":case["case_id"], "backend":backend.name,
                    "target_video_path":str(video), "reference_image_path":str(reference),
                    "input_video_path":str(raw), "source_person_description":description,
                    "replacement_person_description":replacement, "h3_prompt":prompt,
                    "descriptions_origin":str(descriptions_from) if descriptions_from else "qwen_describe_invent",
                    "h3_model_path":str(backend.model_root), "acceleration_checkpoint_path":str(backend.lora),
                    "seed":seed, "requested_duration":plan.requested_duration,
                    "aspect_ratio":aspect, "megapixels":1.0,
                    **{f"source_{k}":v for k,v in asdict(plan.source).items()},
                    **{f"output_{k}":v for k,v in asdict(output).items()},
                    "h3_native_frame_count":plan.native_frame_count, "h3_native_fps":24,
                    "h3_native_duration":output.duration_seconds,
                    "duration_drift_seconds":output.duration_seconds-plan.source.duration_seconds,
                    "backend_wall_seconds":elapsed, "production_timeline_compatible":False,
                }
                path = target / "manifest.json"
                write_json_atomic(path, manifest)
                report["manifests"].append(str(path))
            except Exception as exc:  # noqa: BLE001 -- isolate backends, report failure and nonzero CLI status
                failure = {"case_id":case["case_id"], "backend":backend.name,
                           "error_type":type(exc).__name__, "error":str(exc)}
                write_json_atomic(target / "failure.json", failure)
                report["failures"].append(failure)
    return report
