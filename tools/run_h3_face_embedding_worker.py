#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from r2v_data_v2.h3.audio_backends import fingerprint_local_model_path


def validate_face_model_pack(model_root: Path, model_name: str) -> Path:
    if not model_name.strip() or Path(model_name).name != model_name:
        raise ValueError("face model name must be one local directory name")
    root = model_root.expanduser().resolve(strict=True)
    pack = (root / "models" / model_name).resolve(strict=True)
    pack.relative_to(root)
    if not pack.is_dir() or not any(pack.rglob("*.onnx")):
        raise FileNotFoundError(
            "explicit local InsightFace model pack must contain ONNX files"
        )
    return pack


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent local face embedding worker")
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-identifier", required=True)
    parser.add_argument("--model-fingerprint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--det-size", type=int, default=640)
    parser.add_argument("--det-threshold", type=float, default=0.50)
    return parser.parse_args()


class InsightFaceWorker:
    def __init__(self, args: argparse.Namespace) -> None:
        pack = validate_face_model_pack(args.model_root, args.model_name)
        fingerprint = fingerprint_local_model_path(pack)
        if fingerprint != args.model_fingerprint:
            raise ValueError("face model-pack fingerprint does not match")
        if args.det_size <= 0 or not 0 <= args.det_threshold <= 1:
            raise ValueError("face detector settings are invalid")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        import onnxruntime as ort  # type: ignore[import-not-found]

        try:
            with contextlib.redirect_stdout(sys.stderr):
                ort.preload_dlls()
        except Exception as exc:
            raise RuntimeError(
                "ONNX Runtime CUDA/cuDNN preload failed before InsightFace "
                "session creation"
            ) from exc

        if args.device == "cpu":
            providers = ["CPUExecutionProvider"]
            context_id = -1
        elif args.device.startswith("cuda:"):
            device_index = args.device.split(":", maxsplit=1)[1]
            if not device_index.isdigit():
                raise ValueError("face CUDA device must use cuda:N with N >= 0")
            context_id = int(device_index)
            available_providers = set(ort.get_available_providers())
            if "CUDAExecutionProvider" not in available_providers:
                raise RuntimeError(
                    "CUDAExecutionProvider was requested but is unavailable; "
                    "verify the CUDA-12 ONNX Runtime and cuDNN installation"
                )
            providers = ["CUDAExecutionProvider"]
        else:
            raise ValueError("face device must be cpu or cuda:N")

        import cv2  # type: ignore[import-not-found]
        import insightface  # type: ignore[import-not-found]
        from insightface.utils import face_align  # type: ignore[import-not-found]

        try:
            with contextlib.redirect_stdout(sys.stderr):
                app = insightface.app.FaceAnalysis(
                    name=args.model_name,
                    root=str(args.model_root.expanduser().resolve(strict=True)),
                    providers=providers,
                )
                if context_id >= 0:
                    for model in app.models.values():
                        session = getattr(model, "session", None)
                        if session is None or not hasattr(session, "get_providers"):
                            raise RuntimeError(
                                "InsightFace model does not expose ONNX provider state"
                            )
                        active_providers = session.get_providers()
                        if (
                            not active_providers
                            or active_providers[0] != "CUDAExecutionProvider"
                        ):
                            raise RuntimeError(
                                "InsightFace CUDA session silently fell back from "
                                "CUDAExecutionProvider"
                            )
                app.prepare(
                    ctx_id=context_id,
                    det_size=(args.det_size, args.det_size),
                    det_thresh=args.det_threshold,
                )
        except Exception as exc:
            if context_id >= 0:
                raise RuntimeError(
                    "InsightFace CUDA session initialization failed; verify "
                    "CUDA 12 and cuDNN runtime loading"
                ) from exc
            raise
        self.args = args
        self.app = app
        self.cv2 = cv2
        self.face_align = face_align
        self.backend_version = _package_version("insightface")
        self.onnxruntime_version = str(ort.__version__)
        self.cuda_requested = context_id >= 0

    def process(self, request: dict[str, Any]) -> dict[str, Any]:
        image_path = Path(str(request["image_path"])).expanduser().resolve(strict=True)
        crop_path = Path(str(request["face_crop_output_path"])).expanduser()
        image = self.cv2.imread(str(image_path), self.cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("canonical face source is not a readable image")
        try:
            detected_faces = self.app.get(image)
        except Exception as exc:
            if self.cuda_requested:
                raise RuntimeError(
                    "InsightFace CUDA inference failed; verify CUDA 12 and cuDNN "
                    "runtime loading"
                ) from exc
            raise
        faces = [
            face
            for face in detected_faces
            if float(face.det_score) >= self.args.det_threshold
        ]
        common = {
            "model_identifier": self.args.model_identifier,
            "model_fingerprint": self.args.model_fingerprint,
        }
        if not faces:
            return {
                **common,
                "status": "unavailable",
                "reason": "face_not_found_in_canonical_reference",
            }
        if len(faces) != 1:
            return {
                **common,
                "status": "unavailable",
                "reason": "multiple_faces_in_canonical_reference",
            }
        face = faces[0]
        landmarks = np.asarray(face.kps, dtype=np.float32)
        if landmarks.shape != (5, 2) or not np.isfinite(landmarks).all():
            raise ValueError("face alignment requires five finite landmarks")
        crop = self.face_align.norm_crop(image, landmark=landmarks)
        if crop is None or crop.size == 0:
            raise ValueError("face alignment produced an empty crop")
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.cv2.imwrite(str(crop_path), crop):
            raise OSError("failed to write canonical face crop")
        vector = getattr(face, "normed_embedding", None)
        if vector is None:
            vector = getattr(face, "embedding", None)
        values = np.asarray(vector, dtype=np.float32).reshape(-1)
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError("face model returned an invalid embedding")
        return {
            **common,
            "status": "available",
            "embedding": values.tolist(),
            "dimension": int(values.size),
            "dtype": "float32",
            "face_crop_path": str(crop_path.resolve(strict=True)),
            "backend_metadata": {
                "backend_name": "insightface_arcface",
                "backend_version": self.backend_version,
                "onnxruntime_version": self.onnxruntime_version,
                "detector_confidence": float(face.det_score),
                "detected_bbox_xyxy": np.asarray(face.bbox, dtype=float).tolist(),
                "landmarks_xy": landmarks.astype(float).tolist(),
                "alignment": "insightface_norm_crop_5point",
                "embedding_dimension": int(values.size),
            },
        }


def _emit(response: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(response, ensure_ascii=False, separators=(",", ":")))
    sys.stdout.write("\n")
    sys.stdout.flush()


def main() -> int:
    args = _parse_args()
    with contextlib.redirect_stdout(sys.stderr):
        worker = InsightFaceWorker(args)
    for line in sys.stdin:
        request_id = "unknown"
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise TypeError("embedding worker request must be an object")
            request_id = str(request.get("request_id", "unknown"))
            if request.get("operation") == "shutdown":
                _emit({"request_id": request_id, "status": "shutdown"})
                return 0
            if request.get("operation") != "face_embedding":
                raise ValueError("unsupported face embedding operation")
            with contextlib.redirect_stdout(sys.stderr):
                response = worker.process(request)
            _emit({"request_id": request_id, **response})
        except Exception as exc:  # noqa: BLE001 - isolate one inference request
            traceback.print_exc(file=sys.stderr)
            _emit(
                {
                    "request_id": request_id,
                    "status": "failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "model_identifier": args.model_identifier,
                    "model_fingerprint": args.model_fingerprint,
                }
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
