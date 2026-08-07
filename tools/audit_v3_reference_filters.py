from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from r2v_data_v2.v3.config import load_config
from r2v_data_v2.v3.reference_filter_audit import (
    ExternalReferenceFilterScorer,
    discover_local_models,
    run_reference_filter_audit,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit V3 reference quality without changing the source run",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--artifact-scope",
        choices=("candidates", "final", "both"),
        default="both",
    )
    parser.add_argument("--quality-backend", default="none")
    parser.add_argument("--quality-python", type=Path)
    parser.add_argument("--quality-code-root", type=Path)
    parser.add_argument("--quality-model-path", type=Path)
    parser.add_argument("--embedding-backend", default="none")
    parser.add_argument("--embedding-python", type=Path)
    parser.add_argument("--embedding-code-root", type=Path)
    parser.add_argument("--embedding-model-path", type=Path)
    parser.add_argument("--subject-pose-backend", default="none")
    parser.add_argument("--subject-pose-python", type=Path)
    parser.add_argument("--subject-pose-code-root", type=Path)
    parser.add_argument("--subject-pose-model-path", type=Path)
    parser.add_argument("--discover-local-models", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def _external_scorer(
    arguments: argparse.Namespace,
    *,
    kind: str,
) -> ExternalReferenceFilterScorer | None:
    backend = getattr(arguments, f"{kind}_backend")
    if backend == "none":
        return None
    python_executable = getattr(arguments, f"{kind}_python")
    code_root = getattr(arguments, f"{kind}_code_root")
    model_path = getattr(arguments, f"{kind}_model_path")
    provided = (python_executable, code_root, model_path)
    if all(value is None for value in provided):
        return None
    if any(value is None for value in provided):
        raise ValueError(
            f"{kind} external scorer requires python, code root, and model path"
        )
    worker_kind = "subject_pose" if kind == "subject_pose" else kind
    return ExternalReferenceFilterScorer(
        kind=worker_kind,
        backend=backend,
        python_executable=python_executable,
        code_root=code_root,
        model_path=model_path,
    )


def main(argv: list[str] | None = None) -> dict[str, object]:
    arguments = _parser().parse_args(argv)
    scorers: list[ExternalReferenceFilterScorer] = []
    try:
        quality_scorer = _external_scorer(arguments, kind="quality")
        embedding_scorer = _external_scorer(arguments, kind="embedding")
        subject_pose_scorer = _external_scorer(arguments, kind="subject_pose")
        scorers.extend(
            scorer
            for scorer in (
                quality_scorer,
                embedding_scorer,
                subject_pose_scorer,
            )
            if scorer is not None
        )
        discoveries = (
            discover_local_models() if arguments.discover_local_models else []
        )
        summary = run_reference_filter_audit(
            load_config(arguments.config),
            run_root=arguments.run_root,
            output_root=arguments.output_root,
            artifact_scope=arguments.artifact_scope,
            quality_backend=arguments.quality_backend,
            embedding_backend=arguments.embedding_backend,
            subject_pose_backend=arguments.subject_pose_backend,
            quality_scorer=quality_scorer,
            embedding_scorer=embedding_scorer,
            subject_pose_scorer=subject_pose_scorer,
            fail_fast=arguments.fail_fast,
            discoveries=discoveries,
        )
    finally:
        for scorer in reversed(scorers):
            scorer.close()
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
