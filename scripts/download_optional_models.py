from __future__ import annotations

import argparse
from pathlib import Path

ALLOWED_DESTINATION_ROOTS = (
    Path("/mnt/workspace/litengjie/data/models").resolve(),
    Path("/mnt/workspace/litengjie/data/cache/huggingface").resolve(),
)


def validate_destination(destination: Path) -> Path:
    resolved = destination.expanduser().resolve(strict=False)
    if not any(
        resolved == root or root in resolved.parents
        for root in ALLOWED_DESTINATION_ROOTS
    ):
        raise ValueError(
            "destination must be inside /mnt/workspace/litengjie/data/models "
            "or /mnt/workspace/litengjie/data/cache/huggingface"
        )
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explicitly download optional pipeline models"
    )
    parser.add_argument("--siglip2", required=True, help="Hugging Face repository ID")
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()

    destination = validate_destination(args.destination)
    destination.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=args.siglip2,
        local_dir=destination,
    )
    print(destination)


if __name__ == "__main__":
    main()
