"""Local media transport checks; no HTTP server or model requests."""

import base64
from pathlib import Path

import pytest

from r2v_data_v2.h3.mimo25_backend import MimoMediaResolver
from tools.run_h3_mimo25_stem_reconcile_shadow import _parser


def test_http_persistent_media_quoted_without_reading_bytes(tmp_path, monkeypatch):
    folder = tmp_path / "foo"
    folder.mkdir()
    paths = [folder / name for name in ("bar clip.mp4", "ref #1.png", "music.wav")]
    for path in paths:
        path.write_bytes(b"persistent media")
    def forbidden(*args, **kwargs):
        raise AssertionError("HTTP must not read or base64 encode persistent media")
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(base64, "b64encode", forbidden)
    resolver = MimoMediaResolver(mode="http", media_root=tmp_path, media_base_url="http://127.0.0.1:8766/")
    assert [resolver.resolve(path) for path in paths] == [
        "http://127.0.0.1:8766/foo/bar%20clip.mp4",
        "http://127.0.0.1:8766/foo/ref%20%231.png",
        "http://127.0.0.1:8766/foo/music.wav",
    ]


def test_base64_media_unchanged(tmp_path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"frozen video bytes")
    result = MimoMediaResolver(mode="base64", media_root=tmp_path).resolve(path)
    assert result == "data:video/mp4;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


@pytest.mark.parametrize("environment,explicit,expected", [
    ({}, [], ("base64", None)),
    ({"MIMO_MEDIA_MODE": "http", "MIMO_MEDIA_BASE_URL": "http://127.0.0.1:8766/"}, [],
     ("http", "http://127.0.0.1:8766/")),
    ({"MIMO_MEDIA_MODE": "base64", "MIMO_MEDIA_BASE_URL": "http://env.invalid/"},
     ["--media-mode", "http", "--media-base-url", "http://explicit.invalid/"], ("http", "http://explicit.invalid/")),
    ({"MIMO_MEDIA_MODE": "http"}, ["--media-mode", "base64"], ("base64", None)),
])
def test_runner_media_environment_defaults_and_explicit_precedence(monkeypatch, environment, explicit, expected):
    for name in ("MIMO_MEDIA_MODE", "MIMO_MEDIA_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    args = _parser().parse_args([
        "--visual-production-root", "/visual", "--visual-runs-root", "/runs",
        "--audio-production-root", "/audio", "--case-manifest", "/cases.json", *explicit,
    ])
    assert (args.media_mode, args.media_base_url) == expected
    assert args.media_root == Path("/mnt/workspace")


def test_http_requires_base_url(tmp_path):
    with pytest.raises(ValueError, match="HTTP media mode requires.*base URL"):
        MimoMediaResolver(mode="http", media_root=tmp_path)
