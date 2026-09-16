import json
from pathlib import Path

import pytest

from r2v_data_v2.h3.training_manifest_export import export_training_manifests


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def test_exports_r2va_and_ra2va_as_flat_loader_rows_without_hashes(tmp_path: Path):
    shadow = tmp_path / 'shadow'
    prepared = shadow / 'h3_audio_reuse_prepared_v1'
    products = shadow / 'h3_audio_reuse_products_v1'
    video = '/data/clip-a.mp4'
    image1 = '/data/p1.png'
    image2 = '/data/p2.png'
    audio = '/data/S1.flac'

    write_json(prepared / 'inventory.json', {
        'jobs': [{
            'clip_uid': 'clip-a',
            'target_video_path': video,
            'reference_images': [
                {'image_index': 1, 'image_artifact_path': image1},
                {'image_index': 2, 'image_artifact_path': image2},
            ],
        }]
    })
    write_jsonl(products / 'records.jsonl', [
        {
            'clip_uid': 'clip-a',
            'status': 'ready',
            'conditioning_variant': 'visual_only',
            'audio_references': [],
            'rendered_h3_prompt': 'r2va caption',
            'source_h3_sample_sha256': 'deadbeef',
        },
        {
            'clip_uid': 'clip-a',
            'status': 'ready',
            'conditioning_variant': 'target_speech_reuse',
            'audio_references': [
                {'contract': {'path': audio, 'sha256': 'ignored'}}
            ],
            'rendered_h3_prompt': 'ra2va caption',
        },
        {
            'clip_uid': 'clip-a',
            'status': 'failed',
            'conditioning_variant': 'full_audio_reuse',
            'audio_references': [],
            'rendered_h3_prompt': None,
        },
    ])

    output = tmp_path / 'export'
    export_training_manifests(output_root=output, ra2va_shadow_root=shadow)

    assert read_jsonl(output / 'r2va.jsonl') == [{
        'video': video,
        'images': [image1, image2],
        'audios': [],
        'caption': 'r2va caption',
    }]
    assert read_jsonl(output / 'ra2va.jsonl') == [{
        'video': video,
        'images': [image1, image2],
        'audios': [audio],
        'caption': 'ra2va caption',
    }]
    exported = (output / 'r2va.jsonl').read_text() + (output / 'ra2va.jsonl').read_text()
    assert 'sha256' not in exported
    assert 'fingerprint' not in exported


def test_exports_t2va_and_ta2va_and_builds_video_task_summary(tmp_path: Path):
    t2va = tmp_path / 't2va'
    ta2va = tmp_path / 'ta2va'
    video = '/data/clip-b.mp4'
    audio1 = '/data/full.flac'
    audio2 = '/data/S1.flac'

    write_json(t2va / 'inventory.json', {
        'jobs': [
            {'clip_uid': 'clip-b', 'target_video_path': video},
            {'clip_uid': 'clip-failed', 'target_video_path': '/data/failed.mp4'},
        ]
    })
    write_jsonl(t2va / 'records.jsonl', [
        {'clip_uid': 'clip-b', 'status': 'ready'},
        {'clip_uid': 'clip-failed', 'status': 'failed'},
    ])
    (t2va / 'prompts').mkdir(parents=True)
    (t2va / 'prompts' / 'clip-b.txt').write_text('t2va caption\n', encoding='utf-8')

    write_json(ta2va / 'inventory.json', {'source_t2va_root': str(t2va)})
    write_jsonl(ta2va / 'records.jsonl', [
        {
            'clip_uid': 'clip-b',
            'variant': 'full_audio_reuse',
            'status': 'ready',
            'audio_references': [{'path': audio1, 'sha256': 'ignored'}],
            'prompt': 'ta2va full caption',
        },
        {
            'clip_uid': 'clip-b',
            'variant': 'target_speech_reuse',
            'status': 'ready',
            'audio_references': [{'path': audio2}],
            'prompt': 'ta2va speech caption',
        },
    ])

    output = tmp_path / 'export'
    export_training_manifests(output_root=output, t2va_root=t2va, ta2va_root=ta2va)

    assert read_jsonl(output / 't2va.jsonl') == [{
        'video': video,
        'images': [],
        'audios': [],
        'caption': 't2va caption\n',
    }]
    assert read_jsonl(output / 'ta2va.jsonl') == [
        {'video': video, 'images': [], 'audios': [audio1], 'caption': 'ta2va full caption'},
        {'video': video, 'images': [], 'audios': [audio2], 'caption': 'ta2va speech caption'},
    ]
    assert read_jsonl(output / 'videos.jsonl') == [{
        'video': video,
        'tasks': ['t2va', 'ta2va'],
    }]


def test_summary_merges_all_available_tasks_for_same_video(tmp_path: Path):
    shadow = tmp_path / 'shadow'
    t2va = tmp_path / 't2va'
    video = '/data/shared.mp4'
    write_json(shadow / 'h3_audio_reuse_prepared_v1' / 'inventory.json', {
        'jobs': [{'clip_uid': 'shared', 'target_video_path': video, 'reference_images': []}]
    })
    write_jsonl(shadow / 'h3_audio_reuse_products_v1' / 'records.jsonl', [
        {'clip_uid': 'shared', 'status': 'ready', 'conditioning_variant': 'visual_only',
         'audio_references': [], 'rendered_h3_prompt': 'r'},
        {'clip_uid': 'shared', 'status': 'ready', 'conditioning_variant': 'full_audio_reuse',
         'audio_references': [{'contract': {'path': '/data/a.flac'}}], 'rendered_h3_prompt': 'ra'},
    ])
    write_json(t2va / 'inventory.json', {'jobs': [{'clip_uid': 'shared', 'target_video_path': video}]})
    write_jsonl(t2va / 'records.jsonl', [{'clip_uid': 'shared', 'status': 'ready'}])
    (t2va / 'prompts').mkdir(parents=True)
    (t2va / 'prompts' / 'shared.txt').write_text('t', encoding='utf-8')

    output = tmp_path / 'export'
    export_training_manifests(output_root=output, ra2va_shadow_root=shadow, t2va_root=t2va)

    assert read_jsonl(output / 'videos.jsonl') == [{
        'video': video,
        'tasks': ['r2va', 'ra2va', 't2va'],
    }]
    for name in ('videos', 'r2va', 'ra2va', 't2va', 'ta2va'):
        assert (output / f'{name}.jsonl').is_file()


def test_rejects_existing_output_directory(tmp_path: Path):
    output = tmp_path / 'export'
    output.mkdir()
    with pytest.raises(FileExistsError):
        export_training_manifests(output_root=output)


def test_cli_exports_empty_task_files_with_only_output_root(tmp_path: Path):
    import subprocess
    import sys

    output = tmp_path / 'cli-export'
    script = Path(__file__).resolve().parents[1] / 'tools' / 'export_h3_training_manifests.py'
    env = dict(__import__('os').environ)
    env.pop('PYTHONPATH', None)
    result = subprocess.run(
        [sys.executable, str(script), '--output-root', str(output)],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert read_jsonl(output / 'videos.jsonl') == []
    for task in ('r2va', 'ra2va', 't2va', 'ta2va'):
        assert read_jsonl(output / f'{task}.jsonl') == []
