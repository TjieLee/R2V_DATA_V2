from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from r2v_data_v2.h3.audio_schemas import (
    AudioClipBinding,
    AudioPairSample,
    DraftAnnotation,
    EntityOccurrence,
    PairEvidence,
    PairSubjectAudioBinding,
    PairSubjectBinding,
    PairTarget,
    ProducerProvenance,
    SamePersonEvidence,
    SameVoiceEvidence,
    SpeechTurn,
)


@dataclass(frozen=True)
class AudioPairingConfig:
    top_k: int = 20
    block_size: int = 256
    face_strict_threshold: float = 0.70
    face_without_text_strict_threshold: float = 0.75
    face_margin_threshold: float = 0.04
    voice_strict_threshold: float = 0.75
    voice_margin_threshold: float = 0.04
    text_threshold: float = 0.30
    minimum_local_binding_confidence: float = 0.80
    max_cross_pair_variants_per_target: int = 1
    deterministic_seed: int = 0
    renderer_profile: str = "h3_v1"
    speech_open_tag: str = "<d>"
    speech_close_tag: str = "</d>"

    def __post_init__(self) -> None:
        if self.top_k <= 0 or self.block_size <= 0:
            raise ValueError("pair top-K and block size must be positive")
        thresholds = (
            self.face_strict_threshold,
            self.face_without_text_strict_threshold,
            self.face_margin_threshold,
            self.voice_strict_threshold,
            self.voice_margin_threshold,
            self.text_threshold,
            self.minimum_local_binding_confidence,
        )
        if any(not -1 <= value <= 1 for value in thresholds):
            raise ValueError("pair thresholds must be finite cosine/probability values")
        if self.face_without_text_strict_threshold < self.face_strict_threshold:
            raise ValueError("missing-text face threshold must not be weaker")
        if self.max_cross_pair_variants_per_target < 0:
            raise ValueError("max cross-pair variants must be non-negative")
        if self.deterministic_seed < 0:
            raise ValueError("deterministic seed must be non-negative")
        if (
            not self.renderer_profile.strip()
            or not self.speech_open_tag
            or not self.speech_close_tag
            or self.speech_open_tag == self.speech_close_tag
        ):
            raise ValueError("speech renderer profile/tags are invalid")

    def fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def _normalize_matrix(vectors: np.ndarray) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("candidate index requires a non-empty 2D matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("candidate vectors must be finite")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("candidate vectors must have positive norm")
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


def _bounded_cosine(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


class NumpyBlockwiseTopKIndex:
    def __init__(self, *, block_size: int = 256) -> None:
        if block_size <= 0:
            raise ValueError("blockwise index block_size must be positive")
        self.block_size = block_size
        self.maximum_similarity_block_shape = (0, 0)

    def search(
        self,
        vectors: np.ndarray,
        *,
        top_k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        matrix = _normalize_matrix(vectors)
        count = matrix.shape[0]
        effective_k = min(top_k, max(0, count - 1))
        indices = np.full((count, effective_k), -1, dtype=np.int64)
        scores = np.full((count, effective_k), -np.inf, dtype=np.float32)
        for query_index in range(count):
            best: list[tuple[float, int]] = []
            query = matrix[query_index : query_index + 1]
            for start in range(0, count, self.block_size):
                stop = min(count, start + self.block_size)
                block = query @ matrix[start:stop].T
                self.maximum_similarity_block_shape = max(
                    self.maximum_similarity_block_shape,
                    block.shape,
                )
                for offset, score in enumerate(block[0].tolist()):
                    candidate_index = start + offset
                    if candidate_index != query_index:
                        best.append((float(score), candidate_index))
                best = sorted(best, key=lambda item: (-item[0], item[1]))[:effective_k]
            for rank, (score, candidate_index) in enumerate(best):
                indices[query_index, rank] = candidate_index
                scores[query_index, rank] = score
        return indices, scores


class FaissTopKIndex:
    """Optional exact FAISS adapter loaded only when explicitly instantiated."""

    def search(
        self,
        vectors: np.ndarray,
        *,
        top_k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        try:
            import faiss  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("FAISS backend requires optional faiss installation") from exc
        matrix = _normalize_matrix(vectors)
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        scores, indices = index.search(matrix, min(top_k + 1, matrix.shape[0]))
        output_indices: list[list[int]] = []
        output_scores: list[list[float]] = []
        for query_index in range(indices.shape[0]):
            row_indices = indices[query_index]
            row_scores = scores[query_index]
            filtered = [
                (int(row_indices[position]), float(row_scores[position]))
                for position in range(len(row_indices))
                if int(row_indices[position]) != query_index
            ][: min(top_k, matrix.shape[0] - 1)]
            filtered.sort(key=lambda item: (-item[1], item[0]))
            output_indices.append([item[0] for item in filtered])
            output_scores.append([item[1] for item in filtered])
        return (
            np.asarray(output_indices, dtype=np.int64),
            np.asarray(output_scores, dtype=np.float32),
        )


def _load_embedding(root: Path, occurrence: EntityOccurrence, kind: str) -> np.ndarray:
    asset = {
        "face": occurrence.face_embedding_asset,
        "voice": occurrence.voice_embedding_asset,
        "text": occurrence.identity_text_embedding_asset,
    }[kind]
    if asset is None:
        raise ValueError(f"occurrence lacks {kind} embedding")
    path = (root / asset.path).resolve(strict=True)
    path.relative_to(root.resolve(strict=True))
    if hashlib.sha256(path.read_bytes()).hexdigest() != asset.sha256:
        raise ValueError(f"{kind} embedding SHA-256 mismatch")
    with path.open("rb") as stream:
        vector = np.load(stream, allow_pickle=False)
    flattened = _normalize_matrix(np.asarray(vector).reshape(1, -1))[0]
    if flattened.size != asset.dimension:
        raise ValueError(f"{kind} embedding dimension mismatch")
    return flattened


def _neighbor_maps(
    occurrences: list[EntityOccurrence],
    vectors: np.ndarray,
    *,
    top_k: int,
    index_backend: object,
) -> tuple[dict[str, list[str]], dict[tuple[str, str], float]]:
    indices, scores = index_backend.search(vectors, top_k=top_k)  # type: ignore[attr-defined]
    neighbors: dict[str, list[str]] = {}
    similarities: dict[tuple[str, str], float] = {}
    for row, occurrence in enumerate(occurrences):
        ids: list[str] = []
        if len(indices[row]) != len(scores[row]):
            raise ValueError("candidate index returned misaligned indices/scores")
        for position in range(len(indices[row])):
            candidate_index = indices[row][position]
            score = scores[row][position]
            if int(candidate_index) < 0:
                continue
            candidate_id = occurrences[int(candidate_index)].entity_occurrence_id
            ids.append(candidate_id)
            similarities[(occurrence.entity_occurrence_id, candidate_id)] = (
                _bounded_cosine(float(score))
            )
        neighbors[occurrence.entity_occurrence_id] = ids
    return neighbors, similarities


def _rank(neighbors: dict[str, list[str]], left: str, right: str) -> int | None:
    try:
        return neighbors[left].index(right) + 1
    except (KeyError, ValueError):
        return None


def _pair_margin(
    *,
    left: str,
    right: str,
    score: float,
    neighbors: dict[str, list[str]],
    similarities: dict[tuple[str, str], float],
) -> float:
    alternatives = [
        similarities[(left, candidate)]
        for candidate in neighbors.get(left, [])
        if candidate != right
    ]
    return score - (max(alternatives) if alternatives else -1.0)


def _draft_annotation(
    binding: AudioClipBinding,
    *,
    subject_occurrences: list[EntityOccurrence],
    speech_turns: list[SpeechTurn],
    config: AudioPairingConfig,
) -> DraftAnnotation:
    subject_index = {
        occurrence.entity_id: index
        for index, occurrence in enumerate(subject_occurrences, start=1)
    }
    speech_lines: list[str] = []
    for turn in speech_turns:
        if turn.text is None:
            continue
        if turn.entity_id not in subject_index:
            raise ValueError("published transcript has an unknown subject")
        speech_lines.append(
            f"subject_{subject_index[turn.entity_id]} says "
            f"{config.speech_open_tag}{turn.text}{config.speech_close_tag}"
        )
    text = binding.visual_r2v_instruction.strip()
    if speech_lines:
        text = f"{text}\n" + "\n".join(speech_lines)
    if not text:
        text = binding.visual_t2v_caption.strip()
    if not text:
        raise ValueError("draft annotation requires existing visual text")
    if text.count(config.speech_open_tag) != text.count(config.speech_close_tag):
        raise ValueError("speech delimiter tags must be paired")
    if f"{config.speech_open_tag}{config.speech_close_tag}" in text:
        raise ValueError("empty pseudo-dialogue is forbidden")
    payload = json.dumps(
        {
            "clip_binding_id": binding.clip_binding_id,
            "subjects": [item.entity_occurrence_id for item in subject_occurrences],
            "speech_turns": [
                turn.model_dump(mode="json") for turn in speech_turns
            ],
            "profile": config.renderer_profile,
            "open": config.speech_open_tag,
            "close": config.speech_close_tag,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return DraftAnnotation(
        renderer_profile=config.renderer_profile,
        renderer_version="v1",
        input_sha256=hashlib.sha256(payload).hexdigest(),
        text=text,
    )


def build_pairwise_evidence(
    bindings: list[AudioClipBinding],
    *,
    audio_root: Path,
    config: AudioPairingConfig | None = None,
    face_index_backend: object | None = None,
    voice_index_backend: object | None = None,
) -> list[PairEvidence]:
    active = config or AudioPairingConfig()
    occurrences = sorted(
        (
            occurrence
            for binding in bindings
            for occurrence in binding.entity_occurrences
            if occurrence.cross_pair_eligible
            and occurrence.local_binding_summary.best_binding_confidence
            >= active.minimum_local_binding_confidence
        ),
        key=lambda item: item.entity_occurrence_id,
    )
    if len(occurrences) < 2:
        return []
    face_vectors = np.stack(
        [_load_embedding(audio_root, item, "face") for item in occurrences]
    )
    voice_vectors = np.stack(
        [_load_embedding(audio_root, item, "voice") for item in occurrences]
    )
    face_backend = face_index_backend or NumpyBlockwiseTopKIndex(
        block_size=active.block_size
    )
    voice_backend = voice_index_backend or NumpyBlockwiseTopKIndex(
        block_size=active.block_size
    )
    face_neighbors, face_scores = _neighbor_maps(
        occurrences,
        face_vectors,
        top_k=active.top_k,
        index_backend=face_backend,
    )
    voice_neighbors, voice_scores = _neighbor_maps(
        occurrences,
        voice_vectors,
        top_k=active.top_k,
        index_backend=voice_backend,
    )
    by_id = {item.entity_occurrence_id: item for item in occurrences}
    candidates = sorted(set(face_scores).intersection(voice_scores))
    evidence: list[PairEvidence] = []
    for left_id, right_id in candidates:
        left = by_id[left_id]
        right = by_id[right_id]
        left_clip = left_id.split("/", 1)[0]
        right_clip = right_id.split("/", 1)[0]
        if left_clip == right_clip:
            continue
        face_score = face_scores[(left_id, right_id)]
        voice_score = voice_scores[(left_id, right_id)]
        face_reverse = face_scores.get((right_id, left_id))
        voice_reverse = voice_scores.get((right_id, left_id))
        face_mutual = face_reverse is not None
        voice_mutual = voice_reverse is not None
        face_margin = min(
            _pair_margin(
                left=left_id,
                right=right_id,
                score=face_score,
                neighbors=face_neighbors,
                similarities=face_scores,
            ),
            _pair_margin(
                left=right_id,
                right=left_id,
                score=face_reverse if face_reverse is not None else -1.0,
                neighbors=face_neighbors,
                similarities=face_scores,
            ),
        )
        voice_margin = min(
            _pair_margin(
                left=left_id,
                right=right_id,
                score=voice_score,
                neighbors=voice_neighbors,
                similarities=voice_scores,
            ),
            _pair_margin(
                left=right_id,
                right=left_id,
                score=voice_reverse if voice_reverse is not None else -1.0,
                neighbors=voice_neighbors,
                similarities=voice_scores,
            ),
        )
        text_similarity = None
        text_available = (
            left.identity_text_specific
            and right.identity_text_specific
            and left.identity_text_embedding_asset is not None
            and right.identity_text_embedding_asset is not None
        )
        if text_available:
            text_similarity = _bounded_cosine(
                np.dot(
                    _load_embedding(audio_root, left, "text"),
                    _load_embedding(audio_root, right, "text"),
                )
            )
        face_threshold = (
            active.face_strict_threshold
            if text_available
            else active.face_without_text_strict_threshold
        )
        person_reasons: list[str] = []
        if face_score < face_threshold:
            person_reasons.append("face_below_strict_threshold")
        if not face_mutual:
            person_reasons.append("face_not_mutual_top_k")
        if face_margin < active.face_margin_threshold:
            person_reasons.append("face_margin_too_small")
        if text_similarity is not None and text_similarity < active.text_threshold:
            person_reasons.append("identity_text_conflict")
        person_accepted = not person_reasons
        person_candidate = (
            not person_accepted and face_score >= active.face_strict_threshold
        )
        voice_reasons: list[str] = []
        if voice_score < active.voice_strict_threshold:
            voice_reasons.append("voice_below_strict_threshold")
        if not voice_mutual:
            voice_reasons.append("voice_not_mutual_top_k")
        if voice_margin < active.voice_margin_threshold:
            voice_reasons.append("voice_margin_too_small")
        voice_accepted = not voice_reasons
        voice_candidate = (
            not voice_accepted and voice_score >= active.voice_strict_threshold
        )
        evidence.append(
            PairEvidence(
                target_entity_occurrence_id=left_id,
                reference_entity_occurrence_id=right_id,
                same_person=SamePersonEvidence(
                    status=(
                        "accepted"
                        if person_accepted
                        else "candidate"
                        if person_candidate
                        else "rejected"
                    ),
                    face_similarity=face_score,
                    text_similarity=text_similarity,
                    face_rank_left_to_right=_rank(face_neighbors, left_id, right_id),
                    face_rank_right_to_left=_rank(face_neighbors, right_id, left_id),
                    face_margin=face_margin,
                    policy_version="strict_pairwise_v1",
                    reason_codes=person_reasons or ["strict_face_policy_passed"],
                ),
                same_voice=SameVoiceEvidence(
                    status=(
                        "accepted"
                        if voice_accepted
                        else "candidate"
                        if voice_candidate
                        else "rejected"
                    ),
                    voice_similarity=voice_score,
                    voice_rank_left_to_right=_rank(voice_neighbors, left_id, right_id),
                    voice_rank_right_to_left=_rank(voice_neighbors, right_id, left_id),
                    voice_margin=voice_margin,
                    policy_version="strict_pairwise_v1",
                    reason_codes=voice_reasons or ["strict_voice_policy_passed"],
                ),
                combined_score=face_score + voice_score + (text_similarity or 0.0),
            )
        )
    return evidence


def _target(binding: AudioClipBinding) -> PairTarget:
    return PairTarget(
        clip_binding_id=binding.clip_binding_id,
        clip_uid=binding.clip_uid,
        video=binding.source_video,
        full_audio=binding.full_audio,
    )


def _producer(config: AudioPairingConfig) -> ProducerProvenance:
    return ProducerProvenance(
        producer="r2v_data_v2.h3.audio_pairing",
        version="v1",
        config_fingerprint=config.fingerprint(),
    )


def _published_speech_turns(
    binding: AudioClipBinding,
    subjects: list[EntityOccurrence],
) -> list[SpeechTurn]:
    entity_ids = {item.entity_id for item in subjects}
    return [
        turn
        for turn in binding.speech_turns
        if turn.status == "bound" and turn.entity_id in entity_ids
    ]


def build_audio_pair_samples(
    bindings: list[AudioClipBinding],
    *,
    audio_root: Path,
    config: AudioPairingConfig | None = None,
    face_index_backend: object | None = None,
    voice_index_backend: object | None = None,
) -> tuple[list[AudioPairSample], list[PairEvidence], dict[str, object]]:
    active = config or AudioPairingConfig()
    ordered_bindings = sorted(bindings, key=lambda item: item.clip_uid)
    pairwise = build_pairwise_evidence(
        ordered_bindings,
        audio_root=audio_root,
        config=active,
        face_index_backend=face_index_backend,
        voice_index_backend=voice_index_backend,
    )
    edge_by_target: dict[str, list[PairEvidence]] = {}
    for edge in pairwise:
        if (
            edge.same_person.status == "accepted"
            and edge.same_voice.status == "accepted"
        ):
            edge_by_target.setdefault(edge.target_entity_occurrence_id, []).append(edge)
    binding_by_clip = {item.clip_uid: item for item in ordered_bindings}
    occurrence_by_id = {
        occurrence.entity_occurrence_id: occurrence
        for binding in ordered_bindings
        for occurrence in binding.entity_occurrences
    }
    samples: list[AudioPairSample] = []
    incomplete_cross: list[dict[str, object]] = []
    producer = _producer(active)
    for binding in ordered_bindings:
        speaking = [
            item
            for item in binding.entity_occurrences
            if item.in_pair_eligible and item.primary_voice_reference is not None
        ]
        if speaking:
            speech_turns = _published_speech_turns(binding, speaking)
            subject_bindings = [
                PairSubjectBinding(
                    subject_id=f"subject_{index}",
                    target_entity_occurrence_id=item.entity_occurrence_id,
                    picture_entity_occurrence_id=item.entity_occurrence_id,
                    voice_entity_occurrence_id=item.entity_occurrence_id,
                    target_entity_id=item.entity_id,
                )
                for index, item in enumerate(speaking, start=1)
            ]
            samples.append(
                AudioPairSample(
                    pair_id=f"in_pair/{binding.clip_uid}",
                    pair_kind="in_pair",
                    source_clip_binding_ids=[binding.clip_binding_id],
                    target=_target(binding),
                    subjects=subject_bindings,
                    voice_references=[
                        item.primary_voice_reference for item in speaking
                    ],
                    subject_audio_bindings=[
                        PairSubjectAudioBinding(
                            subject_id=f"subject_{index}",
                            voice_reference_id=(
                                item.primary_voice_reference.voice_reference_id
                            ),
                            entity_occurrence_id=item.entity_occurrence_id,
                        )
                        for index, item in enumerate(speaking, start=1)
                        if item.primary_voice_reference is not None
                    ],
                    speech_turns=speech_turns,
                    pair_evidence=[],
                    annotation_draft=_draft_annotation(
                        binding,
                        subject_occurrences=speaking,
                        speech_turns=speech_turns,
                        config=active,
                    ),
                    producer_provenance=producer,
                )
            )
        if active.max_cross_pair_variants_per_target == 0:
            continue
        if not speaking:
            continue
        if any(not item.cross_pair_eligible for item in speaking):
            incomplete_cross.append(
                {
                    "clip_uid": binding.clip_uid,
                    "reason": "not_all_speaking_subjects_are_cross_pair_eligible",
                }
            )
            continue
        cross_targets = speaking
        selected_edges: list[PairEvidence] = []
        reference_occurrences: list[EntityOccurrence] = []
        used_reference_ids: set[str] = set()
        complete = True
        for target_occurrence in cross_targets:
            candidates = []
            for edge in edge_by_target.get(target_occurrence.entity_occurrence_id, []):
                reference = occurrence_by_id[edge.reference_entity_occurrence_id]
                if reference.entity_occurrence_id in used_reference_ids:
                    continue
                reference_clip = binding_by_clip[
                    reference.entity_occurrence_id.split("/", 1)[0]
                ]
                if (
                    reference.primary_voice_reference is None
                    or not reference.local_binding_summary.high_confidence_bound
                    or reference.local_binding_summary.best_binding_confidence
                    < active.minimum_local_binding_confidence
                    or reference_clip.source_video.sha256 == binding.source_video.sha256
                ):
                    continue
                target_voice = target_occurrence.primary_voice_reference
                reference_voice = reference.primary_voice_reference
                assert target_voice is not None
                if (
                    reference_clip.full_audio.asset.sha256
                    == binding.full_audio.asset.sha256
                    and target_voice.source_start == reference_voice.source_start
                    and target_voice.source_end == reference_voice.source_end
                ):
                    continue
                candidates.append((edge, reference))
            if not candidates:
                complete = False
                break
            edge, reference = min(
                candidates,
                key=lambda item: (
                    -item[0].combined_score,
                    item[1].entity_occurrence_id,
                ),
            )
            selected_edges.append(edge)
            reference_occurrences.append(reference)
            used_reference_ids.add(reference.entity_occurrence_id)
        if not complete:
            incomplete_cross.append(
                {
                    "clip_uid": binding.clip_uid,
                    "reason": "not_all_speaking_subjects_have_strict_reference",
                }
            )
            continue
        subject_bindings = [
            PairSubjectBinding(
                subject_id=f"subject_{index}",
                target_entity_occurrence_id=target.entity_occurrence_id,
                picture_entity_occurrence_id=target.entity_occurrence_id,
                voice_entity_occurrence_id=reference.entity_occurrence_id,
                target_entity_id=target.entity_id,
            )
            for index, (target, reference) in enumerate(
                (
                    (target, reference_occurrences[target_index])
                    for target_index, target in enumerate(cross_targets)
                ),
                start=1,
            )
        ]
        source_ids = [binding.clip_binding_id]
        source_ids.extend(
            sorted(
                {
                    binding_by_clip[
                        item.entity_occurrence_id.split("/", 1)[0]
                    ].clip_binding_id
                    for item in reference_occurrences
                }
            )
        )
        samples.append(
            AudioPairSample(
                pair_id=f"cross_pair/{binding.clip_uid}/1",
                pair_kind="cross_pair",
                source_clip_binding_ids=source_ids,
                target=_target(binding),
                subjects=subject_bindings,
                voice_references=[
                    item.primary_voice_reference for item in reference_occurrences
                ],
                subject_audio_bindings=[
                    PairSubjectAudioBinding(
                        subject_id=f"subject_{index}",
                        voice_reference_id=(
                            item.primary_voice_reference.voice_reference_id
                        ),
                        entity_occurrence_id=item.entity_occurrence_id,
                    )
                    for index, item in enumerate(reference_occurrences, start=1)
                    if item.primary_voice_reference is not None
                ],
                speech_turns=_published_speech_turns(binding, cross_targets),
                pair_evidence=selected_edges,
                annotation_draft=_draft_annotation(
                    binding,
                    subject_occurrences=cross_targets,
                    speech_turns=_published_speech_turns(binding, cross_targets),
                    config=active,
                ),
                producer_provenance=producer,
            )
        )
    samples.sort(key=lambda item: (item.target.clip_uid, item.pair_kind, item.pair_id))
    report = {
        "thresholds_calibrated": False,
        "config": asdict(active),
        "pairwise_edge_count": len(pairwise),
        "in_pair_count": sum(item.pair_kind == "in_pair" for item in samples),
        "cross_pair_count": sum(item.pair_kind == "cross_pair" for item in samples),
        "incomplete_cross_pair_targets": incomplete_cross,
        "transitive_clustering_performed": False,
    }
    return samples, pairwise, report
