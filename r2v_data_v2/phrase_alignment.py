from __future__ import annotations

import re
from collections.abc import Iterable

from r2v_data_v2.caption_validation import exact_phrase_spans

_ARTICLES = {"a", "an", "the"}
_MIN_SHARED_TOKEN_COUNT = 3
_MIN_NON_STOPWORD_COUNT = 2
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "between",
        "by",
        "for",
        "from",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "these",
        "this",
        "those",
        "through",
        "to",
        "under",
        "was",
        "were",
        "with",
    }
)
_WORD = re.compile(r"[^\W_]+", flags=re.UNICODE)


def _casefold_phrase_spans(
    caption: str,
    phrase: str,
) -> list[tuple[int, int]]:
    words = phrase.split()
    if not words:
        return []

    pattern = r"\s+".join(re.escape(word) for word in words)
    if words[0][0].isalnum():
        pattern = rf"(?<!\w){pattern}"
    if words[-1][-1].isalnum():
        pattern = rf"{pattern}(?!\w)"

    return [
        match.span()
        for match in re.finditer(
            pattern,
            caption,
            flags=re.IGNORECASE,
        )
    ]


def _word_tokens(value: str) -> list[tuple[str, int, int]]:
    return [
        (match.group(0).casefold(), *match.span())
        for match in _WORD.finditer(value)
    ]


def _resolve_longest_shared_token_span(
    caption: str,
    phrase: str,
) -> str | None:
    source_tokens = _word_tokens(phrase)
    caption_tokens = _word_tokens(caption)
    maximum_length = min(len(source_tokens), len(caption_tokens))

    for length in range(
        maximum_length,
        _MIN_SHARED_TOKEN_COUNT - 1,
        -1,
    ):
        candidates: dict[tuple[str, ...], tuple[int, int]] = {}
        checked: set[tuple[str, ...]] = set()
        for source_start in range(len(source_tokens) - length + 1):
            candidate = tuple(
                token
                for token, _, _ in source_tokens[
                    source_start : source_start + length
                ]
            )
            if candidate in checked:
                continue
            checked.add(candidate)
            if (
                sum(token not in _STOPWORDS for token in candidate)
                < _MIN_NON_STOPWORD_COUNT
            ):
                continue

            occurrences: list[tuple[int, int]] = []
            for caption_start in range(len(caption_tokens) - length + 1):
                caption_candidate = tuple(
                    token
                    for token, _, _ in caption_tokens[
                        caption_start : caption_start + length
                    ]
                )
                if caption_candidate != candidate:
                    continue
                occurrences.append(
                    (
                        caption_tokens[caption_start][1],
                        caption_tokens[caption_start + length - 1][2],
                    )
                )
            if len(occurrences) == 1:
                candidates[candidate] = occurrences[0]

        if len(candidates) > 1:
            return None
        if len(candidates) == 1:
            start, end = next(iter(candidates.values()))
            return caption[start:end]

    return None


def _resolve_casefold_candidates(
    caption: str,
    candidates: Iterable[str],
    *,
    fail_on_ambiguous: bool,
) -> str | None:
    seen: set[str] = set()
    for candidate in candidates:
        key = candidate.casefold()
        if key in seen:
            continue
        seen.add(key)

        spans = _casefold_phrase_spans(caption, candidate)
        if len(spans) == 1:
            start, end = spans[0]
            return caption[start:end]
        if fail_on_ambiguous and len(spans) > 1:
            return None
    return None


def resolve_reference_caption_phrase(
    caption: str,
    phrase: str,
) -> str | None:
    """Preserve the conservative entity phrase repair behavior."""
    words = phrase.split()
    candidates = [phrase]

    # Common model error:
    # "an enemy soldier" while caption contains "an enemy".
    if len(words) >= 3:
        candidates.append(" ".join(words[:-1]))

    # Common article mismatch:
    # "the red vehicle" while caption contains "red vehicle".
    if len(words) >= 3 and words[0].casefold() in _ARTICLES:
        candidates.append(" ".join(words[1:]))

    return _resolve_casefold_candidates(
        caption,
        candidates,
        fail_on_ambiguous=False,
    )


def resolve_background_caption_phrase(
    caption: str,
    phrase: str,
) -> str | None:
    """Resolve a background phrase to one exact, unique caption substring."""
    exact_spans = exact_phrase_spans(caption, phrase)
    if len(exact_spans) == 1:
        start, end = exact_spans[0]
        return caption[start:end]
    if len(exact_spans) > 1:
        return None

    casefold_spans = _casefold_phrase_spans(caption, phrase)
    if len(casefold_spans) > 1:
        return None
    if len(casefold_spans) == 1:
        start, end = casefold_spans[0]
        return caption[start:end]

    words = phrase.split()
    variants: list[str] = []
    if len(words) >= 2 and words[0].casefold() in _ARTICLES:
        variants.append(" ".join(words[1:]))
    if len(words) >= 3:
        variants.append(" ".join(words[:-1]))

    resolved = _resolve_casefold_candidates(
        caption,
        variants,
        fail_on_ambiguous=True,
    )
    if resolved is not None:
        return resolved
    if any(
        len(_casefold_phrase_spans(caption, candidate)) > 1
        for candidate in variants
    ):
        return None
    return _resolve_longest_shared_token_span(caption, phrase)
