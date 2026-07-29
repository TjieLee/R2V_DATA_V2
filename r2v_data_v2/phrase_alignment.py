from __future__ import annotations

import re
from collections.abc import Iterable

from r2v_data_v2.caption_validation import exact_phrase_spans

_ARTICLES = {"a", "an", "the"}


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

    return _resolve_casefold_candidates(
        caption,
        variants,
        fail_on_ambiguous=True,
    )
