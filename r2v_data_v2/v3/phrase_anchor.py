from __future__ import annotations

import re

_PHRASE_EDGE_PUNCTUATION = (
    " \t\r\n.,;:!?\"'`()[]{}"
    "\u2018\u2019\u201c\u201d"
    "\uff0c\u3002\uff1b\uff1a\uff01\uff1f"
)


def normalize_caption_phrase(value: str) -> str:
    return " ".join(
        value.strip(_PHRASE_EDGE_PUNCTUATION).casefold().split()
    )


def find_caption_phrase_spans(
    *,
    phrase: str,
    caption: str,
) -> tuple[tuple[int, int], ...]:
    normalized_phrase = normalize_caption_phrase(phrase)
    if not normalized_phrase:
        return ()
    pattern = re.compile(
        r"(?<!\w)"
        + r"\s+".join(re.escape(part) for part in normalized_phrase.split())
        + r"(?!\w)",
        flags=re.IGNORECASE,
    )
    return tuple(match.span() for match in pattern.finditer(caption))


def phrase_matches_caption_span(*, phrase: str, caption: str) -> bool:
    return bool(find_caption_phrase_spans(phrase=phrase, caption=caption))
