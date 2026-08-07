from __future__ import annotations

import re

_PHRASE_EDGE_PUNCTUATION = (
    " \t\r\n.,;:!?\"'`()[]{}"
    "\u2018\u2019\u201c\u201d"
    "\uff0c\u3002\uff1b\uff1a\uff01\uff1f"
)
_WORD = re.compile(r"[^\W_]+", flags=re.UNICODE)
_SENTENCE = re.compile(r"[^.!?]+(?:[.!?]+|$)")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "as",
        "at",
        "by",
        "for",
        "from",
        "in",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }
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


def _word_tokens(value: str, *, offset: int = 0) -> list[tuple[str, int, int]]:
    return [
        (match.group(0).casefold(), offset + match.start(), offset + match.end())
        for match in _WORD.finditer(value)
    ]


def _ordered_content_windows(
    *,
    phrase_tokens: list[tuple[str, int, int]],
    caption: str,
) -> list[tuple[int, int]]:
    content = [token for token, _, _ in phrase_tokens if token not in _STOPWORDS]
    if len(content) < 2:
        return []
    windows: list[tuple[int, int]] = []
    for sentence_match in _SENTENCE.finditer(caption):
        sentence_tokens = _word_tokens(
            sentence_match.group(0),
            offset=sentence_match.start(),
        )
        for start_index, (token, start, _) in enumerate(sentence_tokens):
            if token != content[0]:
                continue
            cursor = start_index + 1
            end = sentence_tokens[start_index][2]
            for expected in content[1:]:
                while cursor < len(sentence_tokens) and (
                    sentence_tokens[cursor][0] != expected
                ):
                    cursor += 1
                if cursor == len(sentence_tokens):
                    break
                end = sentence_tokens[cursor][2]
                cursor += 1
            else:
                if cursor - start_index <= max(18, len(content) * 3):
                    windows.append((start, end))
    return windows


def _unique_contiguous_content_span(
    *,
    phrase_tokens: list[tuple[str, int, int]],
    caption_tokens: list[tuple[str, int, int]],
) -> tuple[int, int] | None:
    candidates: list[tuple[int, int, int, int, int]] = []
    seen_sequences: set[tuple[str, ...]] = set()
    for length in range(len(phrase_tokens), 1, -1):
        for source_start in range(len(phrase_tokens) - length + 1):
            source_slice = phrase_tokens[source_start : source_start + length]
            while source_slice and source_slice[0][0] in _STOPWORDS:
                source_slice = source_slice[1:]
            while source_slice and source_slice[-1][0] in _STOPWORDS:
                source_slice = source_slice[:-1]
            sequence = tuple(token for token, _, _ in source_slice)
            content_count = sum(token not in _STOPWORDS for token in sequence)
            if content_count < 2 or sequence in seen_sequences:
                continue
            seen_sequences.add(sequence)
            occurrences: list[tuple[int, int]] = []
            for caption_start in range(len(caption_tokens) - len(sequence) + 1):
                caption_sequence = tuple(
                    token
                    for token, _, _ in caption_tokens[
                        caption_start : caption_start + len(sequence)
                    ]
                )
                if caption_sequence == sequence:
                    occurrences.append(
                        (
                            caption_tokens[caption_start][1],
                            caption_tokens[caption_start + len(sequence) - 1][2],
                        )
                    )
            if len(occurrences) == 1:
                start, end = occurrences[0]
                candidates.append(
                    (content_count, len(sequence), source_start, start, end)
                )
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (-item[0], -item[1], item[2], item[3], item[4])
    )
    _, _, _, start, end = candidates[0]
    return start, end


def find_legacy_caption_phrase_span(
    *,
    phrase: str,
    caption: str,
) -> tuple[int, int] | None:
    """Resolve pre-anchor V3 phrases using unique lexical evidence only."""
    phrase_tokens = _word_tokens(normalize_caption_phrase(phrase))
    caption_tokens = _word_tokens(caption)
    if not phrase_tokens or not caption_tokens:
        return None

    ordered_windows = _ordered_content_windows(
        phrase_tokens=phrase_tokens,
        caption=caption,
    )
    if len(ordered_windows) == 1:
        return ordered_windows[0]
    if len(ordered_windows) > 1:
        return None
    return _unique_contiguous_content_span(
        phrase_tokens=phrase_tokens,
        caption_tokens=caption_tokens,
    )
