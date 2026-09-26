"""Hallucination filters for Whisper output (docs/tz/03_stt.md §3.2).

Whisper "makes up" text on silence or noise. ``clean_transcript`` runs four
checks in order and returns an empty string as soon as one of them decides
the hypothesis is not real speech; otherwise it returns the (still
un-normalized) text with runaway repeated phrases collapsed.
"""

from __future__ import annotations

import re

_MIN_AUDIO_S = 0.3

_BLOCKLIST_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"subtit",
        r"субтитр",
        r"продолжение следует",
        r"rahmat[,]?\s*(ki\s+)?tomosha\s+qil",
        r"tomosha\s+qilganingiz\s+uchun\s+rahmat",
        r"obuna\s+bo['ʻ’]?ling",
        r"подписывайтесь",
        r"amara\.org",
        r"♪",  # ♪
    )
]

_MIN_PHRASE_WORDS = 3
_MAX_PHRASE_WORDS = 8
_MAX_REPEATS_ALLOWED = 3


def is_too_short(duration_s: float) -> bool:
    return duration_s < _MIN_AUDIO_S


def is_hallucination(
    *,
    no_speech_prob: float,
    avg_logprob: float,
    no_speech_threshold: float,
    log_prob_threshold: float,
) -> bool:
    return no_speech_prob > no_speech_threshold and avg_logprob < log_prob_threshold


def contains_blocklisted_phrase(text: str) -> bool:
    return any(pattern.search(text) for pattern in _BLOCKLIST_PATTERNS)


def collapse_repetitions(text: str) -> str:
    """Collapse a 3+ word phrase repeated more than 3 times in a row to one occurrence."""
    words = text.split()
    n = len(words)
    out: list[str] = []
    i = 0
    while i < n:
        collapsed = False
        max_len = min(_MAX_PHRASE_WORDS, (n - i) // (_MAX_REPEATS_ALLOWED + 1))
        for phrase_len in range(max_len, _MIN_PHRASE_WORDS - 1, -1):
            phrase = words[i : i + phrase_len]
            repeats = 1
            j = i + phrase_len
            while words[j : j + phrase_len] == phrase:
                repeats += 1
                j += phrase_len
            if repeats > _MAX_REPEATS_ALLOWED:
                out.extend(phrase)
                i = j
                collapsed = True
                break
        if not collapsed:
            out.append(words[i])
            i += 1
    return " ".join(out)


def clean_transcript(
    raw_text: str,
    *,
    duration_s: float,
    no_speech_prob: float,
    avg_logprob: float,
    no_speech_threshold: float,
    log_prob_threshold: float,
) -> str:
    """Apply all hallucination filters; returns "" if the hypothesis should be dropped."""
    if is_too_short(duration_s):
        return ""
    if is_hallucination(
        no_speech_prob=no_speech_prob,
        avg_logprob=avg_logprob,
        no_speech_threshold=no_speech_threshold,
        log_prob_threshold=log_prob_threshold,
    ):
        return ""
    text = raw_text.strip()
    if not text:
        return ""
    if contains_blocklisted_phrase(text):
        return ""
    return collapse_repetitions(text)
