"""Fuzzy wake-word matching (docs/tz/04_wake_word.md §3.1).

Only the first ``max_prefix_words`` words of an utterance are ever
considered, and only as progressive concatenations starting from word 0
("al tron" -> "al", "altron"). This is what makes mid-sentence mentions like
"men altron haqida o'qidim" harmless without any denylist: "men" and
"menaltron" both score far below threshold against every configured phrase.

The denylist below exists for the opposite failure mode: some short words
that are *not* prefix concatenations of anything score high enough against
"altron"/"ultron" on raw edit distance alone to clear a typical threshold
(e.g. "patron" vs "altron" is 83, above the default 82) even though they are
not built from a multi-word split. It is intentionally a plain module-level
list (like ``stt/filters.py``'s hallucination blocklist) rather than a config
key, since 00-TZ's config schema is fixed and this list is expected to grow
from benchmark findings rather than from user configuration.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rapidfuzz import fuzz

_WORD_STRIP_CHARS = ' \t\n\r.,!?;:()[]{}"“”«»-–—…'
_LEADING_STRIP_CHARS = _WORD_STRIP_CHARS + "'"

_LEADING_VOWEL_RUN_RE = re.compile(r"^[aou]+")
_DOUBLE_LETTER_RE = re.compile(r"(.)\1+")

# Words phonetically close to "altron"/"ultron" that must be rejected even if
# they clear the fuzzy threshold on raw edit distance (docs/tz/04_wake_word.md
# §3.1). Rejection only applies when the candidate *exactly* equals one of
# these (after stripping punctuation and lowercasing) -- a partial/fuzzy hit
# against this list does not, by itself, block a genuine wake word.
_NEGATIVE_WORDS = frozenset(
    {
        "elektron",
        "elektronika",
        "patron",
        "patronlar",
        "altin",
        "electron",
    }
)


@dataclass
class MatchResult:
    matched: bool
    score: float  # 0..100 (rapidfuzz scale, comparable directly to fuzzy_threshold)
    variant: str  # the configured phrase that scored best
    trailing_text: str | None  # text after the wake word, or None if nothing follows


def _phonetic_key(word: str) -> str:
    """Collapse a handful of common uz/ru mis-hearings onto one canonical form.

    Normalizes the leading vowel run (Oltron/Ultron/Altron -> same start),
    collapses doubled letters, and merges the "lt"/"ltr" cluster ambiguity
    ("al-tron" heard with or without an extra "r").
    """
    key = word.lower()
    key = _LEADING_VOWEL_RUN_RE.sub("a", key)
    key = key.replace("ltr", "lt")
    key = _DOUBLE_LETTER_RE.sub(r"\1", key)
    return key


class WakeMatcher:
    """Fuzzy-matches the start of a normalized transcript against ``phrases``."""

    def __init__(self, *, phrases: list[str], fuzzy_threshold: int, max_prefix_words: int) -> None:
        self._phrases = [p.lower() for p in phrases]
        self._phonetic_phrases = [_phonetic_key(p) for p in self._phrases]
        self._fuzzy_threshold = fuzzy_threshold
        self._max_prefix_words = max_prefix_words

    def match(self, text: str) -> MatchResult:
        words = text.strip().split()
        if not words:
            return MatchResult(matched=False, score=0.0, variant="", trailing_text=None)

        prefix = words[: self._max_prefix_words]
        cleaned_prefix = [w.strip(_WORD_STRIP_CHARS) for w in prefix]

        best_score = 0.0
        best_variant = ""
        best_span = 0
        for span in range(1, len(cleaned_prefix) + 1):
            candidate = "".join(cleaned_prefix[:span]).lower()
            if not candidate:
                continue
            score, variant = self._score_candidate(candidate)
            if score > best_score:
                best_score = score
                best_variant = variant
                best_span = span

        matched = best_score >= self._fuzzy_threshold
        if matched and self._is_negative(cleaned_prefix[:best_span]):
            matched = False

        trailing_text = None
        if matched:
            remainder = " ".join(words[best_span:]).strip()
            remainder = remainder.lstrip(_LEADING_STRIP_CHARS).strip()
            trailing_text = remainder or None

        return MatchResult(
            matched=matched, score=best_score, variant=best_variant, trailing_text=trailing_text
        )

    def _score_candidate(self, candidate: str) -> tuple[float, str]:
        phon_candidate = _phonetic_key(candidate)
        best_score = 0.0
        best_phrase = ""
        for phrase, phon_phrase in zip(self._phrases, self._phonetic_phrases, strict=True):
            score = max(fuzz.ratio(candidate, phrase), fuzz.ratio(phon_candidate, phon_phrase))
            if score > best_score:
                best_score = score
                best_phrase = phrase
        return best_score, best_phrase

    @staticmethod
    def _is_negative(words: list[str]) -> bool:
        joined = "".join(words).lower()
        if joined in _NEGATIVE_WORDS:
            return True
        return any(w.lower() in _NEGATIVE_WORDS for w in words)
