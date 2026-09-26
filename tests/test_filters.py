"""Tests for ultron.stt.filters (docs/tz/03_stt.md §4)."""

from __future__ import annotations

from ultron.stt.filters import (
    clean_transcript,
    collapse_repetitions,
    contains_blocklisted_phrase,
    is_hallucination,
    is_too_short,
)

_NO_SPEECH_THRESHOLD = 0.6
_LOG_PROB_THRESHOLD = -1.0


def test_is_too_short() -> None:
    assert is_too_short(0.1) is True
    assert is_too_short(0.29) is True
    assert is_too_short(0.3) is False
    assert is_too_short(2.0) is False


def test_is_hallucination_requires_both_conditions() -> None:
    kwargs = dict(no_speech_threshold=_NO_SPEECH_THRESHOLD, log_prob_threshold=_LOG_PROB_THRESHOLD)
    assert is_hallucination(no_speech_prob=0.8, avg_logprob=-1.5, **kwargs) is True
    assert is_hallucination(no_speech_prob=0.8, avg_logprob=-0.5, **kwargs) is False
    assert is_hallucination(no_speech_prob=0.3, avg_logprob=-1.5, **kwargs) is False
    assert is_hallucination(no_speech_prob=0.3, avg_logprob=-0.5, **kwargs) is False


def test_blocklist_catches_known_hallucination_phrases() -> None:
    assert contains_blocklisted_phrase("bu yerda subtitrlar bor") is True
    assert contains_blocklisted_phrase("тасвир учун субтитр qo'shildi") is True
    assert contains_blocklisted_phrase("продолжение следует") is True
    assert contains_blocklisted_phrase("rahmat tomosha qilganingiz uchun") is True
    assert contains_blocklisted_phrase("obuna bo'ling do'stlar") is True
    assert contains_blocklisted_phrase("подписывайтесь kanalga") is True
    assert contains_blocklisted_phrase("musiqa ♪ yangradi") is True
    assert contains_blocklisted_phrase("amara.org orqali tarjima qilindi") is True


def test_blocklist_leaves_ordinary_text_alone() -> None:
    assert contains_blocklisted_phrase("bugun ob-havo juda yaxshi") is False
    assert contains_blocklisted_phrase("Altron, telegramni och") is False


def test_collapse_repetitions_leaves_normal_text_alone() -> None:
    text = "bugun ob-havo juda yaxshi va quyoshli"
    assert collapse_repetitions(text) == text


def test_collapse_repetitions_keeps_exactly_three_repeats() -> None:
    phrase = "yangi kitob keldi"
    text = " ".join([phrase] * 3)
    assert collapse_repetitions(text) == text


def test_collapse_repetitions_collapses_more_than_three_repeats() -> None:
    phrase = "men judayam yaxshiman"
    text = " ".join([phrase] * 4)
    assert collapse_repetitions(text) == phrase

    text_five = " ".join([phrase] * 5)
    assert collapse_repetitions(text_five) == phrase


def test_clean_transcript_rejects_too_short_audio() -> None:
    result = clean_transcript(
        "salom",
        duration_s=0.1,
        no_speech_prob=0.1,
        avg_logprob=-0.1,
        no_speech_threshold=_NO_SPEECH_THRESHOLD,
        log_prob_threshold=_LOG_PROB_THRESHOLD,
    )
    assert result == ""


def test_clean_transcript_rejects_hallucination_gate() -> None:
    result = clean_transcript(
        "salom",
        duration_s=2.0,
        no_speech_prob=0.9,
        avg_logprob=-1.5,
        no_speech_threshold=_NO_SPEECH_THRESHOLD,
        log_prob_threshold=_LOG_PROB_THRESHOLD,
    )
    assert result == ""


def test_clean_transcript_rejects_blocklisted_text() -> None:
    result = clean_transcript(
        "obuna bo'ling",
        duration_s=2.0,
        no_speech_prob=0.1,
        avg_logprob=-0.1,
        no_speech_threshold=_NO_SPEECH_THRESHOLD,
        log_prob_threshold=_LOG_PROB_THRESHOLD,
    )
    assert result == ""


def test_clean_transcript_rejects_blank_text() -> None:
    result = clean_transcript(
        "   ",
        duration_s=2.0,
        no_speech_prob=0.1,
        avg_logprob=-0.1,
        no_speech_threshold=_NO_SPEECH_THRESHOLD,
        log_prob_threshold=_LOG_PROB_THRESHOLD,
    )
    assert result == ""


def test_clean_transcript_passes_and_collapses_good_text() -> None:
    phrase = "salom qanday ahvol"
    text = " ".join([phrase] * 5)
    result = clean_transcript(
        text,
        duration_s=2.0,
        no_speech_prob=0.1,
        avg_logprob=-0.1,
        no_speech_threshold=_NO_SPEECH_THRESHOLD,
        log_prob_threshold=_LOG_PROB_THRESHOLD,
    )
    assert result == phrase
