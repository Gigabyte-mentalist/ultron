"""Tests for ultron.text.normalize (docs/tz/03_stt.md §4)."""

from __future__ import annotations

from ultron.text.normalize import normalize_transcript


def test_empty_input_returns_empty() -> None:
    assert normalize_transcript("") == ""


def test_mostly_cyrillic_text_is_converted_to_latin() -> None:
    assert normalize_transcript("салом дунё") == "salom dunyo"


def test_mostly_latin_text_with_a_stray_cyrillic_word_is_left_alone() -> None:
    text = "salom дунё bugun juda yaxshi kun"
    result = normalize_transcript(text)
    assert result == text  # cyrillic ratio well under 30%, no conversion triggered


def test_apostrophe_variants_are_canonicalized() -> None:
    assert normalize_transcript("to‘g‘ri so’z") == "to'g'ri so'z"


def test_whitespace_and_trailing_punctuation_are_trimmed() -> None:
    assert normalize_transcript("  Salom dunyo!!  ") == "Salom dunyo"
    assert normalize_transcript("bir   ikki    uch") == "bir ikki uch"


def test_wake_word_variants_are_canonicalized() -> None:
    assert normalize_transcript("ultron, brauzerni och") == "Altron, brauzerni och"
    assert normalize_transcript("Oltron radio qo'y") == "Altron radio qo'y"
    assert normalize_transcript("Al tron kelasan") == "Altron kelasan"


def test_normalize_is_idempotent() -> None:
    once = normalize_transcript("салом, Ultron!")
    twice = normalize_transcript(once)
    assert once == twice
