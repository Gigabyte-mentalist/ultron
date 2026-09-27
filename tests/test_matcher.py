"""Tests for ultron.wake.matcher.WakeMatcher (docs/tz/04_wake_word.md §4).

Uses the real default ``wake:`` config values (phrases, fuzzy_threshold,
max_prefix_words) so these tests exercise the matcher exactly as configured,
not an artificially loosened stand-in.
"""

from __future__ import annotations

import pytest

from ultron.config import WakeConfig
from ultron.wake.matcher import WakeMatcher

_DEFAULTS = WakeConfig()


def _matcher() -> WakeMatcher:
    return WakeMatcher(
        phrases=_DEFAULTS.phrases,
        fuzzy_threshold=_DEFAULTS.fuzzy_threshold,
        max_prefix_words=_DEFAULTS.max_prefix_words,
    )


_POSITIVE_TEXTS = [
    "altron",
    "Altron",
    "Altron,",
    "altron.",
    "ALTRON",
    "al tron",
    "al-tron",
    "Al tron,",
    "oltron",
    "ultron",
    "ultiron",
    "altiron",
    "олтрон",
    "алтрон",
    "ультрон",
    "Ультрон",
    "altron brauzerni och",
    "Altron, brauzerni och",
    "ultron ovozni balandlat",
    "oltron telegramni och",
    "altran",
    "ulttron",
    "al'tron",
    "Altron eslatma qo'y",
    "altron soat nechchi",
    "ultron ob-havo qanday",
    "Oltron, xonadagi chiroqni yoq",
    "altiron kompyuterni ochib ber",
    "ultiron musiqa qo'y",
    "Altron  brauzer",
    "ULTRON, telefonga qara",
    "altron?",
    "Oltron!",
]

_NEGATIVE_TEXTS = [
    "elektron",
    "patron",
    "altin",
    "salom",
    "electron",
    "men altron haqida o'qidim",
    "bugun ob-havo juda yaxshi",
    "kitob stol ustida",
    "telefonni olib kel",
    "assalomu alaykum",
    "rahmat sizga",
    "nima gap",
    "qalaysiz",
    "yaxshimisiz",
    "bugun dushanba",
    "ertaga bayram",
    "ovqat tayyor",
    "men patron sotib oldim",
    "u elektronika sotadi",
    "bu altin uzuk",
    "kompyuter sekin ishlayapti",
    "internet uzildi",
    "telegram ochilmadi",
    "brauzerni yop",
    "musiqa qo'y",
    "chiroqni o'chir",
    "bolalar maktabga bordi",
    "mashina yo'lda turibdi",
    "osmon bulutli",
    "daryo bo'yida sayr qildik",
    "kun juda issiq",
    "sovuq suv kerak",
    "u menga qara dedi",
    "bugun men elektron pochta yubordim",
    "patronlar tugadi ovda",
    "altin baliq ertagi",
    "",
]


@pytest.mark.parametrize("text", _POSITIVE_TEXTS)
def test_positive_cases_match(text: str) -> None:
    result = _matcher().match(text.lower())
    assert result.matched is True
    assert result.score >= _DEFAULTS.fuzzy_threshold


@pytest.mark.parametrize("text", _NEGATIVE_TEXTS)
def test_negative_cases_do_not_match(text: str) -> None:
    result = _matcher().match(text.lower())
    assert result.matched is False


def test_trailing_text_extracted_after_single_word_wake() -> None:
    result = _matcher().match("altron brauzerni och")
    assert result.matched is True
    assert result.trailing_text == "brauzerni och"


def test_trailing_text_extracted_after_split_wake_word() -> None:
    result = _matcher().match("al tron eslatma qo'y")
    assert result.matched is True
    assert result.trailing_text == "eslatma qo'y"


def test_trailing_text_strips_leading_punctuation() -> None:
    result = _matcher().match("altron, ovozni balandlat")
    assert result.matched is True
    assert result.trailing_text == "ovozni balandlat"


def test_no_trailing_text_when_wake_word_alone() -> None:
    result = _matcher().match("altron")
    assert result.matched is True
    assert result.trailing_text is None


def test_midsentence_wake_word_does_not_trigger() -> None:
    result = _matcher().match("men altron haqida o'qidim")
    assert result.matched is False


def test_denylist_rejects_patron_even_though_it_clears_raw_threshold() -> None:
    # "patron" alone scores ~83 against "altron" on raw edit distance, above
    # the default 82 threshold -- the denylist is what keeps this from
    # false-accepting.
    result = _matcher().match("patron")
    assert result.matched is False


def test_denylist_rejects_altin() -> None:
    result = _matcher().match("altin")
    assert result.matched is False


def test_denylist_does_not_block_a_genuine_wake_word() -> None:
    result = _matcher().match("altron")
    assert result.matched is True


def test_empty_text_does_not_match() -> None:
    result = _matcher().match("")
    assert result.matched is False
    assert result.score == 0.0


def test_fuzzy_threshold_is_configurable() -> None:
    strict = WakeMatcher(
        phrases=_DEFAULTS.phrases, fuzzy_threshold=99, max_prefix_words=_DEFAULTS.max_prefix_words
    )
    assert strict.match("altran").matched is False  # 83 < 99

    loose = WakeMatcher(
        phrases=_DEFAULTS.phrases, fuzzy_threshold=10, max_prefix_words=_DEFAULTS.max_prefix_words
    )
    assert loose.match("salom").matched is True  # low bar, clears easily
