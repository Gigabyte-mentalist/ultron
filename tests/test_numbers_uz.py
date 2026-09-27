"""Tests for ultron.text.numbers_uz (docs/tz/05_tts.md §4)."""

from __future__ import annotations

import pytest

from ultron.text.numbers_uz import (
    convert_numbers_in_text,
    date_to_words,
    decimal_to_words,
    number_to_words,
    ordinal_to_words,
    percent_to_words,
    temperature_to_words,
    time_to_words,
    words_to_number,
)

_NUMBER_CASES = [
    (0, "nol"),
    (1, "bir"),
    (2, "ikki"),
    (5, "besh"),
    (9, "to'qqiz"),
    (10, "o'n"),
    (11, "o'n bir"),
    (15, "o'n besh"),
    (19, "o'n to'qqiz"),
    (20, "yigirma"),
    (21, "yigirma bir"),
    (26, "yigirma olti"),
    (29, "yigirma to'qqiz"),
    (30, "o'ttiz"),
    (40, "qirq"),
    (45, "qirq besh"),
    (50, "ellik"),
    (60, "oltmish"),
    (70, "yetmish"),
    (80, "sakson"),
    (90, "to'qson"),
    (99, "to'qson to'qqiz"),
    (100, "yuz"),
    (101, "yuz bir"),
    (110, "yuz o'n"),
    (111, "yuz o'n bir"),
    (125, "yuz yigirma besh"),
    (150, "yuz ellik"),
    (200, "ikki yuz"),
    (500, "besh yuz"),
    (999, "to'qqiz yuz to'qson to'qqiz"),
    (1000, "ming"),
    (1001, "ming bir"),
    (1026, "ming yigirma olti"),
    (1999, "ming to'qqiz yuz to'qson to'qqiz"),
    (2000, "ikki ming"),
    (2026, "ikki ming yigirma olti"),
    (10000, "o'n ming"),
    (12345, "o'n ikki ming uch yuz qirq besh"),
    (100000, "yuz ming"),
    (500000, "besh yuz ming"),
    (999999, "to'qqiz yuz to'qson to'qqiz ming to'qqiz yuz to'qson to'qqiz"),
    (1000000, "bir million"),
    (1000001, "bir million bir"),
    (2000000, "ikki million"),
    (1000000000, "bir milliard"),
    (
        999999999999,
        "to'qqiz yuz to'qson to'qqiz milliard to'qqiz yuz to'qson to'qqiz million "
        "to'qqiz yuz to'qson to'qqiz ming to'qqiz yuz to'qson to'qqiz",
    ),
    (-5, "minus besh"),
    (-100, "minus yuz"),
    (-1000, "minus ming"),
]


@pytest.mark.parametrize("n,expected", _NUMBER_CASES)
def test_number_to_words(n: int, expected: str) -> None:
    assert number_to_words(n) == expected


def test_at_least_fifty_number_cases_covered() -> None:
    assert len(_NUMBER_CASES) >= 50


def test_number_to_words_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        number_to_words(1_000_000_000_000)


_ORDINAL_CASES = [
    (1, "birinchi"),
    (2, "ikkinchi"),
    (3, "uchinchi"),
    (4, "to'rtinchi"),
    (5, "beshinchi"),
    (6, "oltinchi"),
    (7, "yettinchi"),
    (8, "sakkizinchi"),
    (9, "to'qqizinchi"),
    (10, "o'ninchi"),
    (11, "o'n birinchi"),
    (20, "yigirmanchi"),
    (21, "yigirma birinchi"),
    (27, "yigirma yettinchi"),
    (100, "yuzinchi"),
    (1000, "minginchi"),
    (2026, "ikki ming yigirma oltinchi"),
]


@pytest.mark.parametrize("n,expected", _ORDINAL_CASES)
def test_ordinal_to_words(n: int, expected: str) -> None:
    assert ordinal_to_words(n) == expected


def test_decimal_reads_as_fraction() -> None:
    assert decimal_to_words(3, "5") == "uch butun o'ndan besh"
    assert decimal_to_words(3, "05") == "uch butun yuzdan besh"
    assert decimal_to_words(3, "25") == "uch butun yuzdan yigirma besh"
    assert decimal_to_words(3, "50") == "uch butun o'ndan besh"  # trailing zero reduces


def test_decimal_with_all_zero_fraction_has_no_fraction_words() -> None:
    assert decimal_to_words(3, "00") == "uch butun"


def test_percent_to_words() -> None:
    assert percent_to_words(50) == "ellik foiz"
    assert percent_to_words(0) == "nol foiz"
    assert percent_to_words(100) == "yuz foiz"


def test_time_to_words() -> None:
    assert time_to_words(9, 30) == "to'qqiz o'ttiz"
    assert time_to_words(14, 5) == "o'n to'rt nol besh"
    assert time_to_words(9, 0) == "to'qqiz nol"
    assert time_to_words(23, 59) == "yigirma uch ellik to'qqiz"


def test_date_to_words() -> None:
    assert (
        date_to_words(27, 9, 2026) == "yigirma yettinchi sentyabr, ikki ming yigirma oltinchi yil"
    )
    assert date_to_words(1, 1, 2000) == "birinchi yanvar, ikki minginchi yil"


def test_date_to_words_rejects_bad_month() -> None:
    with pytest.raises(ValueError):
        date_to_words(1, 13, 2026)


def test_temperature_to_words() -> None:
    assert temperature_to_words(-3) == "minus uch daraja"
    assert temperature_to_words(25) == "yigirma besh daraja"
    assert temperature_to_words(0) == "nol daraja"


# -- text-level pipeline ------------------------------------------------------


def test_convert_numbers_in_text_handles_date_before_decimal() -> None:
    text = "Bugun 27.09.2026, soat 09:30."
    result = convert_numbers_in_text(text)
    assert result == (
        "Bugun yigirma yettinchi sentyabr, ikki ming yigirma oltinchi yil, soat to'qqiz o'ttiz."
    )


def test_convert_numbers_in_text_percent_and_temperature() -> None:
    text = "Harorat -3°C, 50% ehtimol bilan yomg'ir."
    result = convert_numbers_in_text(text)
    assert result == "Harorat minus uch daraja, ellik foiz ehtimol bilan yomg'ir."


def test_convert_numbers_in_text_decimal() -> None:
    assert convert_numbers_in_text("3.5 kilometr") == "uch butun o'ndan besh kilometr"


def test_convert_numbers_in_text_ordinal_glued_to_next_word() -> None:
    assert convert_numbers_in_text("5-sinf o'quvchisi") == "beshinchi sinf o'quvchisi"
    assert convert_numbers_in_text("2025-yil boshida") == "ikki ming yigirma beshinchi yil boshida"


def test_convert_numbers_in_text_plain_integer() -> None:
    assert convert_numbers_in_text("Men 3 ta olma yedim") == "Men uch ta olma yedim"


def test_convert_numbers_in_text_negative_integer_without_unit() -> None:
    assert convert_numbers_in_text("Harorat -5 edi") == "Harorat minus besh edi"


# -- words_to_number (docs/tz/06_llm.md §3.5) ---------------------------------

_WORDS_TO_NUMBER_CASES = [
    ("nol", 0),
    ("bir", 1),
    ("besh", 5),
    ("o'n", 10),
    ("o'n besh", 15),
    ("yigirma", 20),
    ("yigirma bir", 21),
    ("ellik", 50),
    ("o'ttiz besh", 35),
    ("to'qson to'qqiz", 99),
    ("yuz", 100),
    ("bir yuz", 100),
    ("ikki yuz", 200),
    ("yuz o'ttiz", 130),
    ("ming", 1000),
    ("ikki ming yigirma olti", 2026),
    ("bir million", 1_000_000),
]


@pytest.mark.parametrize("text,expected", _WORDS_TO_NUMBER_CASES)
def test_words_to_number(text: str, expected: int) -> None:
    assert words_to_number(text) == expected


def test_words_to_number_stops_at_trailing_unrelated_words() -> None:
    assert words_to_number("ellik foiz qil") == 50
    assert words_to_number("o'ttiz besh foiz") == 35


def test_words_to_number_returns_none_for_non_numeric_text() -> None:
    assert words_to_number("salom qandaysiz") is None
    assert words_to_number("") is None


def test_words_to_number_is_inverse_of_number_to_words() -> None:
    for n in (0, 1, 9, 10, 25, 50, 99, 100, 130, 999, 1000, 2026):
        assert words_to_number(number_to_words(n)) == n
