"""Browser and internet tools: open URLs, web/YouTube search, weather
(docs/tz/07_tools.md §7C).

The project's "hammasi lokal" (everything local) requirement is about AI
models specifically -- CLAUDE.md and docs/tz/00_umumiy_arxitektura.md both
scope it to cloud LLM APIs. Opening a browser and calling Open-Meteo's free,
keyless, non-AI weather API are the two explicitly-allowed exceptions
(§7C's own note: "Brauzer va ob-havo API'si LLM'ga ma'lumot bermaydi, faqat
foydalanuvchi so'roviga javob uchun ishlatiladi"). ``get_weather`` is
unregistered entirely when ``tools.internet_tools_enabled`` is false.
"""

from __future__ import annotations

import asyncio
import re
import webbrowser
from typing import Annotated
from urllib.parse import quote_plus

import httpx
from loguru import logger

from ultron.config import RootConfig
from ultron.tools.base import ToolResult, default_registry, is_mock_mode, tool

_SEARCH_ENGINES: dict[str, str] = {
    "google": "https://www.google.com/search?q={q}",
    "bing": "https://www.bing.com/search?q={q}",
    "duckduckgo": "https://duckduckgo.com/?q={q}",
}

# Popular sites dictionary (§7C): spoken Uzbek/plain name -> URL.
KNOWN_SITES: dict[str, str] = {
    "youtube": "https://www.youtube.com",
    "telegram web": "https://web.telegram.org",
    "gmail": "https://mail.google.com",
    "github": "https://github.com",
    "kun.uz": "https://kun.uz",
    "google tarjimon": "https://translate.google.com",
}

_BARE_DOMAIN_RE = re.compile(r"^[a-z0-9-]+(\.[a-z0-9-]+)+$")


def is_url_reference(name: str) -> bool:
    """True if ``name`` is a known site name or looks like a bare domain
    (e.g. "kun.uz"). ``llm/fastpath.py`` uses this to route "kun.uz ni och"
    to ``open_url`` instead of the generic ``open_app`` pattern it would
    otherwise match."""
    normalized = name.strip().lower()
    return normalized in KNOWN_SITES or bool(_BARE_DOMAIN_RE.match(normalized))


# WMO weather codes (Open-Meteo's ``weather_code``) -> short Uzbek text.
_WMO_CODES_UZ: dict[int, str] = {
    0: "ochiq osmon",
    1: "asosan ochiq",
    2: "qisman bulutli",
    3: "bulutli",
    45: "tuman",
    48: "muzli tuman",
    51: "yengil yomg'ir",
    53: "o'rtacha yomg'ir",
    55: "kuchli yomg'ir",
    56: "yengil muzli yomg'ir",
    57: "kuchli muzli yomg'ir",
    61: "yengil yomg'ir",
    63: "o'rtacha yomg'ir",
    65: "kuchli yomg'ir",
    66: "yengil muzli yomg'ir",
    67: "kuchli muzli yomg'ir",
    71: "yengil qor",
    73: "o'rtacha qor",
    75: "kuchli qor",
    77: "qor donachalari",
    80: "yengil jala",
    81: "o'rtacha jala",
    82: "kuchli jala",
    85: "yengil qor jalasi",
    86: "kuchli qor jalasi",
    95: "chaqmoq bilan bo'ron",
    96: "yengil do'l bilan bo'ron",
    99: "kuchli do'l bilan bo'ron",
}

_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_DAILY_FIELDS = "temperature_2m_max,temperature_2m_min,precipitation_probability_max,weather_code"
_HTTP_TIMEOUT_S = 5.0
_DEFAULT_CITY = "Tashkent"

_config: RootConfig | None = None


def configure(config: RootConfig) -> None:
    global _config
    _config = config
    weather_tool = default_registry.get("get_weather")
    if weather_tool is not None:
        weather_tool.enabled = config.tools.internet_tools_enabled


def _wmo_text(code: int | None) -> str:
    if code is None:
        return "noma'lum"
    return _WMO_CODES_UZ.get(code, "noma'lum ob-havo")


def _launch_browser(url: str) -> None:
    browser_name = _config.tools.browser.default if _config else "default"
    if browser_name != "default":
        try:
            webbrowser.get(browser_name).open(url)
            return
        except webbrowser.Error:
            pass  # configured browser not registered with `webbrowser` -- fall back
    webbrowser.open(url)


@tool(
    name="open_url",
    description="Open a website: a full http(s) URL, or a well-known site name"
    " (e.g. 'kun.uz', 'gmail', 'github').",
)
async def open_url(
    url: Annotated[str, "Full http(s) URL, or a well-known site name"],
) -> ToolResult:
    resolved_url = (
        url if url.startswith(("http://", "https://")) else KNOWN_SITES.get(url.strip().lower())
    )
    if resolved_url is None:
        return ToolResult(ok=False, error="Faqat http/https havolalar qo'llab-quvvatlanadi")

    if is_mock_mode():
        logger.bind(audit=True).info("[MOCK] open_url -> {}", resolved_url)
        return ToolResult(ok=True, data={"url": resolved_url}, speak="Havola ochildi.")

    await asyncio.to_thread(_launch_browser, resolved_url)
    return ToolResult(ok=True, data={"url": resolved_url}, speak="Havola ochildi.")


@tool(
    name="web_search",
    description="Search the web for a query and open the results in the browser.",
)
async def web_search(
    query: Annotated[str, "Search text"],
    engine: Annotated[str, "Search engine: google, bing, or duckduckgo"] | None = None,
) -> ToolResult:
    default_engine = _config.tools.browser.search_engine if _config else "google"
    chosen = (engine or default_engine).lower()
    template = _SEARCH_ENGINES.get(chosen, _SEARCH_ENGINES["google"])
    url = template.format(q=quote_plus(query))

    if is_mock_mode():
        logger.bind(audit=True).info("[MOCK] web_search -> {}", url)
        return ToolResult(ok=True, data={"url": url}, speak=f"'{query}' bo'yicha qidirildi.")

    await asyncio.to_thread(_launch_browser, url)
    return ToolResult(ok=True, data={"url": url}, speak=f"'{query}' bo'yicha qidirildi.")


@tool(
    name="youtube_search",
    description="Search YouTube for a query and open the results.",
)
async def youtube_search(
    query: Annotated[str, "Search text"],
    autoplay: Annotated[
        bool, "Reserved for future use; currently opens the results page either way"
    ] = False,
) -> ToolResult:
    del autoplay  # no local, keyless way to jump straight to the top result yet
    url = f"https://www.youtube.com/results?search_query={quote_plus(query)}"

    if is_mock_mode():
        logger.bind(audit=True).info("[MOCK] youtube_search -> {}", url)
        return ToolResult(ok=True, data={"url": url}, speak=f"YouTube'da '{query}' qidirildi.")

    await asyncio.to_thread(_launch_browser, url)
    return ToolResult(ok=True, data={"url": url}, speak=f"YouTube'da '{query}' qidirildi.")


async def _geocode_city(client: httpx.AsyncClient, city: str) -> tuple[float, float] | None:
    """Resolve a city name to coordinates, preferring an Uzbekistan match.

    Open-Meteo's geocoding search can rank an unrelated foreign place above
    the intended Uzbek city for an Uzbek-spelled name -- verified live:
    "Samarqand" (the Uzbek Latin spelling users actually speak) returns a
    village in Pakistan as its *first* result, while "Samarkand" (English)
    correctly returns Samarkand, Uzbekistan. Since every user of this
    project is in Uzbekistan, requesting a few candidates and preferring
    ``country_code == "UZ"`` fixes this without a per-city translation table.
    """
    resp = await client.get(_GEOCODE_URL, params={"name": city, "count": 5, "language": "en"})
    resp.raise_for_status()
    results = resp.json().get("results")
    if not results:
        return None
    best = next((r for r in results if r.get("country_code") == "UZ"), results[0])
    return best["latitude"], best["longitude"]


def _day_word(days: int) -> str:
    if days == 1:
        return "bugun"
    if days == 2:
        return "ertaga"
    return f"{days - 1} kundan keyin"


@tool(
    name="get_weather",
    description="Get the weather for a city on a given day (1=today, 2=tomorrow, up to 7);"
    " defaults to the configured home city.",
)
async def get_weather(
    city: Annotated[str, "City name; defaults to the configured home city"] | None = None,
    days: Annotated[int, "Which day to report: 1=today (default), 2=tomorrow, ..., up to 7"] = 1,
) -> ToolResult:
    if _config is not None and not _config.tools.internet_tools_enabled:
        return ToolResult(ok=False, error="Internet orqali ob-havo xizmati o'chirilgan")

    days = max(1, min(days, 7))
    weather_city = city or (_config.tools.weather.city if _config else _DEFAULT_CITY)

    if is_mock_mode():
        logger.bind(audit=True).info("[MOCK] get_weather city={} days={}", weather_city, days)
        return ToolResult(
            ok=True,
            data={
                "city": weather_city,
                "days": days,
                "temperature_max_c": 20.0,
                "condition": "ochiq osmon",
            },
            speak=f"{weather_city}da {_day_word(days)} ochiq osmon, harorat 20 daraja.",
        )

    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_S) as client:
            if city is None and _config is not None:
                lat, lon = _config.tools.weather.lat, _config.tools.weather.lon
            else:
                coords = await _geocode_city(client, weather_city)
                if coords is None:
                    return ToolResult(ok=False, error=f"'{weather_city}' shahri topilmadi")
                lat, lon = coords

            resp = await client.get(
                _FORECAST_URL,
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "daily": _DAILY_FIELDS,
                    "forecast_days": days,
                    "timezone": "auto",
                },
            )
            resp.raise_for_status()
            payload = resp.json()
    except httpx.HTTPError as exc:
        logger.warning("Ob-havo API xatosi: {}", exc)
        return ToolResult(
            ok=False, error="Internetga ulanib bo'lmadi", speak="Internetga ulanib bo'lmadi."
        )

    daily = payload.get("daily", {})
    codes = daily.get("weather_code") or []
    idx = days - 1
    if idx >= len(codes):
        return ToolResult(ok=False, error="Bu kun uchun ma'lumot yo'q")

    temp_max = (daily.get("temperature_2m_max") or [None] * len(codes))[idx]
    temp_min = (daily.get("temperature_2m_min") or [None] * len(codes))[idx]
    precip = (daily.get("precipitation_probability_max") or [None] * len(codes))[idx]
    condition = _wmo_text(codes[idx])

    speak = (
        f"{weather_city}da {_day_word(days)} {condition}, "
        f"harorat {temp_min} dan {temp_max} darajagacha."
    )
    if precip is not None:
        speak += f" Yomg'ir ehtimoli {precip} foiz."
    return ToolResult(
        ok=True,
        data={
            "city": weather_city,
            "days": days,
            "temperature_min_c": temp_min,
            "temperature_max_c": temp_max,
            "condition": condition,
            "precipitation_probability": precip,
        },
        speak=speak,
    )
