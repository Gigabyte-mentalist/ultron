"""Tests for ultron.tools.browser_tools (docs/tz/07_tools.md §7C).

Nothing here opens a real browser or hits the real network: mock mode
covers the "action" tools, and ``get_weather``'s real-mode tests monkeypatch
``httpx.AsyncClient`` with a fake that returns canned Open-Meteo-shaped JSON.
"""

from __future__ import annotations

from urllib.parse import quote_plus

import httpx
import pytest

from ultron.config import RootConfig
from ultron.tools import browser_tools
from ultron.tools.base import default_registry


@pytest.fixture(autouse=True)
def _reset_config() -> None:
    browser_tools._config = None
    yield
    browser_tools._config = None
    weather_tool = default_registry.get("get_weather")
    if weather_tool is not None:
        weather_tool.enabled = True


class _FakeResponse:
    def __init__(self, json_data: dict) -> None:
        self._json = json_data

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._json


class _FakeAsyncClient:
    """Records every ``get()`` call; routes geocoding vs. forecast by URL."""

    instances: list[_FakeAsyncClient] = []

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.calls: list[tuple[str, dict]] = []
        _FakeAsyncClient.instances.append(self)

    async def __aenter__(self) -> _FakeAsyncClient:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def get(self, url: str, params: dict | None = None) -> _FakeResponse:
        self.calls.append((url, params or {}))
        if "geocoding" in url:
            return _FakeResponse({"results": [{"latitude": 39.65, "longitude": 66.96}]})
        return _FakeResponse(
            {
                "daily": {
                    "temperature_2m_max": [22.5, 25.0, 18.0, 20.0, 21.0, 23.0, 24.0],
                    "temperature_2m_min": [10.0, 12.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                    "weather_code": [0, 61, 3, 0, 0, 0, 0],
                    "precipitation_probability_max": [5, 60, 20, 5, 5, 5, 5],
                }
            }
        )


class _FakeAsyncClientNoGeoResults(_FakeAsyncClient):
    async def get(self, url: str, params: dict | None = None) -> _FakeResponse:
        self.calls.append((url, params or {}))
        if "geocoding" in url:
            return _FakeResponse({"results": []})
        return _FakeResponse({"daily": {}})


class _FailingAsyncClient:
    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    async def __aenter__(self) -> _FailingAsyncClient:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def get(self, *args: object, **kwargs: object) -> None:
        raise httpx.ConnectError("no network")


# -- open_url ----------------------------------------------------------------


async def test_open_url_rejects_non_http_scheme() -> None:
    result = await browser_tools.open_url(url="ftp://example.com")

    assert result.ok is False


async def test_open_url_mock_mode_does_not_call_launch_browser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ULTRON_TOOLS_MOCK", "1")
    calls = []
    monkeypatch.setattr(browser_tools, "_launch_browser", lambda url: calls.append(url))

    result = await browser_tools.open_url(url="https://example.com")

    assert result.ok is True
    assert calls == []


async def test_open_url_real_mode_calls_launch_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ULTRON_TOOLS_MOCK", raising=False)
    calls = []
    monkeypatch.setattr(browser_tools, "_launch_browser", lambda url: calls.append(url))

    result = await browser_tools.open_url(url="https://example.com")

    assert result.ok is True
    assert calls == ["https://example.com"]


async def test_open_url_resolves_known_site_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ULTRON_TOOLS_MOCK", "1")

    result = await browser_tools.open_url(url="kun.uz")

    assert result.ok is True
    assert result.data["url"] == "https://kun.uz"


async def test_open_url_unknown_bare_name_rejected() -> None:
    result = await browser_tools.open_url(url="not a url or known site")

    assert result.ok is False


# -- web_search / youtube_search ---------------------------------------------


async def test_web_search_builds_encoded_google_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ULTRON_TOOLS_MOCK", "1")

    result = await browser_tools.web_search(query="Python asyncio nima?")

    assert result.ok is True
    assert (
        result.data["url"]
        == f"https://www.google.com/search?q={quote_plus('Python asyncio nima?')}"
    )


async def test_web_search_uses_configured_default_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ULTRON_TOOLS_MOCK", "1")
    browser_tools.configure(
        RootConfig.model_validate({"tools": {"browser": {"search_engine": "bing"}}})
    )

    result = await browser_tools.web_search(query="test")

    assert result.data["url"].startswith("https://www.bing.com/search?q=")


async def test_web_search_explicit_engine_overrides_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ULTRON_TOOLS_MOCK", "1")
    browser_tools.configure(
        RootConfig.model_validate({"tools": {"browser": {"search_engine": "google"}}})
    )

    result = await browser_tools.web_search(query="test", engine="duckduckgo")

    assert result.data["url"].startswith("https://duckduckgo.com/?q=")


async def test_youtube_search_builds_encoded_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ULTRON_TOOLS_MOCK", "1")

    result = await browser_tools.youtube_search(query="lofi musiqa")

    assert result.ok is True
    expected = f"https://www.youtube.com/results?search_query={quote_plus('lofi musiqa')}"
    assert result.data["url"] == expected


# -- _launch_browser ----------------------------------------------------------


def test_launch_browser_falls_back_to_default_on_registration_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import webbrowser

    browser_tools.configure(
        RootConfig.model_validate({"tools": {"browser": {"default": "chrome"}}})
    )
    monkeypatch.setattr(
        webbrowser, "get", lambda name: (_ for _ in ()).throw(webbrowser.Error("no"))
    )
    calls = []
    monkeypatch.setattr(webbrowser, "open", lambda url: calls.append(url))

    browser_tools._launch_browser("https://example.com")

    assert calls == ["https://example.com"]


# -- WMO -> Uzbek text ---------------------------------------------------------


@pytest.mark.parametrize(
    "code,expected",
    [(0, "ochiq osmon"), (3, "bulutli"), (61, "yengil yomg'ir"), (95, "chaqmoq bilan bo'ron")],
)
def test_wmo_text_known_codes(code: int, expected: str) -> None:
    assert browser_tools._wmo_text(code) == expected


def test_wmo_text_unknown_code_has_fallback() -> None:
    assert browser_tools._wmo_text(12345) == "noma'lum ob-havo"


def test_wmo_text_none_code() -> None:
    assert browser_tools._wmo_text(None) == "noma'lum"


# -- get_weather ---------------------------------------------------------------


async def test_get_weather_disabled_by_config() -> None:
    browser_tools.configure(RootConfig.model_validate({"tools": {"internet_tools_enabled": False}}))

    result = await browser_tools.get_weather()

    assert result.ok is False


def test_get_weather_unregistered_from_registry_when_disabled() -> None:
    browser_tools.configure(RootConfig.model_validate({"tools": {"internet_tools_enabled": False}}))

    assert "get_weather" not in default_registry


def test_get_weather_stays_registered_when_enabled() -> None:
    browser_tools.configure(RootConfig.model_validate({"tools": {"internet_tools_enabled": True}}))

    assert "get_weather" in default_registry


async def test_get_weather_mock_mode_does_not_call_http(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ULTRON_TOOLS_MOCK", "1")
    monkeypatch.setattr(httpx, "AsyncClient", _FailingAsyncClient)  # would raise if ever called

    result = await browser_tools.get_weather(city="Tashkent")

    assert result.ok is True
    assert result.data["city"] == "Tashkent"


async def test_get_weather_default_city_uses_configured_coords_no_geocode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ULTRON_TOOLS_MOCK", raising=False)
    _FakeAsyncClient.instances = []
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    browser_tools.configure(
        RootConfig.model_validate(
            {"tools": {"weather": {"city": "Tashkent", "lat": 41.3, "lon": 69.2}}}
        )
    )

    result = await browser_tools.get_weather()

    assert result.ok is True
    assert result.data["city"] == "Tashkent"
    assert result.data["condition"] == "ochiq osmon"
    urls = [url for url, _params in _FakeAsyncClient.instances[0].calls]
    assert not any("geocoding" in u for u in urls)


async def test_get_weather_tomorrow_uses_second_daily_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ULTRON_TOOLS_MOCK", raising=False)
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    browser_tools.configure(
        RootConfig.model_validate(
            {"tools": {"weather": {"city": "Tashkent", "lat": 41.3, "lon": 69.2}}}
        )
    )

    result = await browser_tools.get_weather(days=2)

    assert result.ok is True
    assert result.data["days"] == 2
    assert result.data["condition"] == "yengil yomg'ir"
    assert result.data["temperature_max_c"] == 25.0
    assert "ertaga" in result.speak


async def test_get_weather_explicit_city_geocodes_first(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ULTRON_TOOLS_MOCK", raising=False)
    _FakeAsyncClient.instances = []
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)

    result = await browser_tools.get_weather(city="Samarqand")

    assert result.ok is True
    assert result.data["city"] == "Samarqand"
    urls = [url for url, _params in _FakeAsyncClient.instances[0].calls]
    assert any("geocoding" in u for u in urls)


async def test_geocode_city_prefers_uzbekistan_match(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: verified live that Open-Meteo ranks a Pakistani village
    above the real Samarkand for the Uzbek spelling "Samarqand"."""

    class _ForeignFirstClient(_FakeAsyncClient):
        async def get(self, url: str, params: dict | None = None) -> _FakeResponse:
            self.calls.append((url, params or {}))
            if "geocoding" in url:
                return _FakeResponse(
                    {
                        "results": [
                            {"latitude": 34.34, "longitude": 71.95, "country_code": "PK"},
                            {"latitude": 39.65, "longitude": 66.96, "country_code": "UZ"},
                        ]
                    }
                )
            return await super().get(url, params)

    monkeypatch.delenv("ULTRON_TOOLS_MOCK", raising=False)
    monkeypatch.setattr(httpx, "AsyncClient", _ForeignFirstClient)

    lat, lon = await browser_tools._geocode_city(httpx.AsyncClient(), "Samarqand")

    assert (lat, lon) == (39.65, 66.96)


async def test_get_weather_city_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ULTRON_TOOLS_MOCK", raising=False)
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClientNoGeoResults)

    result = await browser_tools.get_weather(city="Notarealcityxyz")

    assert result.ok is False


async def test_get_weather_network_failure_returns_friendly_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ULTRON_TOOLS_MOCK", raising=False)
    monkeypatch.setattr(httpx, "AsyncClient", _FailingAsyncClient)

    result = await browser_tools.get_weather(city="Tashkent")

    assert result.ok is False
    assert "internet" in result.error.lower()


async def test_get_weather_days_clamped_to_valid_range(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ULTRON_TOOLS_MOCK", "1")

    low = await browser_tools.get_weather(days=0)
    high = await browser_tools.get_weather(days=99)

    assert low.ok is True
    assert high.ok is True


@pytest.mark.parametrize(
    "days,expected", [(1, "bugun"), (2, "ertaga"), (3, "2 kundan keyin"), (7, "6 kundan keyin")]
)
def test_day_word(days: int, expected: str) -> None:
    assert browser_tools._day_word(days) == expected
