"""
tests/test_search.py — Unit tests for the search provider layer.

Mocks all network calls so tests run offline.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.models.schemas import SearchCandidate


# ── SearchProvider interface ──────────────────────────────────────────────────

class TestSearchProviderInterface:
    def test_cannot_instantiate_abstract(self):
        from app.search.web_search import SearchProvider
        with pytest.raises(TypeError):
            SearchProvider()  # type: ignore


# ── SerpApiLensProvider ───────────────────────────────────────────────────────

class TestSerpApiLensProvider:
    def _provider(self):
        from app.search.web_search import SerpApiLensProvider
        return SerpApiLensProvider(api_key="test-key")

    def test_no_key_raises(self, monkeypatch):
        monkeypatch.setattr("app.search.web_search.SERPAPI_KEY", "")
        from app.search.web_search import SerpApiLensProvider
        with pytest.raises(EnvironmentError, match="SERPAPI_KEY"):
            SerpApiLensProvider(api_key="")

    def _host_ok(self, monkeypatch):
        monkeypatch.setattr(
            "app.search.web_search._host_image_temporarily",
            lambda _p: "https://tmpfiles.org/dl/1/face.png",
        )

    def test_returns_candidates(self, tmp_path, monkeypatch):
        from PIL import Image
        img_path = tmp_path / "face.png"
        Image.new("RGB", (64, 64)).save(img_path)
        self._host_ok(monkeypatch)

        fake_results = {
            "visual_matches": [
                {
                    "link": "https://instagram.com/p/abc123",
                    "title": "Test post",
                    "snippet": "A face post",
                    "thumbnail": "https://example.com/thumb.jpg",
                },
                {
                    "link": "https://twitter.com/user/status/1234",
                    "title": "Tweet",
                    "snippet": "",
                    "thumbnail": "",
                },
            ]
        }

        mock_gs = MagicMock()
        mock_gs.return_value.get_dict.return_value = fake_results
        monkeypatch.setattr("app.search.web_search.GoogleSearch", mock_gs, raising=False)
        # GoogleSearch is imported inside the method; patch at serpapi too
        import sys
        fake_mod = MagicMock()
        fake_mod.GoogleSearch = mock_gs
        monkeypatch.setitem(sys.modules, "serpapi", fake_mod)

        provider = self._provider()
        candidates = provider.search(img_path)

        assert len(candidates) == 2
        assert candidates[0].platform == "Instagram"
        assert candidates[1].platform == "Twitter / X"

    def test_api_error_raises(self, tmp_path, monkeypatch):
        from PIL import Image
        img_path = tmp_path / "face.png"
        Image.new("RGB", (64, 64)).save(img_path)
        self._host_ok(monkeypatch)

        mock_gs = MagicMock()
        mock_gs.return_value.get_dict.return_value = {"error": "rate limit exceeded"}
        import sys
        fake_mod = MagicMock()
        fake_mod.GoogleSearch = mock_gs
        monkeypatch.setitem(sys.modules, "serpapi", fake_mod)

        provider = self._provider()
        with pytest.raises(RuntimeError, match="SerpAPI error"):
            provider.search(img_path)

    def test_zero_results(self, tmp_path, monkeypatch):
        from PIL import Image
        img_path = tmp_path / "face.png"
        Image.new("RGB", (64, 64)).save(img_path)
        self._host_ok(monkeypatch)

        mock_gs = MagicMock()
        mock_gs.return_value.get_dict.return_value = {"visual_matches": []}
        import sys
        fake_mod = MagicMock()
        fake_mod.GoogleSearch = mock_gs
        monkeypatch.setitem(sys.modules, "serpapi", fake_mod)

        provider = self._provider()
        results = provider.search(img_path)
        assert results == []


# ── BingVisualSearchProvider ──────────────────────────────────────────────────

class TestBingVisualSearchProvider:
    def test_no_key_raises(self, monkeypatch):
        monkeypatch.setattr("app.search.web_search.BING_SEARCH_API_KEY", "")
        from app.search.web_search import BingVisualSearchProvider
        with pytest.raises(EnvironmentError, match="BING_SEARCH_API_KEY"):
            BingVisualSearchProvider(api_key="")

    def test_returns_candidates(self, tmp_path):
        from PIL import Image
        import requests

        img_path = tmp_path / "face.png"
        Image.new("RGB", (64, 64)).save(img_path)

        fake_response = MagicMock()
        fake_response.ok = True
        fake_response.json.return_value = {
            "tags": [
                {
                    "actions": [
                        {
                            "actionType": "PagesIncluding",
                            "data": {
                                "value": [
                                    {
                                        "contentUrl": "https://example.com/page",
                                        "name": "Example page",
                                        "thumbnailUrl": "https://example.com/thumb.jpg",
                                        "description": "A face",
                                    }
                                ]
                            },
                        }
                    ]
                }
            ]
        }

        with patch("app.search.web_search.requests.post", return_value=fake_response):
            from app.search.web_search import BingVisualSearchProvider
            provider = BingVisualSearchProvider(api_key="test-key")
            candidates = provider.search(img_path)

        assert len(candidates) == 1
        assert candidates[0].url == "https://example.com/page"

    def test_http_error_raises(self, tmp_path):
        from PIL import Image

        img_path = tmp_path / "face.png"
        Image.new("RGB", (64, 64)).save(img_path)

        fake_response = MagicMock()
        fake_response.ok = False
        fake_response.status_code = 401
        fake_response.text = "Unauthorized"

        with patch("app.search.web_search.requests.post", return_value=fake_response):
            from app.search.web_search import BingVisualSearchProvider
            provider = BingVisualSearchProvider(api_key="test-key")
            with pytest.raises(RuntimeError, match="Bing Visual Search HTTP 401"):
                provider.search(img_path)


# ── build_search_provider factory ─────────────────────────────────────────────

class TestBuildSearchProvider:
    def test_serpapi_preferred(self, monkeypatch):
        monkeypatch.setattr("app.search.web_search.SERPAPI_KEY", "key123")
        monkeypatch.setattr("app.search.web_search.BING_SEARCH_API_KEY", "bing-key")
        from app.search.web_search import SerpApiLensProvider, build_search_provider
        p = build_search_provider()
        assert isinstance(p, SerpApiLensProvider)

    def test_bing_fallback(self, monkeypatch):
        monkeypatch.setattr("app.search.web_search.SERPAPI_KEY", "")
        monkeypatch.setattr("app.search.web_search.BING_SEARCH_API_KEY", "bing-key")
        from app.search.web_search import BingVisualSearchProvider, build_search_provider
        p = build_search_provider()
        assert isinstance(p, BingVisualSearchProvider)

    def test_no_keys_raises(self, monkeypatch):
        monkeypatch.setattr("app.search.web_search.SERPAPI_KEY", "")
        monkeypatch.setattr("app.search.web_search.BING_SEARCH_API_KEY", "")
        from app.search.web_search import build_search_provider
        with pytest.raises(EnvironmentError):
            build_search_provider()


# ── result_validator ──────────────────────────────────────────────────────────

class TestResultValidator:
    def _cand(self, url: str) -> SearchCandidate:
        return SearchCandidate(url=url, title="", image_url="https://ex.com/img.jpg")

    def test_filters_pdf(self):
        from app.search.result_validator import filter_accessible_candidates
        cands = [self._cand("https://example.com/resume.pdf")]
        assert filter_accessible_candidates(cands) == []

    def test_keeps_valid_urls(self):
        from app.search.result_validator import filter_accessible_candidates
        cands = [self._cand("https://instagram.com/p/abc")]
        assert len(filter_accessible_candidates(cands)) == 1

    def test_summarise_matches(self):
        from app.face.matcher import MatchResult
        from app.search.result_validator import summarise_matches

        c = SearchCandidate(url="https://ex.com", title="X", image_url="https://ex.com/i.jpg")
        results = [
            MatchResult(candidate=c, matched=True, confidence=0.91),
            MatchResult(candidate=c, matched=False, confidence=0.45),
        ]
        s = summarise_matches(results)
        assert s["total_candidates"] == 2
        assert s["matched"] == 1
        assert abs(s["best_confidence"] - 0.91) < 1e-4


# ── SerpApiYandexProvider & URL Validation ────────────────────────────────────

class TestSerpApiYandexProvider:
    def _provider(self):
        from app.search.web_search import SerpApiYandexProvider
        return SerpApiYandexProvider(api_key="test-key")

    def test_returns_yandex_candidates(self, tmp_path, monkeypatch):
        from urllib.parse import urlparse
        from PIL import Image
        img_path = tmp_path / "face.png"
        Image.new("RGB", (64, 64)).save(img_path)

        monkeypatch.setattr(
            "app.search.web_search._host_image_temporarily",
            lambda _p: "https://tmpfiles.org/dl/1/face.png",
        )

        fake_results = {
            "image_results": [
                {
                    "link": "https://yandex.com/images/search?text=test",
                    "title": "Yandex Result Title",
                    "snippet": "Snippet text",
                    "thumbnail": {"link": "https://example.com/thumb.jpg"},
                }
            ]
        }

        mock_gs = MagicMock()
        mock_gs.return_value.get_dict.return_value = fake_results
        import sys
        fake_mod = MagicMock()
        fake_mod.GoogleSearch = mock_gs
        monkeypatch.setitem(sys.modules, "serpapi", fake_mod)

        provider = self._provider()
        candidates = provider.search(img_path)

        assert len(candidates) == 1
        cand = candidates[0]
        # Assert stored URL is non-empty, contains no truncation ellipsis, and is a valid absolute URL
        assert cand.url != ""
        assert "…" not in cand.url
        parsed = urlparse(cand.url)
        assert parsed.scheme in ("http", "https")
        assert parsed.netloc != ""
