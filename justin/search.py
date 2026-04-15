from __future__ import annotations

import json
import re
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from urllib import parse, request

from .types import Citation, now_iso


@dataclass(slots=True)
class SearchItem:
    title: str
    url: str
    snippet: str
    source: str
    fetched_at: str
    confidence: float = 0.5


class SearchProvider:
    name = "search"

    def search(self, query: str, top_k: int, locale: str) -> list[SearchItem]:
        raise NotImplementedError


class DuckDuckGoHTMLSearchProvider(SearchProvider):
    name = "duckduckgo"

    def search(self, query: str, top_k: int, locale: str) -> list[SearchItem]:
        encoded = parse.urlencode({"q": query, "kl": _duckduckgo_region(locale)})
        url = f"https://html.duckduckgo.com/html/?{encoded}"
        html = _read_text(url)
        parser = _DuckDuckGoResultParser()
        parser.feed(html)
        parser.close()

        items: list[SearchItem] = []
        for result in parser.results[:top_k]:
            title = result.get("title", "").strip()
            href = _normalize_duckduckgo_url(result.get("url", "").strip())
            if not title or not href:
                continue
            items.append(
                SearchItem(
                    title=title,
                    url=href,
                    snippet=result.get("snippet", "").strip(),
                    source=self.name,
                    fetched_at=now_iso(),
                    confidence=0.74,
                )
            )
        return items


class WikipediaSearchProvider(SearchProvider):
    name = "wikipedia"

    def search(self, query: str, top_k: int, locale: str) -> list[SearchItem]:
        language = "zh" if locale.lower().startswith("zh") else "en"
        params = parse.urlencode(
            {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": str(max(1, min(top_k, 10))),
                "utf8": "1",
                "format": "json",
            }
        )
        url = f"https://{language}.wikipedia.org/w/api.php?{params}"
        payload = json.loads(_read_text(url))
        results = payload.get("query", {}).get("search", [])
        items: list[SearchItem] = []
        for entry in results[:top_k]:
            title = str(entry.get("title", "")).strip()
            if not title:
                continue
            page_url = f"https://{language}.wikipedia.org/wiki/{parse.quote(title.replace(' ', '_'))}"
            snippet = _strip_html(str(entry.get("snippet", "")))
            items.append(
                SearchItem(
                    title=title,
                    url=page_url,
                    snippet=snippet,
                    source=self.name,
                    fetched_at=now_iso(),
                    confidence=0.67,
                )
            )
        return items


class SearchService:
    def __init__(
        self,
        store,
        providers: list[SearchProvider] | None = None,
        cache_ttl_hours: int = 12,
    ) -> None:
        self.store = store
        self.providers = providers or [DuckDuckGoHTMLSearchProvider(), WikipediaSearchProvider()]
        self.cache_ttl_hours = max(int(cache_ttl_hours), 1)

    def search(self, query: str, top_k: int = 5, locale: str = "en-US") -> list[SearchItem]:
        normalized_query = query.strip()
        if not normalized_query:
            return []

        merged: list[SearchItem] = []
        for provider in self.providers:
            cache_key = f"{provider.name}:{locale.lower()}:{normalized_query.lower()}"
            cached = self.store.get_search_cache(cache_key)
            if cached is None:
                results = provider.search(normalized_query, top_k=top_k, locale=locale)
                self.store.save_search_cache(
                    cache_key=cache_key,
                    provider=provider.name,
                    query=normalized_query,
                    locale=locale,
                    results=results,
                    ttl_hours=self.cache_ttl_hours,
                )
            else:
                results = [SearchItem(**item) for item in cached]
            merged.extend(results)
            if len(self._dedupe_and_rank(normalized_query, merged, top_k)) >= top_k:
                break
        return self._dedupe_and_rank(normalized_query, merged, top_k)

    def citations_for(self, items: list[SearchItem]) -> list[Citation]:
        citations: list[Citation] = []
        for index, item in enumerate(items, start=1):
            label = f"S{index}"
            citations.append(
                Citation(
                    id=label.lower(),
                    label=label,
                    title=item.title,
                    url=item.url,
                    snippet=item.snippet,
                    source=item.source,
                )
            )
        return citations

    def _dedupe_and_rank(self, query: str, items: list[SearchItem], top_k: int) -> list[SearchItem]:
        query_terms = {term for term in re.findall(r"[a-zA-Z0-9\u4e00-\u9fff]+", query.lower()) if term}
        ranked: dict[str, tuple[float, SearchItem]] = {}
        for order, item in enumerate(items):
            key = item.url.strip().lower() or item.title.strip().lower()
            if not key:
                continue
            haystack = f"{item.title} {item.snippet}".lower()
            overlap = sum(1 for term in query_terms if term in haystack)
            score = (item.confidence * 10) + overlap - (order * 0.01)
            current = ranked.get(key)
            if current is None or score > current[0]:
                ranked[key] = (score, item)
        return [pair[1] for pair in sorted(ranked.values(), key=lambda pair: pair[0], reverse=True)[:top_k]]


class _DuckDuckGoResultParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, str]] = []
        self._current: dict[str, str] | None = None
        self._capture_title = False
        self._capture_snippet = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {key: value or "" for key, value in attrs}
        classes = attr_map.get("class", "")
        if tag == "a" and "result__a" in classes:
            self._flush()
            self._current = {"title": "", "url": attr_map.get("href", ""), "snippet": ""}
            self._capture_title = True
            return
        if self._current and tag in {"a", "div"} and "result__snippet" in classes:
            self._capture_snippet = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._capture_title:
            self._capture_title = False
        if tag in {"a", "div"} and self._capture_snippet:
            self._capture_snippet = False

    def handle_data(self, data: str) -> None:
        if not self._current:
            return
        if self._capture_title:
            self._current["title"] += data
        elif self._capture_snippet:
            self._current["snippet"] += data

    def _flush(self) -> None:
        if not self._current:
            return
        title = _normalize_whitespace(self._current.get("title", ""))
        href = self._current.get("url", "").strip()
        if title and href:
            self.results.append(
                {
                    "title": title,
                    "url": href,
                    "snippet": _normalize_whitespace(self._current.get("snippet", "")),
                }
            )
        self._current = None
        self._capture_title = False
        self._capture_snippet = False


def _read_text(url: str, timeout: int = 20) -> str:
    req = request.Request(
        url,
        headers={
            "User-Agent": "Justin-Agent/0.2 (+https://github.com/RichGua/Justin-Agent)",
            "Accept-Language": "en-US,en;q=0.8,zh-CN;q=0.6",
        },
    )
    with request.urlopen(req, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def _normalize_duckduckgo_url(raw_url: str) -> str:
    if raw_url.startswith("//"):
        return f"https:{raw_url}"
    parsed = parse.urlparse(raw_url)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        qs = parse.parse_qs(parsed.query)
        target = qs.get("uddg", [raw_url])[0]
        return parse.unquote(target)
    return raw_url


def _strip_html(value: str) -> str:
    return _normalize_whitespace(re.sub(r"<[^>]+>", " ", unescape(value)))


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", unescape(value)).strip()


def _duckduckgo_region(locale: str) -> str:
    normalized = locale.lower()
    if normalized.startswith("zh"):
        return "cn-zh"
    return "us-en"
