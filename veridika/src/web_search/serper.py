from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import http.client
import json
import logging
import os

from veridika.src.api import ApiHandler, get_api_key
from veridika.src.web_search.default_ban_domains import ban_domains
from veridika.src.web_search.utils import (
    clean_url,
    download_content,
    get_domain_name,
)

# ────────────────────────────────────────────────
# Pricing
# ────────────────────────────────────────────────
_PRICE_PER_QUERY = 50 / 50_000  # USD


class Serper(ApiHandler):
    """
    Thin wrapper around **google.serper.dev** batch search.

    Args:
        pretty_name: Optional alias used in cost tracking. Defaults to ``"serper"``.
    """

    _HOST = "google.serper.dev"
    _SEARCH_ENDPOINT = "/search"

    def __init__(self, *, pretty_name: str | None = None) -> None:
        super().__init__(pretty_name or "serper")
        self._headers = {
            "X-API-KEY": get_api_key("SERPER_API_KEY"),
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------
    @lru_cache(maxsize=256)
    def _domain(self, url: str) -> str:
        return get_domain_name(url)

    def _result_stub(
        self, url: str, *, banned: Sequence[str] | None = None
    ) -> dict[str, str] | None:
        """Return a minimal result dict or ``None`` if the URL is excluded."""
        if url.endswith(".pdf"):
            return None
        if banned and any(b in url for b in banned):
            logging.debug("Serper: Skipping %s (banned domain)", url)
            return None
        return {
            "url": url,
            "title": None,
            "favicon": None,
            "snippet": None,
            "source": self._domain(url),
            "base_url": clean_url(url),
        }

    def _result_stub_with_data(
        self, result: dict[str, str], *, banned: Sequence[str] | None = None
    ) -> dict[str, str] | None:
        """Return a result dict from search data or ``None`` if the URL is excluded."""
        url = result.get("link", "")
        if not url:
            return None
        if url.endswith(".pdf"):
            return None
        if banned and any(b in url for b in banned):
            logging.debug("Serper: Skipping %s (banned domain)", url)
            return None
        return {
            "url": url,
            "title": result.get("title"),
            "favicon": None,
            "snippet": result.get("snippet"),
            "source": self._domain(url),
            "base_url": clean_url(url),
        }

    # ------------------------------------------------------
    # API call
    # ------------------------------------------------------
    def _batch_search(
        self,
        queries: Sequence[str],
        *,
        language: str,
        top_k: int,
        location: str | None,
    ) -> list[dict[str, str]]:
        """
        Call Serper once with a list of queries and return **unique** results with url, title, and snippet.
        Deduplicates by URL and title independently (if either matches, the result is considered duplicate).

        Raises:
            ValueError: if Serper returns an error payload.
        """
        payload = [{"q": q, "hl": language, "gl": location, "num": top_k} for q in queries]
        conn = http.client.HTTPSConnection(self._HOST)
        try:
            conn.request("POST", self._SEARCH_ENDPOINT, json.dumps(payload), self._headers)
            resp = conn.getresponse()
            raw = json.loads(resp.read().decode("utf-8"))
        finally:
            conn.close()

        if len(raw) == 1 and "error" in raw[0]:
            raise ValueError(f"Serper API error: {raw[0]}")

        # Collect all results with deduplication
        seen_urls: set[str] = set()
        seen_titles: set[str] = set()
        results: list[dict[str, str]] = []

        for q_res in raw:
            for item in q_res.get("organic", []):
                url = item.get("link", "")
                title = item.get("title", "")
                snippet = item.get("snippet", "")

                # Skip if we've seen this URL or title before
                if url in seen_urls or title in seen_titles:
                    continue

                # Add to seen sets and results
                if url:
                    seen_urls.add(url)
                if title:
                    seen_titles.add(title)

                results.append({"link": url, "title": title, "snippet": snippet})

        return results

    # ------------------------------------------------------
    # Public search API
    # ------------------------------------------------------
    def __call__(
        self,
        queries: list[str],
        language: str = "es",
        location: str | None = None,
        top_k: int = 10,
        ban_domains: Sequence[str] = ban_domains,
        download_text: bool = True,
        download_favicon: bool = True,
    ) -> tuple[list[dict[str, str]], float]:
        """
        Execute *queries* and return processed documents.

        Args:
            queries: List of query strings.
            language: Interface language for Google (``"en"`` or ``"es"``).
            location: Two-letter country code or ``None``.
            top_k: SERP depth per query.
            ban_domains: Domains to exclude (substring match).
            download_text: If ``True`` fetch page text & favicon.

        Returns:
            (documents, cost)
        """
        # 1. API call
        search_results = self._batch_search(
            queries,
            language=language,
            top_k=top_k,
            location=location,
        )

        # 2. Build preliminary document stubs
        with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as pool:
            docs: list[dict[str, str]] = list(
                filter(
                    None,
                    pool.map(
                        lambda result: self._result_stub_with_data(result, banned=ban_domains),
                        search_results,
                    ),
                )
            )

        # 3. Optional text and favicon download
        self._download_content(
            docs,
            download_text=download_text,
            download_favicon=download_favicon,
            download_title=False,
        )

        # 4. Cost accounting
        cost = len(queries) * _PRICE_PER_QUERY
        self.add_cost(cost)
        return docs, cost

    # ------------------------------------------------------
    # Text / favicon enrichment
    # ------------------------------------------------------
    @staticmethod
    def _download_content(
        docs: list[dict[str, str]],
        download_text: bool = True,
        download_favicon: bool = True,
        download_title: bool = True,
    ) -> None:
        """
        Download *title*, *text* and *favicon* for each document **in-place**.
        Invalid or too-short pages (< 50 words) are removed.
        """
        urls = [d["url"] for d in docs]
        results = download_content(
            urls=urls,
            download_text=download_text,
            download_favicon=download_favicon,
            download_title=download_title,
        )

        keep: list[dict[str, str]] = []
        for (title, text, url, favicon), doc in zip(results, docs, strict=False):
            if download_text and (text is None or len(text.split()) < 50):
                continue

            if download_title:
                doc.update(title=title)
            if download_text:
                doc.update(text=text)
            if download_favicon:
                doc.update(favicon=favicon)
            doc.update(url=url)
            keep.append(doc)

        docs[:] = keep
