"""Zero-dependency URL fetcher with HTML-to-text extraction.

Stelle and Cyrene both expose this as a tool so they can resolve links the
client has dropped into ``transcripts/`` (e.g. "SACHIL SENT THIS ARTICLE:
https://forbes.com/...") without hallucinating the content from URL tokens.

Intentionally stdlib-only. `html.parser` is crude compared to trafilatura or
readability-lxml but avoids introducing a new dependency for a small feature.
Good enough for most news / blog articles; paywalls, JS-rendered SPAs, and
login-gated sites return degraded output (login-wall text), which is honest —
the agent sees what an unauthenticated fetcher would see.
"""

from __future__ import annotations

import logging
from html.parser import HTMLParser
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_SECONDS = 15.0
_MAX_RETURN_CHARS = 15000
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# Tags whose bodies should be dropped wholesale (not visible content).
_SKIP_TAGS = {
    "script", "style", "nav", "footer", "header", "aside",
    "noscript", "svg", "form", "button", "iframe",
}

# Block-level tags that should produce a line break when closed, for
# readable plain-text output.
_BLOCK_TAGS = {
    "p", "div", "br", "hr", "li", "h1", "h2", "h3", "h4", "h5", "h6",
    "article", "section", "blockquote", "pre", "tr", "td", "th",
}


class _ReadableExtractor(HTMLParser):
    """Collect visible text, honoring block-tag line breaks and skipping
    structural/scripting tags entirely. Also captures ``<title>``."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0
        self._in_title = False
        self.title: Optional[str] = None

    def handle_starttag(self, tag: str, attrs):  # noqa: ARG002
        tag = tag.lower()
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if tag == "title":
            self._in_title = True
            return
        if tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _SKIP_TAGS:
            if self._skip_depth > 0:
                self._skip_depth -= 1
            return
        if tag == "title":
            self._in_title = False
            return
        if tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_startendtag(self, tag: str, attrs) -> None:  # noqa: ARG002
        # Self-closing tags like <br/> or <hr/>.
        if tag.lower() in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        if self._in_title:
            # Accumulate title text without affecting main body.
            self.title = (self.title or "") + data
            return
        self._parts.append(data)

    def get_text(self) -> str:
        raw = "".join(self._parts)
        lines: list[str] = []
        for line in raw.splitlines():
            s = " ".join(line.split())  # collapse runs of whitespace
            if s:
                lines.append(s)
        # Collapse 3+ consecutive blank-line-like gaps (handled already by the
        # filter above), and join.
        return "\n".join(lines).strip()


def fetch_url(url: str, max_chars: int = _MAX_RETURN_CHARS, timeout: float = _DEFAULT_TIMEOUT_SECONDS) -> dict:
    """Fetch a URL and extract readable plain text.

    Returns a dict with ``url``, ``status`` (int HTTP status or 0 on
    exception), ``title``, ``text``, ``n_chars``, ``truncated``, and
    ``content_type``. Never raises — always returns a dict. On any
    failure the ``text`` field carries a short error explanation so the
    calling agent can reason about what went wrong.
    """
    url = (url or "").strip()
    if not url:
        return {
            "url": "",
            "status": 0,
            "title": None,
            "text": "(empty url)",
            "n_chars": 0,
            "truncated": False,
            "content_type": None,
        }

    if not (url.startswith("http://") or url.startswith("https://")):
        return {
            "url": url,
            "status": 0,
            "title": None,
            "text": "(url must start with http:// or https://)",
            "n_chars": 0,
            "truncated": False,
            "content_type": None,
        }

    try:
        resp = httpx.get(
            url,
            headers={
                "User-Agent": _USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.5",
                "Accept-Language": "en-US,en;q=0.9",
            },
            timeout=timeout,
            follow_redirects=True,
        )
    except Exception as e:
        logger.debug("[fetch_url] %s: request failed: %s", url, e)
        return {
            "url": url,
            "status": 0,
            "title": None,
            "text": f"(request failed: {type(e).__name__}: {str(e)[:200]})",
            "n_chars": 0,
            "truncated": False,
            "content_type": None,
        }

    content_type = (resp.headers.get("content-type") or "").lower()
    status = resp.status_code

    if status >= 400:
        return {
            "url": url,
            "status": status,
            "title": None,
            "text": f"(HTTP {status})",
            "n_chars": 0,
            "truncated": False,
            "content_type": content_type,
        }

    # Only parse text-ish content. Binary (PDFs, images, video) returns a
    # note — agent can decide what to do.
    if "text/html" in content_type or "application/xhtml" in content_type:
        try:
            parser = _ReadableExtractor()
            parser.feed(resp.text)
            parser.close()
            body = parser.get_text()
            title = (parser.title or "").strip() or None
        except Exception as e:
            logger.debug("[fetch_url] %s: parse failed: %s", url, e)
            body = resp.text  # fall back to raw HTML — agent can scan it
            title = None
    elif "text/" in content_type or "application/json" in content_type:
        body = resp.text
        title = None
    else:
        return {
            "url": url,
            "status": status,
            "title": None,
            "text": f"(non-text content: {content_type}, {len(resp.content)} bytes)",
            "n_chars": 0,
            "truncated": False,
            "content_type": content_type,
        }

    truncated = len(body) > max_chars
    return {
        "url": str(resp.url),  # resolved URL after redirects
        "status": status,
        "title": title,
        "text": body[:max_chars],
        "n_chars": len(body),
        "truncated": truncated,
        "content_type": content_type,
    }
