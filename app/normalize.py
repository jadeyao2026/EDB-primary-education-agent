"""HTML -> clean sections. Traditional text is kept for display, simplified for indexing."""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from urllib.parse import urljoin

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

from .config import PATH_KEYWORDS
from .fetcher import canonical_url, is_allowed_host

HEADING_TAGS = ("h1", "h2", "h3", "h4")

# Removed before hashing: chrome that changes without the content changing.
DROP_TAGS = (
    "script",
    "style",
    "noscript",
    "nav",
    "header",
    "footer",
    "aside",
    "form",
    "iframe",
    "svg",
    "button",
)
DROP_SELECTORS = (
    "#header",
    "#footer",
    "#nav",
    "#navigation",
    "#breadcrumb",
    ".breadcrumb",
    ".nav",
    ".navbar",
    ".menu",
    ".sidebar",
    ".skip-link",
    ".back-to-top",
    ".share",
    ".social",
    ".language",
    ".search",
)
# Most specific first. The EDB template nests real prose inside
# .inner_page_content_container while <header> holds the whole mega-menu, so selecting the
# content container *before* stripping chrome is what keeps short pages from being emptied.
MAIN_SELECTORS = (
    ".inner_page_content_container",
    ".generic_page_content",
    ".inner_page_content",
    "#inner_page_content",
    "main",
    "#main",
    "#content",
    "#main-content",
    ".main-content",
    ".content",
    "article",
)

# A content container can legitimately be small: the primary-education hub page carries only
# a heading plus a link list.
MIN_CONTAINER_CHARS = 40

# A "chrome" element holding more than this share of the text is not chrome.
MAX_STRIP_RATIO = 0.6

# EDB pages carry a single <h1> and no h2/h3. Real subheadings are marked up either with a
# title-ish class or as a short paragraph whose whole content is <strong>. Without treating
# those as headings, a page collapses into one giant chunk and citations become useless.
PSEUDO_HEADING_CLASS_HINTS = ("title", "heading", "subtitle", "subhead")
PSEUDO_HEADING_MAX_CHARS = 60

# Retrieval and citation granularity. Long circular tables get sliced on line boundaries.
MAX_SECTION_CHARS = 1200
MIN_SECTION_CHARS = 10

BLOCK_TAGS = frozenset(
    {
        "p", "div", "li", "td", "th", "tr", "section", "article", "blockquote",
        "dd", "dt", "dl", "ul", "ol", "table", "tbody", "caption", "figcaption",
        "h1", "h2", "h3", "h4", "h5", "h6", "pre", "address", "fieldset", "legend",
    }
)

# Volatile noise that would otherwise make the hash change on every check.
VOLATILE_PATTERNS = (
    re.compile(r"(?:jsessionid|sessionid|sid)=[A-Za-z0-9._-]+", re.I),
    re.compile(r"(?:瀏覽次數|访问次数|visitor count)[：:\s]*[\d,]+"),
    re.compile(r"\b\d{10,13}\b"),  # cache-busting epoch timestamps
)

_CJK = re.compile(r"[\u3000-\u9fff\uf900-\ufaff\uff00-\uffef]")


@lru_cache(maxsize=1)
def _converter():
    """OpenCC t2s. Missing dependency degrades to identity rather than crashing."""
    try:
        from opencc import OpenCC

        return OpenCC("t2s")
    except Exception:  # pragma: no cover - optional at runtime
        return None


def to_simplified(text: str) -> str:
    conv = _converter()
    return conv.convert(text) if conv else text


def collapse_ws(text: str) -> str:
    text = text.replace("\u3000", " ").replace("\xa0", " ")
    return re.sub(r"[ \t\r\f\v]+", " ", text).strip()


def _smart_join(left: str, right: str) -> str:
    """No space between two CJK fragments, a space when Latin text is involved."""
    if not left:
        return right
    if not right:
        return left
    if _CJK.search(left[-1]) and _CJK.search(right[0]):
        return left + right
    if left.endswith(" ") or right.startswith(" "):
        return left + right
    return f"{left} {right}"


def _strip_chrome(node: Tag, guard: bool = True) -> None:
    """Remove nav/header/menu furniture.

    `guard` refuses any single removal that would take more than MAX_STRIP_RATIO of the node's
    text. On this site <header> wraps the entire mega-menu *and* sometimes the content, so an
    unguarded decompose() can empty the page.
    """
    baseline = len(node.get_text(strip=True))
    for target in (*((t,) for t in DROP_TAGS), *DROP_SELECTORS):
        if isinstance(target, tuple):
            found = node.find_all(list(target))
        else:
            found = node.select(target)
        for tag in found:
            if tag is node or not tag.parent:
                continue
            if guard and baseline:
                if len(tag.get_text(strip=True)) / baseline > MAX_STRIP_RATIO:
                    continue
            tag.decompose()
    for comment in node.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()


def _main_container(soup: BeautifulSoup) -> Tag:
    """Pick the content container first, then clean inside it."""
    for selector in MAIN_SELECTORS:
        nodes = soup.select(selector)
        if not nodes:
            continue
        best = max(nodes, key=lambda n: len(n.get_text(strip=True)))
        if len(best.get_text(strip=True)) >= MIN_CONTAINER_CHARS:
            return best
    return soup.body or soup


def page_title(soup: BeautifulSoup) -> str:
    for selector in ("h1", "title"):
        node = soup.select_one(selector)
        if node:
            text = collapse_ws(node.get_text(" "))
            if text:
                return text
    return ""


def _is_pseudo_heading(block: Tag, text: str) -> bool:
    """A short block that is entirely bold, or carries a title-ish class."""
    if len(text) > PSEUDO_HEADING_MAX_CHARS:
        return False
    classes = " ".join(block.get("class") or []).lower()
    if any(hint in classes for hint in PSEUDO_HEADING_CLASS_HINTS):
        return True
    if block.name not in ("p", "div"):
        return False
    emphasis = block.find_all(["strong", "b"])
    if not emphasis:
        return False
    bold_text = collapse_ws(" ".join(e.get_text(" ") for e in emphasis))
    return bool(bold_text) and bold_text == text


def _slice_long(text: str) -> list[str]:
    """Split oversized prose on line boundaries so each chunk stays citable."""
    if len(text) <= MAX_SECTION_CHARS:
        return [text]
    pieces: list[str] = []
    current: list[str] = []
    size = 0
    for line in text.splitlines():
        while len(line) > MAX_SECTION_CHARS:
            if current:
                pieces.append("\n".join(current))
                current, size = [], 0
            pieces.append(line[:MAX_SECTION_CHARS])
            line = line[MAX_SECTION_CHARS:]
        if size + len(line) > MAX_SECTION_CHARS and current:
            pieces.append("\n".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line) + 1
    if current:
        pieces.append("\n".join(current))
    return [p for p in pieces if p.strip()]


def _block_text(block: Tag) -> str:
    """Text of one block element, excluding any nested block's text."""
    out = ""
    for child in block.children:
        if isinstance(child, NavigableString):
            out = _smart_join(out, collapse_ws(str(child)))
        elif isinstance(child, Tag):
            if child.name in BLOCK_TAGS:
                continue
            out = _smart_join(out, collapse_ws(child.get_text(" ")))
    return out.strip()


def split_sections(html: str, url: str) -> tuple[str, list[dict[str, str]]]:
    """Return (page_title, sections). Each section is one h1-h4 heading plus its prose."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all(["script", "style", "noscript"]):
        tag.decompose()
    title = page_title(soup)
    root = _main_container(soup)
    _strip_chrome(root)

    sections: list[dict[str, str]] = []
    current = {"section_title": "", "anchor": "", "parts": []}

    seen_headings: set[str] = set()
    for block in root.find_all(list(BLOCK_TAGS)):
        text = _block_text(block)
        if not text:
            continue
        if block.name in HEADING_TAGS or _is_pseudo_heading(block, text):
            # The template repeats the page title as both a div and an h1; keep one.
            if text in seen_headings and not current["parts"]:
                continue
            seen_headings.add(text)
            if current["parts"]:
                sections.append(current)
            anchor = block.get("id") or ""
            if not anchor:
                nested = block.find(id=True)
                anchor = nested.get("id", "") if nested else ""
            current = {"section_title": text, "anchor": anchor, "parts": []}
        else:
            current["parts"].append(text)

    if current["parts"]:
        sections.append(current)

    result: list[dict[str, str]] = []
    for section in sections:
        body = "\n".join(section["parts"]).strip()
        if len(body) < MIN_SECTION_CHARS:
            continue
        heading = section["section_title"]
        for part in _slice_long(body):
            display = f"{heading}\n{part}".strip() if heading else part
            result.append(
                {
                    "section_title": heading or title,
                    "anchor": section["anchor"],
                    "text_display": display,
                    "text_index": to_simplified(display),
                }
            )

    if not result:
        fallback = collapse_ws(root.get_text("\n"))
        if fallback:
            result.append(
                {
                    "section_title": title,
                    "anchor": "",
                    "text_display": fallback,
                    "text_index": to_simplified(fallback),
                }
            )
    return title, result


def body_text_for_hash(sections: list[dict[str, str]]) -> str:
    """Canonical text used for change detection: section titles plus prose, volatiles removed."""
    joined = "\n\n".join(
        f"## {s['section_title']}\n{s['text_display']}" if s["section_title"] else s["text_display"]
        for s in sections
    )
    for pattern in VOLATILE_PATTERNS:
        joined = pattern.sub("", joined)
    lines = [collapse_ws(line) for line in joined.splitlines()]
    return "\n".join(line for line in lines if line)


def content_hash(body_text: str) -> str:
    return hashlib.sha256(body_text.encode("utf-8")).hexdigest()


def extract_links(html: str, base_url: str) -> list[str]:
    """Links the page itself points at, not the site-wide mega-menu.

    Restricted to the content container for exactly that reason: the EDB template repeats a few
    hundred menu links on every page, and following those would drift off primary education.
    """
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all(["script", "style", "noscript"]):
        tag.decompose()
    root = _main_container(soup)
    _strip_chrome(root)
    found: list[str] = []
    seen: set[str] = set()
    for anchor in root.find_all("a", href=True):
        href = anchor["href"].strip()
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue
        url = canonical_url(urljoin(base_url, href))
        if url in seen or not is_allowed_host(url):
            continue
        path = url.lower()
        if not path.endswith((".html", ".htm", "/")) and "." in path.rsplit("/", 1)[-1]:
            continue  # skip PDFs and other attachments
        if not any(keyword in path for keyword in PATH_KEYWORDS):
            continue
        seen.add(url)
        found.append(url)
    return found
