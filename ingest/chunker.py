"""Heading-aware chunking with configurable size and overlap.

Each section (`#`, `##`, `###`) becomes a chunk. Sections exceeding `chunk_size`
characters are split by paragraph (and by word when necessary), with each subsequent
part starting with the previous part's final `overlap` characters.

`chunk_id` is deterministic: `{doc_id}#{section_slug}`, with `-2`, `-3`, ... suffixes
for subsequent parts. This lets the evaluation dataset reference specific chunks.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass

from ingest.loaders import Document

_HEADING = re.compile(r"^(#{1,3})\s+(.+?)\s*#*\s*$")
_SECTION_SEPARATOR = " > "


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    title: str
    section: str
    doc_type: str
    date: str | None
    text: str

    @property
    def indexed_text(self) -> str:
        """Text to index; the section path contextualizes an isolated paragraph."""
        return f"{self.section}\n{self.text}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Chunk:
        return cls(**data)


def slugify(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", normalized.lower()).strip("-") or "section"


def chunk_document(doc: Document, chunk_size: int = 900, overlap: int = 150) -> list[Chunk]:
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and less than chunk_size.")
    chunks: list[Chunk] = []
    used_slugs: dict[str, int] = {}
    for path, body in _sections(doc.text):
        heading = path[-1] if path else ""
        slug = (doc.section_slugs or {}).get(heading) or (slugify(heading) if heading else "intro")
        used_slugs[slug] = used_slugs.get(slug, 0) + 1
        if used_slugs[slug] > 1:
            slug = f"{slug}-{used_slugs[slug]}"
        section = _SECTION_SEPARATOR.join(path) or doc.title
        for index, text in enumerate(_split(body, chunk_size, overlap), start=1):
            suffix = "" if index == 1 else f"-{index}"
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#{slug}{suffix}",
                    doc_id=doc.doc_id,
                    source=doc.source,
                    title=doc.title,
                    section=section,
                    doc_type=doc.doc_type,
                    date=doc.date,
                    text=text,
                )
            )
    return chunks


def chunk_documents(docs: list[Document], chunk_size: int = 900, overlap: int = 150) -> list[Chunk]:
    return [chunk for doc in docs for chunk in chunk_document(doc, chunk_size, overlap)]


def _sections(text: str) -> list[tuple[list[str], str]]:
    """Return non-empty sections as `(heading_path, body)` pairs."""
    sections: list[tuple[list[str], str]] = []
    stack: list[tuple[int, str]] = []
    body: list[str] = []

    def flush() -> None:
        content = "\n".join(body).strip()
        if content:
            sections.append(([title for _, title in stack], content))
        body.clear()

    for line in text.splitlines():
        match = _HEADING.match(line)
        if match:
            flush()
            level = len(match.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, match.group(2)))
        else:
            body.append(line)
    flush()
    return sections


def _split(body: str, chunk_size: int, overlap: int) -> list[str]:
    if len(body) <= chunk_size:
        return [body]
    pieces: list[str] = []
    for paragraph in re.split(r"\n\s*\n", body):
        paragraph = " ".join(paragraph.split())
        if paragraph:
            pieces.extend(_split_words(paragraph, chunk_size - overlap))

    parts: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}\n\n{piece}" if current else piece
        if current and len(candidate) > chunk_size:
            parts.append(current)
            tail = _tail(current, overlap)
            current = f"{tail}\n\n{piece}" if tail else piece
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts


def _split_words(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    pieces, current = [], ""
    for word in text.split():
        candidate = f"{current} {word}" if current else word
        if current and len(candidate) > limit:
            pieces.append(current)
            current = word
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def _tail(text: str, size: int) -> str:
    """Return the final `size` characters without splitting a word."""
    if size <= 0:
        return ""
    tail = text[-size:]
    cut = tail.find(" ")
    return tail[cut + 1 :].strip() if cut != -1 and len(text) > size else tail.strip()
