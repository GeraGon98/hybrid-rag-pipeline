"""Chunking por encabezados con tamaño y solapamiento configurables.

Cada sección (`#`, `##`, `###`) es un chunk. Si una sección supera `chunk_size`
caracteres, se parte por párrafos (y por palabras si un párrafo no cabe) y cada parte
arranca con los últimos `overlap` caracteres de la anterior.

`chunk_id` es determinista: `{doc_id}#{slug_de_la_seccion}`, con sufijo `-2`, `-3`... en
las partes siguientes. Así el dataset de evaluación puede referirse a chunks concretos.
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
        """Texto que se indexa: la ruta de la sección da contexto a un párrafo suelto."""
        return f"{self.section}\n{self.text}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Chunk:
        return cls(**data)


def slugify(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", normalized.lower()).strip("-") or "seccion"


def chunk_document(doc: Document, chunk_size: int = 900, overlap: int = 150) -> list[Chunk]:
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap debe ser >= 0 y menor que chunk_size.")
    chunks: list[Chunk] = []
    used_slugs: dict[str, int] = {}
    for path, body in _sections(doc.text):
        slug = slugify(path[-1]) if path else "intro"
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
    """Secciones con cuerpo como (ruta de encabezados, cuerpo). Las vacías se omiten."""
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
    """Últimos `size` caracteres sin cortar una palabra a la mitad."""
    if size <= 0:
        return ""
    tail = text[-size:]
    cut = tail.find(" ")
    return tail[cut + 1 :].strip() if cut != -1 and len(text) > size else tail.strip()
