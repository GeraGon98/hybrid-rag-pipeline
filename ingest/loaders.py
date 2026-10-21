"""Carga de Markdown y PDF a documentos con metadatos.

- Markdown: front-matter YAML con `id`, `titulo`, `tipo` y `fecha`.
- PDF: metadatos del archivo (`title`, `subject` = tipo, `keywords` = fecha) con respaldo
  en el nombre del archivo y la primera línea. Los encabezados se reconstruyen como
  Markdown (`#`, `##`, `###`) a partir del tamaño de fuente para que el chunker los use.
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pymupdf
import yaml

SUPPORTED_SUFFIXES = frozenset({".md", ".pdf"})
_FRONT_MATTER = "---"
_HEADING_RATIO = 1.15


class UnsupportedDocumentError(ValueError):
    """El archivo no es Markdown ni PDF, o no se pudo leer."""


@dataclass(frozen=True)
class Document:
    doc_id: str
    source: str
    title: str
    doc_type: str
    date: str | None
    text: str


def load_document(path: Path) -> Document:
    suffix = path.suffix.lower()
    if suffix == ".md":
        return _load_markdown(path)
    if suffix == ".pdf":
        return _load_pdf(path)
    raise UnsupportedDocumentError(
        f"{path.name}: formato no soportado ({suffix or 'sin extensión'})."
    )


def load_directory(directory: Path) -> list[Document]:
    """Carga todos los documentos soportados de una carpeta (recursivo), en orden estable."""
    if not directory.is_dir():
        raise FileNotFoundError(f"No existe la carpeta {directory}.")
    paths = sorted(p for p in directory.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES)
    return [load_document(p) for p in paths]


def _load_markdown(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8")
    meta, body = _split_front_matter(raw, path)
    return Document(
        doc_id=str(meta.get("id") or path.stem),
        source=path.name,
        title=str(meta.get("titulo") or _first_line(body) or path.stem),
        doc_type=str(meta.get("tipo") or "documento"),
        date=str(meta["fecha"]) if meta.get("fecha") else None,
        text=body.strip(),
    )


def _split_front_matter(raw: str, path: Path) -> tuple[dict, str]:
    if not raw.startswith(_FRONT_MATTER):
        return {}, raw
    parts = raw.split(_FRONT_MATTER, 2)
    if len(parts) < 3:
        raise UnsupportedDocumentError(f"{path.name}: front-matter sin cerrar.")
    meta = yaml.safe_load(parts[1]) or {}
    if not isinstance(meta, dict):
        raise UnsupportedDocumentError(f"{path.name}: el front-matter no es un mapa YAML.")
    return meta, parts[2]


def _load_pdf(path: Path) -> Document:
    try:
        pdf = pymupdf.open(path)
    except Exception as e:
        raise UnsupportedDocumentError(f"{path.name}: no se pudo abrir el PDF ({e}).") from e
    with pdf:
        meta = pdf.metadata or {}
        lines = [line for page in pdf for line in _page_lines(page)]
    text = _lines_to_markdown(lines)
    return Document(
        doc_id=path.stem,
        source=path.name,
        title=meta.get("title") or _first_line(text) or path.stem,
        doc_type=meta.get("subject") or "documento",
        date=meta.get("keywords") or None,
        text=text,
    )


def _page_lines(page) -> list[tuple[float, str]]:
    """Líneas de la página como (tamaño de fuente máximo, texto)."""
    lines = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = "".join(span["text"] for span in line["spans"])
            text = unicodedata.normalize("NFKC", text).strip()  # Ligaduras: "ﬁ" -> "fi".
            if text:
                lines.append((max(span["size"] for span in line["spans"]), text))
    return lines


def _lines_to_markdown(lines: list[tuple[float, str]]) -> str:
    """Convierte líneas con tamaño de fuente en Markdown con encabezados y párrafos."""
    if not lines:
        return ""
    body_size = Counter(round(size, 1) for size, _ in lines).most_common(1)[0][0]
    heading_sizes = sorted(
        {round(size, 1) for size, _ in lines if size >= body_size * _HEADING_RATIO}, reverse=True
    )
    levels = {size: min(index + 1, 3) for index, size in enumerate(heading_sizes)}

    out: list[str] = []
    paragraph: list[str] = []
    last_level: int | None = None

    def flush() -> None:
        if paragraph:
            out.append(" ".join(paragraph))
            paragraph.clear()

    for size, text in lines:
        level = levels.get(round(size, 1))
        if level and level == last_level:
            out[-1] = f"{out[-1]} {text}"  # Encabezado largo partido en varias líneas.
        elif level:
            flush()
            out.append(f"{'#' * level} {text}")
        else:
            paragraph.append(text)
            if text.endswith((".", ":", "?", "!")):
                flush()
        last_level = level
    flush()
    return "\n\n".join(out)


def _first_line(text: str) -> str:
    for line in text.splitlines():
        line = line.strip().lstrip("#").strip()
        if line:
            return line
    return ""
