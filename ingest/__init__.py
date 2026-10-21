"""Ingesta: carga de Markdown/PDF y chunking por secciones con metadatos."""

from ingest.chunker import Chunk, chunk_document, chunk_documents, slugify
from ingest.loaders import (
    SUPPORTED_SUFFIXES,
    Document,
    UnsupportedDocumentError,
    load_directory,
    load_document,
)

__all__ = [
    "SUPPORTED_SUFFIXES",
    "Chunk",
    "Document",
    "UnsupportedDocumentError",
    "chunk_document",
    "chunk_documents",
    "load_directory",
    "load_document",
    "slugify",
]
