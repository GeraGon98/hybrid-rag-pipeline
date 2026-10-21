from __future__ import annotations

import pytest

from ingest import Document, chunk_document, slugify
from ingest.chunker import Chunk


def make_doc(text: str, doc_id: str = "RB-01") -> Document:
    return Document(doc_id, f"{doc_id}.md", "Caída del ERP", "runbook", "2026-01-15", text)


TEXT = """# RB-01 · Caída del ERP

## Síntomas

Error ORN-503 al iniciar sesión.

## Diagnóstico

### Base de datos

Revisa el pool de conexiones.
"""


def test_slugify_removes_accents_and_symbols():
    assert slugify("Tiempos de respuesta y resolución") == "tiempos-de-respuesta-y-resolucion"
    assert slugify("¿Cuándo?") == "cuando"
    assert slugify("!!!") == "seccion"


def test_one_chunk_per_section_with_metadata():
    chunks = chunk_document(make_doc(TEXT))

    assert [c.chunk_id for c in chunks] == ["RB-01#sintomas", "RB-01#base-de-datos"]
    first = chunks[0]
    assert first.section == "RB-01 · Caída del ERP > Síntomas"
    assert first.text == "Error ORN-503 al iniciar sesión."
    assert (first.doc_type, first.date, first.source) == ("runbook", "2026-01-15", "RB-01.md")


def test_nested_headings_build_section_path():
    chunks = chunk_document(make_doc(TEXT))
    assert chunks[1].section == "RB-01 · Caída del ERP > Diagnóstico > Base de datos"
    assert chunks[1].indexed_text.startswith(chunks[1].section)


def test_text_before_headings_is_intro():
    chunks = chunk_document(make_doc("Texto sin encabezado."))
    assert chunks[0].chunk_id == "RB-01#intro"
    assert chunks[0].section == "Caída del ERP"


def test_duplicated_headings_get_unique_ids():
    text = "## Notas\n\nuno\n\n## Notas\n\ndos"
    assert [c.chunk_id for c in chunk_document(make_doc(text))] == ["RB-01#notas", "RB-01#notas-2"]


def test_long_section_is_split_with_overlap():
    paragraphs = [f"Párrafo {i} " + "palabra " * 20 for i in range(6)]
    text = "## Procedimiento\n\n" + "\n\n".join(paragraphs)

    chunks = chunk_document(make_doc(text), chunk_size=300, overlap=60)

    assert len(chunks) > 1
    assert chunks[0].chunk_id == "RB-01#procedimiento"
    assert chunks[1].chunk_id == "RB-01#procedimiento-2"
    assert all(len(c.text) <= 300 for c in chunks)
    tail = chunks[0].text[-40:].split(" ", 1)[1]
    assert chunks[1].text.startswith(tail[:20])


def test_very_long_paragraph_is_split_by_words():
    text = "## Sección\n\n" + "palabra " * 200
    chunks = chunk_document(make_doc(text), chunk_size=200, overlap=20)
    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)


def test_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_document(make_doc(TEXT), chunk_size=100, overlap=100)


def test_chunk_roundtrip_dict():
    chunk = chunk_document(make_doc(TEXT))[0]
    assert Chunk.from_dict(chunk.to_dict()) == chunk
