from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from ingest import UnsupportedDocumentError, load_directory, load_document

MARKDOWN = """---
id: RB-07
titulo: Desincronización de inventario
tipo: runbook
fecha: 2026-03-10
---

# RB-07 · Desincronización de inventario

## Descripción

El ticket T-1042 sigue abierto.
"""


def write_pdf(path: Path) -> None:
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "SLA-C · DataCenter Monterrey", fontsize=20)
    page.insert_text((72, 110), "Acceso físico", fontsize=15)
    page.insert_text((72, 140), "El acceso requiere registro previo.", fontsize=11)
    page.insert_text((72, 160), "Para emergencias autoriza el gerente.", fontsize=11)
    pdf.set_metadata({"title": "DataCenter Monterrey", "subject": "sla", "keywords": "2025-09-01"})
    pdf.save(path)
    pdf.close()


def test_markdown_front_matter(tmp_path):
    path = tmp_path / "RB-07.md"
    path.write_text(MARKDOWN, encoding="utf-8")

    doc = load_document(path)

    assert (doc.doc_id, doc.doc_type, doc.date) == ("RB-07", "runbook", "2026-03-10")
    assert doc.title == "Desincronización de inventario"
    assert doc.text.startswith("# RB-07")
    assert "---" not in doc.text


def test_markdown_without_front_matter_uses_file_name(tmp_path):
    path = tmp_path / "NOTA-1.md"
    path.write_text("# Nota rápida\n\nTexto.", encoding="utf-8")

    doc = load_document(path)

    assert doc.doc_id == "NOTA-1"
    assert doc.title == "Nota rápida"


def test_pdf_metadata_and_headings_from_font_size(tmp_path):
    path = tmp_path / "SLA-C.pdf"
    write_pdf(path)

    doc = load_document(path)

    assert (doc.doc_id, doc.title, doc.doc_type, doc.date) == (
        "SLA-C",
        "DataCenter Monterrey",
        "sla",
        "2025-09-01",
    )
    assert doc.text.splitlines()[0] == "# SLA-C · DataCenter Monterrey"
    assert "## Acceso físico" in doc.text
    assert "El acceso requiere registro previo." in doc.text


def test_unsupported_suffix(tmp_path):
    path = tmp_path / "notas.docx"
    path.write_bytes(b"x")
    with pytest.raises(UnsupportedDocumentError, match="no soportado"):
        load_document(path)


def test_load_directory_is_sorted_and_filtered(tmp_path):
    (tmp_path / "b.md").write_text("# B\n\nb", encoding="utf-8")
    (tmp_path / "a.md").write_text("# A\n\na", encoding="utf-8")
    (tmp_path / "ignorar.txt").write_text("x", encoding="utf-8")

    assert [d.doc_id for d in load_directory(tmp_path)] == ["a", "b"]


def test_load_directory_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_directory(tmp_path / "no-existe")
