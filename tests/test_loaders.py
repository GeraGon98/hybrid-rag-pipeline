from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from ingest import UnsupportedDocumentError, load_directory, load_document

MARKDOWN = """---
id: RB-07
title: Inventory Synchronization Issue
type: runbook
date: 2026-03-10
---

# RB-07 · Inventory Synchronization Issue

## Description

Ticket T-1042 remains open.
"""


def write_pdf(path: Path) -> None:
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "SLA-C · DataCenter Monterrey", fontsize=20)
    page.insert_text((72, 110), "Physical Access", fontsize=15)
    page.insert_text((72, 140), "Access requires advance registration.", fontsize=11)
    page.insert_text((72, 160), "The manager authorizes emergency access.", fontsize=11)
    pdf.set_metadata({"title": "DataCenter Monterrey", "subject": "sla", "keywords": "2025-09-01"})
    pdf.save(path)
    pdf.close()


def test_markdown_front_matter(tmp_path):
    path = tmp_path / "RB-07.md"
    path.write_text(MARKDOWN, encoding="utf-8")

    doc = load_document(path)

    assert (doc.doc_id, doc.doc_type, doc.date) == ("RB-07", "runbook", "2026-03-10")
    assert doc.title == "Inventory Synchronization Issue"
    assert doc.text.startswith("# RB-07")
    assert "---" not in doc.text


def test_markdown_without_front_matter_uses_file_name(tmp_path):
    path = tmp_path / "NOTA-1.md"
    path.write_text("# Quick Note\n\nText.", encoding="utf-8")

    doc = load_document(path)

    assert doc.doc_id == "NOTA-1"
    assert doc.title == "Quick Note"


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
    assert "## Physical Access" in doc.text
    assert "Access requires advance registration." in doc.text


def test_unsupported_suffix(tmp_path):
    path = tmp_path / "notas.docx"
    path.write_bytes(b"x")
    with pytest.raises(UnsupportedDocumentError, match="unsupported format"):
        load_document(path)


def test_load_directory_is_sorted_and_filtered(tmp_path):
    (tmp_path / "b.md").write_text("# B\n\nb", encoding="utf-8")
    (tmp_path / "a.md").write_text("# A\n\na", encoding="utf-8")
    (tmp_path / "ignore.txt").write_text("x", encoding="utf-8")

    assert [d.doc_id for d in load_directory(tmp_path)] == ["a", "b"]


def test_load_directory_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_directory(tmp_path / "does-not-exist")
