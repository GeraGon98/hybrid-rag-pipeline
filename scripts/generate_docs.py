"""Generate the synthetic operations corpus and retrieval evaluation dataset.

Writes 32 documents to `data/docs/` (29 Markdown and 3 PDF files) and 40 questions to
`data/eval_questions.jsonl`. It then loads and chunks the documents through the real
pipeline to verify that every `expected_chunk_id` exists.

Usage:
    uv run scripts/generate_docs.py
"""

from __future__ import annotations

import html
import io
import json
import sys
from pathlib import Path

import pymupdf
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ingest import chunk_documents, load_directory  # noqa: E402

DOCS_DIR = ROOT / "data" / "docs"
EVAL_PATH = ROOT / "data" / "eval_questions.jsonl"
SOURCE_PATH = ROOT / "data" / "source_docs.json"
PDF_IDS = {"RB-11", "SLA-C", "POL-07"}


def load_source() -> tuple[list[dict], list[dict]]:
    source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    questions = [
        {key: value for key, value in json.loads(line).items() if key != "id"}
        for line in EVAL_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return source["documents"], questions


def to_markdown(document: dict) -> str:
    section_slugs = {section["heading"]: section["slug"] for section in document["sections"]}
    metadata = {
        "id": document["id"],
        "title": document["title"],
        "type": document["type"],
        "date": document["date"],
        "section_slugs": section_slugs,
    }
    lines = ["---", yaml.safe_dump(metadata, sort_keys=False).strip(), "---", ""]
    lines.extend((f"# {document['id']} · {document['title']}", ""))
    for section in document["sections"]:
        lines.extend((f"## {section['heading']}", "", section["body"], ""))
    return "\n".join(lines)


def to_pdf(path: Path, document: dict) -> None:
    title = document["title"]
    heading = f"{document['id']} · {title}"
    parts = [f"<h1>{html.escape(heading)}</h1>"]
    for section in document["sections"]:
        parts.append(
            f"<h2>{html.escape(section['heading'])}</h2><p>{html.escape(section['body'])}</p>"
        )
    story = pymupdf.Story(html="".join(parts), user_css="body {font-family: sans-serif;}")
    mediabox = pymupdf.paper_rect("letter")
    where = mediabox + (54, 54, -54, -54)
    buffer = io.BytesIO()
    writer = pymupdf.DocumentWriter(buffer)
    more = True
    while more:
        device = writer.begin_page(mediabox)
        more, _ = story.place(where)
        story.draw(device)
        writer.end_page()
    writer.close()

    section_slugs = {section["heading"]: section["slug"] for section in document["sections"]}
    with pymupdf.open(stream=buffer.getvalue(), filetype="pdf") as pdf:
        pdf.set_metadata(
            {
                "title": title,
                "subject": document["type"],
                "keywords": document["date"],
                "producer": "chunk-slugs:" + json.dumps(section_slugs, separators=(",", ":")),
            }
        )
        pdf.save(path)


def main() -> int:
    documents, questions = load_source()
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    for old in DOCS_DIR.glob("*"):
        if old.suffix in {".md", ".pdf"}:
            old.unlink()

    for document in documents:
        doc_id = document["id"]
        if doc_id in PDF_IDS:
            to_pdf(DOCS_DIR / f"{doc_id}.pdf", document)
        else:
            (DOCS_DIR / f"{doc_id}.md").write_text(
                to_markdown(document), encoding="utf-8"
            )

    with EVAL_PATH.open("w", encoding="utf-8") as output:
        for index, question in enumerate(questions, start=1):
            row = {"id": f"Q{index:02d}", **question}
            output.write(json.dumps(row, ensure_ascii=False) + "\n")

    chunk_ids = {chunk.chunk_id for chunk in chunk_documents(load_directory(DOCS_DIR))}
    expected_ids = {
        chunk_id
        for question in questions
        for chunk_id in question["expected_chunk_ids"]
    }
    missing = sorted(expected_ids - chunk_ids)
    if missing:
        print(f"Expected chunk IDs do not exist: {missing}", file=sys.stderr)
        return 1
    print(
        f"{len(documents)} documents in {DOCS_DIR} · {len(chunk_ids)} chunks · "
        f"{len(questions)} questions in {EVAL_PATH}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
