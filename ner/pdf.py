"""PDF text extraction with PyMuPDF."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pymupdf


def find_pdf(pdf_dir: str | Path, doc_id: str) -> Path | None:
    """Find the PDF of a document: '<id>.pdf' or '<id>_scanned.pdf'."""
    pdf_dir = Path(pdf_dir)
    for name in (f"{doc_id}.pdf", f"{doc_id}_scanned.pdf"):
        if (pdf_dir / name).exists():
            return pdf_dir / name
    return None


def extract_text_from_pdf(pdf_path: str | Path, mode: str = "blocks") -> str:
    """Extract the text of a PDF.

    mode = "raw"    -> page.get_text(): keeps the PDF line breaks as they are
    mode = "blocks" -> one block per paragraph: lines joined with spaces,
                       paragraphs separated by a blank line (\n\n)
    """
    with pymupdf.open(pdf_path) as doc:
        if mode == "raw":
            return "".join(page.get_text() for page in doc)

        paragraphs = []
        for page in doc:
            # Each block = (x0, y0, x1, y1, text, block_no, block_type); type 0 = text
            for block in page.get_text("blocks", sort=True):
                if block[6] == 0:
                    paragraph = re.sub(r"\s+", " ", block[4]).strip()   # join the lines
                    if paragraph:
                        paragraphs.append(paragraph)
        return "\n\n".join(paragraphs)


def extract_pdf_dataset(
    pdf_dir: str | Path,
    gold_samples: list[dict[str, Any]],
    mode: str = "blocks",
) -> list[dict[str, Any]]:
    """Extract the PDFs following the order of the gold file (same ids)."""
    samples = []
    for gold in gold_samples:
        pdf_path = find_pdf(pdf_dir, gold["id"])
        if pdf_path is None:
            # Stop at the first missing PDF so the ids stay aligned with the gold file
            print(f"No PDF for {gold['id']}: stopping at {len(samples)} documents.")
            break
        samples.append({
            "id": gold["id"],
            "text": extract_text_from_pdf(pdf_path, mode=mode),
            "language": gold.get("language"),
        })
    return samples