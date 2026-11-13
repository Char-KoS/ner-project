"""PDF text extraction with PyMuPDF (with OCR for scanned PDFs)."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import os

import pymupdf


TESSDATA_DIR = Path(os.environ.get(
    "TESSDATA_PREFIX",
    Path(__file__).resolve().parent.parent / "tessdata",
))


def find_pdf(pdf_dir: str | Path, doc_id: str) -> Path | None:
    pdf_dir = Path(pdf_dir)
    for name in (f"{doc_id}.pdf", f"{doc_id}_scanned.pdf"):
        if (pdf_dir / name).exists():
            return pdf_dir / name
    return None


def _page_text(page, mode: str, textpage=None) -> list[str]:
    if mode == "raw":
        return [page.get_text(textpage=textpage)]

    paragraphs = []
    for block in page.get_text("blocks", textpage=textpage, sort=True):
        if block[6] == 0:
            paragraph = re.sub(r"\s+", " ", block[4]).strip()   # join the lines of the block
            if paragraph:
                paragraphs.append(paragraph)
    return paragraphs


def extract_text_from_pdf(
    pdf_path: str | Path,
    mode: str = "blocks",
    ocr: bool = False,
    language: str = "spa+cat",
    dpi: int = 300,
) -> str:
    """Extract the text of a PDF.
    mode = "raw"    -> keeps the PDF line breaks as they are
    mode = "blocks" -> one block per paragraph: lines joined with spaces,
                       paragraphs separated by a blank line (\n\n)
    ocr  = True     -> read the text from the page image with Tesseract (scanned PDFs)
    """
    parts: list[str] = []
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            textpage = None
            if ocr:
                # Render the page at `dpi` and recognise the text with Tesseract
                textpage = page.get_textpage_ocr(
                    language=language, dpi=dpi, full=True, tessdata=str(TESSDATA_DIR)
                )
            parts.extend(_page_text(page, mode, textpage))

    return "".join(parts) if mode == "raw" else "\n\n".join(parts)


def extract_pdf_dataset(
    pdf_dir: str | Path,
    gold_samples: list[dict[str, Any]],
    mode: str = "blocks",
    ocr: bool = False,
) -> list[dict[str, Any]]:
    """Extract the PDFs following the order of the gold file (same ids)."""
    from tqdm.auto import tqdm   # progress bar (OCR is slow)

    samples = []
    for gold in tqdm(gold_samples, desc="PyMuPDF"):
        pdf_path = find_pdf(pdf_dir, gold["id"])
        if pdf_path is None:
            print(f"No PDF for {gold['id']}: stopping at {len(samples)} documents.")
            break
        samples.append({
            "id": gold["id"],
            "text": extract_text_from_pdf(pdf_path, mode=mode, ocr=ocr),
            "language": gold.get("language"),
        })
    return samples