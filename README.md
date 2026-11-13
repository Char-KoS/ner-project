# NER Project — Named Entity Recognition on administrative documents

Pt1.2 · Intel·ligència Artificial · 

Pipeline to extract named entities (people, ID numbers, addresses, registration numbers, dates,
amounts and organizations) from Spanish and Catalan administrative documents, using **GLiNER** on
plain text, clean PDFs and scanned PDFs (**PyMuPDF** + OCR), and comparing it with **LLM chats**.

---

## 1.- Group Members
- **Alexander Mansilla Arteaga** ([@Char-KoS](https://github.com/Char-KoS))

**Work organization:**
(2–4 lines in English. For example: how you split the tasks, that you worked in PyCharm with uv and
Jupyter notebooks, that you used Git with one commit per step, how long it took…)

---

## 2. System and configuration

### Pipeline
```
Task 1  plain text (JSONL)
Task 2  clean PDF   ──► PyMuPDF (text blocks)
Task 3  scanned PDF ──► PyMuPDF + Tesseract OCR (spa+cat)
Task 4  plain text  ──► prompt ──► LLM chat ──► answer ──► offsets
```
1. **Chunking:** GLiNER can only read ~384 tokens at once, and the documents have ~4,000 characters,
   so each document is split into chunks of complete sentences (≤ 1,000 characters). The entity
   offsets of each chunk are moved back to the full text.
2. **NER:** GLiNER `predict_entities` with the 7 entity labels.
3. **Merge post-processing:** GLiNER splits long entities, so consecutive entities with the same label
   are merged when the text between them is only commas/numbers (ADDRESS) or a preposition
   (`de`, `de la`, `del`… for ORGANIZATION).
4. **Evaluation:** teacher's `evaluate_ner_strict()` and `evaluate_ner_relaxed()` (occurrences and
   unique views), micro metrics and metrics per entity type.

### Configuration

| Item | Value |
|---|---|
| GLiNER model | `gliner-community/gliner_small-v2.5` (small) |
| Threshold | 0.5 |
| Label descriptions | Tested, **not used** in the final configuration (they made results worse) |
| Labels | `PERSON_NAME`, `ID_CARD_NUMBER`, `ADDRESS`, `REGISTRATION_NUMBER`, `DATE`, `AMOUNT`, `ORGANIZATION` |
| PDF extraction | PyMuPDF 2.1 — `page.get_text("blocks")` (keeps the paragraphs) |
| OCR | Tesseract (built into PyMuPDF), `tessdata_fast` models **spa + cat**, 300 dpi |
| LLM chats (Task 4) | ChatGPT (model: ...) and Gemini (model: ...), web chat, no API |
| Environment | Python 3.13.17 (python.org), `uv`, PyTorch 2.8.0 (CPU), PyCharm + Jupyter |
| Hardware | Windows 11 laptop, **CPU only** (no GPU) |

---

## 3. Results summary

Micro precision (P), recall (R) and F1. Tasks 1 and 2 use the 50 documents, Task 3 the 10 scanned
PDFs available and Task 4 the first 5 documents. In Tasks 3 and 4, GLiNER on plain text is also
evaluated on the same documents, so the comparison is fair.

| Input | Model | Label descriptions | Extraction | Evaluation | P | R | F1 |
|---|---|---|---|---|---|---|---|
| Plain text (50) | gliner_small | No | — | Strict | 0.74 | 0.85 | 0.79 |
| Plain text (50) | gliner_small | No | — | Relaxed occurrences | 0.74 | 0.85 | 0.79 |
| Plain text (50) | gliner_small | No | — | Relaxed unique | 0.72 | 0.88 | 0.79 |
| Plain text (50) | gliner_small + merge | No | — | Strict | 0.88 | 0.92 | **0.90** |
| Plain text (50) | gliner_small + merge | No | — | Relaxed occurrences | 0.88 | 0.92 | 0.90 |
| Plain text (50) | gliner_small + merge | No | — | Relaxed unique | 0.84 | 0.94 | 0.89 |
| Plain text (50) | gliner_small + merge | Yes | — | Strict | 0.84 | 0.83 | 0.83 |
| Plain text (50) | gliner_small + merge | Yes | — | Relaxed occurrences | 0.84 | 0.83 | 0.83 |
| Plain text (50) | gliner_small + merge | Yes | — | Relaxed unique | 0.81 | 0.89 | 0.85 |
| Clean PDF (50) | gliner_small + merge | No | PyMuPDF `get_text()` (raw) | Strict | 0.60 | 0.64 | 0.62 |
| Clean PDF (50) | gliner_small + merge | No | PyMuPDF `get_text()` (raw) | Relaxed occurrences | 0.88 | 0.92 | 0.90 |
| Clean PDF (50) | gliner_small + merge | No | PyMuPDF `get_text()` (raw) | Relaxed unique | 0.84 | 0.94 | 0.89 |
| Clean PDF (50) | gliner_small + merge | No | PyMuPDF blocks | Strict | 0.88 | 0.92 | **0.90** |
| Clean PDF (50) | gliner_small + merge | No | PyMuPDF blocks | Relaxed occurrences | 0.88 | 0.92 | 0.90 |
| Clean PDF (50) | gliner_small + merge | No | PyMuPDF blocks | Relaxed unique | 0.84 | 0.94 | 0.89 |
| Plain text (10) | gliner_small + merge | No | — | Strict | 0.87 | 0.92 | 0.90 |
| Plain text (10) | gliner_small + merge | No | — | Relaxed occurrences | 0.87 | 0.92 | 0.90 |
| Plain text (10) | gliner_small + merge | No | — | Relaxed unique | 0.83 | 0.94 | 0.88 |
| Scanned PDF (10) | gliner_small + merge | No | PyMuPDF + OCR (spa+cat) | Strict | 0.66 | 0.70 | **0.68** |
| Scanned PDF (10) | gliner_small + merge | No | PyMuPDF + OCR (spa+cat) | Relaxed occurrences | 0.78 | 0.83 | 0.81 |
| Scanned PDF (10) | gliner_small + merge | No | PyMuPDF + OCR (spa+cat) | Relaxed unique | 0.73 | 0.85 | 0.78 |
| Plain text (5) | gliner_small + merge | No | — | Strict | 0.91 | 0.94 | 0.93 |
| Plain text (5) | gliner_small + merge | No | — | Relaxed occurrences | 0.91 | 0.94 | 0.93 |
| Plain text (5) | gliner_small + merge | No | — | Relaxed unique | 0.88 | 0.94 | 0.91 |
| Plain text (5) | LLM chat: ChatGPT | — | — | Strict / Relaxed occ. / Relaxed unique | 1.00 | 1.00 | **1.00** |
| Plain text (5) | LLM chat: Gemini | — | — | Strict / Relaxed occ. / Relaxed unique | 1.00 | 1.00 | **1.00** |

### Per entity type — best GLiNER configuration (plain text, 50 documents, strict F1)

| PERSON_NAME | ID_CARD_NUMBER | ADDRESS | REGISTRATION_NUMBER | DATE | AMOUNT | ORGANIZATION |
|---|---|---|---|---|---|---|
| 0.98 | 0.82 | 0.71 | 0.93 | 0.99 | 0.79 | 0.87 |

---

## 4. Difficulties and decisions

### Model and NER
- **Long documents:** GLiNER only reads ~384 tokens, but the documents have ~4,000 characters.
  Each document is split into chunks of complete sentences (≤ 1,000 characters) and the offsets of
  the entities are corrected with the position of each chunk in the full text.
- **Split entities:** the baseline split addresses and organization names into pieces
  (*Avenida de Aragón* + *102, 2-B* + *17001 Girona*), so ADDRESS had a strict F1 of only 0.04.
  We added a merge post-processing step that joins adjacent pieces with the same label:
  strict F1 0.79 → 0.90 (ADDRESS 0.04 → 0.71).
- **Label descriptions** made the model worse (strict F1 0.90 → 0.83, mainly because of recall),
  so we kept the short label names.

### PDF extraction and OCR
- **Clean PDFs:** `page.get_text()` puts a line break where the original text has a space and does
  not keep the blank line between paragraphs, so the offsets were shifted (strict F1 0.62).
  Using `get_text("blocks")` and joining the lines of each block with spaces rebuilds the original
  text exactly (50/50 documents identical).
- **Scanned PDFs** contain only an image per page (0 characters without OCR). The OCR is done with
  the Tesseract engine included in PyMuPDF (`page.get_textpage_ocr`, 300 dpi).
- **Catalan documents:** some documents are in Catalan. With only the Spanish model the OCR failed on
  `ç`, `·`, `à`, `è`. Using **`spa+cat`** the character similarity with the original text is
  ≥ 0.997 for all 10 documents.
- **Windows path with special characters:** Tesseract could not load the language files from the
  project folder because the path contains `º` and spaces (`2º DAM`). The tessdata folder is
  configured with the `TESSDATA_PREFIX` environment variable.
- `difflib.SequenceMatcher` with default settings gave a wrong similarity (0.76) for one long
  document; `autojunk=False` fixed the measurement.
- Only **10 scanned PDFs** are available, so Task 3 is compared with plain text on the same
  10 documents.

### LLM chats
- LLMs give no character offsets, so they are computed by searching each entity in the original text
  (repeated mentions take the next occurrence).
- The prompt had to be copied with PowerShell (`Get-Content ... | Set-Clipboard`) because a manual
  copy was incomplete.
- The copied answers contained Markdown escapes (`\_`, `\[`) and all the documents on one line;
  the parser in `ner/llm.py` handles both cases.

### Environment (Windows)
- The Hugging Face model download (Xet backend) froze for hours; it worked after disabling it with
  `HF_HUB_DISABLE_XET=1`.
- **Windows Smart App Control** blocked unsigned DLLs of the Python installed by `uv` and of the
  newest PyTorch. We solved it without disabling the protection: official python.org Python 3.13
  (`python-preference = "only-system"` in `pyproject.toml`) and PyTorch **2.8.0**.
- PyCharm could not manage the `uv` virtual environment because it has no `pip`; `pip` was added
  as a dev dependency with `uv add --dev pip`.

---

## 5. Conclusions

### Main results
- On **plain text**, GLiNER small + merge reaches a strict F1 of **0.90** (baseline 0.79).
- **Clean PDFs** reach the same score (0.90) when the extraction keeps the paragraph structure
  (PyMuPDF blocks); with a raw extraction the strict score drops to 0.62.
- **Scanned PDFs** drop to **0.68 strict / 0.81 relaxed**, compared with 0.90 / 0.90 on the same
  10 documents in plain text.
- The **LLM chats** (ChatGPT and Gemini) were perfect (F1 1.00) on 5 documents, where GLiNER got 0.93.

### Most difficult entity types
- **ADDRESS** (strict F1 0.71): long entities with several parts (street, number, floor, postal
  code, city) that the model splits into pieces.
- **AMOUNT** (0.79): different formats (`4.416,13 euros`, `1.107,16 EUR`, `5.523,29 €`).
- **ID_CARD_NUMBER** (0.82): the model finds every ID number (unique recall 0.98) but misses some
  repeated mentions.
- With **OCR**, **ORGANIZATION** is the most affected type (relaxed F1 0.89 → 0.49): many names are in
  Catalan and contain accents that the OCR reads differently (`Econòmica` → `Económica`).
- **DATE** and **PERSON_NAME** are almost perfect in all the scenarios (F1 ≥ 0.94).

### Plain text vs PDF
- In **clean PDFs** the text is the same as in plain text: GLiNER finds exactly the same entities
  (identical relaxed scores). The only difference is the **character offsets**, which depend on how
  the text is extracted (line breaks, paragraph separators).
- In **scanned PDFs** the text itself changes because of OCR errors: missing spaces shift the
  offsets, and wrong characters inside entities (`12082063W` → `12082063VV`) change the entity text.
  For this reason both strict and relaxed scores drop.

### Strict vs relaxed evaluation
- **Strict** requires the exact offsets, so it is very sensitive to the extraction step
  (raw clean PDF: 0.62 strict vs 0.90 relaxed). It measures the whole pipeline.
- **Relaxed** only compares the label and the entity text, so it measures whether the NER model finds
  the right entities, which is more useful when the text comes from PDFs.
- With OCR, strict can even count as correct an entity with a wrong letter, if replacing the letter
  does not change the offsets.
- **Relaxed unique** counts each entity once per document, so it shows whether the model finds every
  different entity even if it misses some repetitions.

### GLiNER vs LLM chat
- The LLMs were more accurate, but the comparison is limited: only 5 documents, and the prompt
  included **annotation rules** (full address, amount with currency, one entity per mention) that
  GLiNER does not receive.
- A chat **does not scale** (manual copy/paste, length limits), gives no character offsets, and
  sending real administrative documents (ID numbers, addresses) to an external service is a
  **privacy** risk.
- GLiNER is small, free, runs **locally on CPU** and returns offsets directly, so it is a better
  option for processing many documents with personal data.

---

## 6. GitHub repository

**Repository:** https://github.com/Char-KoS/ner-project

### Structure

```
Practica2 (ner_project)
├── data/
│   ├── plain/validation.jsonl        gold standard (50 documents)
│   ├── pdf_clean/                    test-0001.pdf … test-0050.pdf
│   └── pdf_scanned/                  test-0001_scanned.pdf … test-0010_scanned.pdf
├── ner/
│   ├── evaluation.py                 strict and relaxed evaluation (teacher)
│   ├── utils.py                      JSONL helpers (teacher)
│   ├── inference.py                  GLiNER loading, chunking, prediction and merge
│   ├── pdf.py                        PyMuPDF extraction (raw / blocks) + OCR
│   └── llm.py                        prompt, LLM answer parsing and offsets
├── notebooks/
│   ├── 00_exploration.ipynb          GLiNER basic usage, spans, PyMuPDF quick start
│   ├── 01_plain_text.ipynb           Task 1
│   ├── 02_pdf_clean.ipynb            Task 2
│   ├── 02_pdf_scanned.ipynb          Task 3
│   └── 03_llm.ipynb                  Task 4
├── results/
│   ├── plain/                        GLiNER predictions on plain text
│   ├── pdf_clean/                    GLiNER predictions on clean PDFs
│   ├── pdf_scanned/                  GLiNER predictions on scanned PDFs (OCR)
│   └── llm/                          prompt, LLM answers and predictions
├── tessdata/                         Tesseract language models (spa, cat)
├── tests/
├── pyproject.toml / uv.lock          dependencies (managed with uv)
└── README.md                         this report
```

### How to run

```powershell
uv sync              # create the virtual environment and install the dependencies
uv run jupyter lab   # or open the notebooks in PyCharm with the .venv interpreter
```

Then run the notebooks in order (`00` → `03`).

**Notes for Windows:**
- The first run downloads the GLiNER model from Hugging Face. If the download freezes,
  set `HF_HUB_DISABLE_XET=1`.
- If the project path contains special characters (such as `º`), copy `tessdata/` to a simple path
  and set the `TESSDATA_PREFIX` environment variable to it before importing `ner.pdf`.