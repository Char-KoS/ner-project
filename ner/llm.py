from __future__ import annotations

import re
import json
from pathlib import Path
from typing import Any

PROMPT_TEMPLATE = """You are a Named Entity Recognition system for administrative documents written in Spanish or Catalan.
Extract ALL the entities of these types: {labels}.
Rules:
- Copy the entity text EXACTLY as it appears in the document (same characters, accents and punctuation).
- If an entity appears several times in a document, list it once for EACH appearance, in order.
- ADDRESS is the full address as one entity (street, number, floor/door, postal code and city).
- ORGANIZATION is the full name of the public body (e.g. "Servicio de ... de la Región Central").
- AMOUNT includes the currency (e.g. "4.416,13 euros", "1.107,16 EUR", "5.523,29 €").
- Answer ONLY with JSON Lines: one line per document, no explanations, no markdown.
- Format of each line: {{"id": "<id>", "entities": [{{"text": "...", "label": "..."}}]}}

Documents:
{documents}"""



def build_prompt(samples: list[dict[str, Any]], labels: list[str]) -> str: #Junta las instrucciones con los documentos para pegarlos en el chat
    """Create the text to paste into the LLM chat."""

    documents = "\n\n".join(f"[id={s['id']}]\n{s['text']}" for s in samples)
    return PROMPT_TEMPLATE.format(labels=", ".join(labels), documents=documents)


def parse_llm_answer(answer: str) -> list[dict[str, Any]]:
    """Read the documents of the LLM answer.

    Robust to the usual copy/paste problems of chat interfaces:
    - Markdown escapes added when copying (\\[  \\]  \\_  ...)
    - all the documents on one line, or with text / ``` fences around them
    """
    # Remove the Markdown escape backslashes: '\[' -> '[', 'PERSON\_NAME' -> 'PERSON_NAME'
    answer = re.sub(r"\\([\[\]_*`#])", r"\1", answer)

    # Read every JSON object that starts with {"id" ... wherever it is in the text
    decoder = json.JSONDecoder()
    records = []
    position = 0
    while True:
        position = answer.find('{"id"', position)
        if position == -1:
            break
        record, position = decoder.raw_decode(answer, position)   # reads one full object
        records.append(record)
    return records


def load_llm_answer(path: str | Path) -> list[dict[str, Any]]:  #Leen la respuesta del chat (una línea JSON por documento)
    """Read the LLM answer saved in a text file."""

    return parse_llm_answer(Path(path).read_text(encoding="utf-8"))


def add_offsets(text: str, entities: list[dict[str, Any]]) -> list[dict[str, Any]]:  #Busca cada entidad en el texto para calcular start y end
    """The LLM gives no character positions, so find each entity in the text.

       Repeated mentions take the next occurrence after the previous one found.
       Entities not found in the text get start = end = -1 (they will be FP in strict).
       """

    next_search: dict[str, int] = {}
    result = []
    for ent in entities:
        ent_text = ent["text"]
        pos = text.find(ent_text, next_search.get(ent_text, 0))
        if pos == -1:
            pos = text.find(ent_text)
        if pos == -1:
            start, end = -1, -1
        else:
            start, end = pos, pos + len(ent_text)
            next_search[ent_text] = end
        result.append({"start": start, "end": end, "text": ent_text, "label": ent["label"]})
    return result


def llm_records_to_responses(  #Lo deja en el mismo formato que las predicciones de GLiNER
    llm_records: list[dict[str, Any]],
    gold_samples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id = {record["id"]: record for record in llm_records}
    responses = []
    for gold in gold_samples:
        if gold["id"] not in by_id:
            break
        entities = add_offsets(gold["text"], by_id[gold["id"]].get("entities", []))
        responses.append({"id": gold["id"], "text": gold["text"], "entities": entities})
    return responses