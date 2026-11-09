"""GLiNER model loading and inference helpers."""
from __future__ import annotations

import re
from typing import Any

DEFAULT_MODEL = "gliner-community/gliner_small-v2.5"

# Maximum characters per chunk (~1000 chars stays below GLiNER's 384-token limit)
MAX_CHUNK_CHARS = 1000

# Regular expression: end of sentence = '.' or ';' followed by whitespace
_SENTENCE_END = re.compile(r"(?<=[.;])\s+")


def load_gliner_model(model_name: str = DEFAULT_MODEL, device: str | None = None):
    """Load a GLiNER model from Hugging Face and move it to GPU or CPU."""
    import torch
    from gliner import GLiNER

    # If no device is given, use the GPU when it exists, otherwise the CPU
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GLiNER.from_pretrained(model_name)
    return model.to(device)


def split_into_chunks(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[tuple[int, str]]:
    """Split the text into (offset, chunk) pairs, cutting only between sentences.

    offset = position of the first character of the chunk in the full text.
    """
    chunks: list[tuple[int, str]] = []
    chunk_start = 0   # where the current chunk begins
    chunk_end = 0     # where the last complete sentence of the chunk ends

    # Positions where each sentence ends (+ the end of the text)
    boundaries = [m.end() for m in _SENTENCE_END.finditer(text)] + [len(text)]

    for boundary in boundaries:
        # If adding this sentence makes the chunk too long, close the current chunk first
        if boundary - chunk_start > max_chars and chunk_end > chunk_start:
            chunks.append((chunk_start, text[chunk_start:chunk_end]))
            chunk_start = chunk_end
        chunk_end = boundary

    # Last chunk
    if chunk_end > chunk_start:
        chunks.append((chunk_start, text[chunk_start:chunk_end]))

    return chunks


def predict_entities(
    model,
    text: str,
    labels: list[str],
    threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Run GLiNER over a full document, chunk by chunk, with positions in the full text."""
    entities: list[dict[str, Any]] = []

    for offset, chunk in split_into_chunks(text):
        for ent in model.predict_entities(chunk, labels, threshold=threshold):
            # Move the positions from the chunk to the full document
            start = offset + ent["start"]
            end = offset + ent["end"]
            entities.append({
                "start": start,
                "end": end,
                "text": text[start:end],
                "label": ent["label"],
                "score": float(ent["score"]),
            })

    # Sort the entities by their position in the text
    return sorted(entities, key=lambda e: e["start"])


def predict_dataset(
    model,
    samples: list[dict[str, Any]],
    labels: list[str],
    threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Predict every document and keep the same id / text / language as the input."""
    from tqdm.auto import tqdm   # progress bar

    responses = []
    for sample in tqdm(samples, desc="GLiNER"):
        responses.append({
            "id": sample.get("id"),
            "text": sample["text"],
            "entities": predict_entities(model, sample["text"], labels, threshold),
            "language": sample.get("language"),
        })
    return responses

# ---------------------------------------------------------------------------
# Post-processing: merge adjacent pieces of the same entity
# ---------------------------------------------------------------------------

# Text allowed BETWEEN two pieces of the same entity so that they are merged
MERGE_GAPS = {
    # Address pieces: commas, numbers, short letters, floor/door (e.g. ", 102, 2-B, ")
    "ADDRESS": re.compile(r"^,?[\s\dA-Za-zºª\-/,]{0,20}$"),
    # Organization pieces joined by a preposition (e.g. " de la ")
    "ORGANIZATION": re.compile(r"^\s+(de|del|de la|de los|de las)\s+$"),
}


def merge_adjacent_entities(text: str, entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join consecutive entities with the same label when the gap between them is allowed."""
    entities = sorted(entities, key=lambda e: e["start"])
    merged: list[dict[str, Any]] = []

    for entity in entities:
        if merged:
            previous = merged[-1]
            gap = text[previous["end"]:entity["start"]]          # text between the two pieces
            rule = MERGE_GAPS.get(entity["label"])
            if previous["label"] == entity["label"] and rule and rule.match(gap):
                # Extend the previous entity until the end of this one
                previous["end"] = entity["end"]
                previous["text"] = text[previous["start"]:previous["end"]]
                previous["score"] = min(previous["score"], entity["score"])
                continue
        merged.append(dict(entity))          # copy, so the original is not modified

    return merged


def merge_dataset(responses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply merge_adjacent_entities to every document of a predictions list."""
    return [
        {**doc, "entities": merge_adjacent_entities(doc["text"], doc["entities"])}
        for doc in responses
    ]