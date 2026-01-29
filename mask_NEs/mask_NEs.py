import json
import spacy
from pathlib import Path

INPUT_JSONL = "data/test_track_a.jsonl"
OUTPUT_JSONL = "data/test_track_a_NE_masked.jsonl"
SPACY_MODEL = "en_core_web_sm"
TARGET_ENTS = {"PERSON", "GPE", "LOC", "ORG"}

TEXT_FIELDS = {"anchor_text", "text_a", "text_b", "text"}

nlp = spacy.load(SPACY_MODEL)


def replace_ner(text: str) -> str:
    """
    Replace named entities by their entity label.
    """
    if not isinstance(text, str) or not text.strip():
        return text

    doc = nlp(text)
    masked_text = text

    for ent in sorted(doc.ents, key=lambda e: e.start_char, reverse=True):
        if ent.label_ in TARGET_ENTS:
            masked_text = (
                masked_text[:ent.start_char]
                + ent.label_
                + masked_text[ent.end_char:]
            )

    return masked_text


def process_jsonl(input_path: str, output_path: str):
    input_path = Path(input_path)
    output_path = Path(output_path)

    with input_path.open("r", encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout:

        for line_id, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Line {line_id}: invalid JSON ({e})")
                continue

            # Mask all known text fields if present
            for field in TEXT_FIELDS:
                if field in data:
                    data[field] = replace_ner(data[field])

            fout.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"NER masking finished : {output_path}")


if __name__ == "__main__":
    process_jsonl(INPUT_JSONL, OUTPUT_JSONL)
