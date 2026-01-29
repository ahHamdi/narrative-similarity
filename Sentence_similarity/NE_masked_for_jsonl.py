import json
import spacy
from pathlib import Path

# =========================
# CONFIGURATION
# =========================
INPUT_JSONL = "data/SemEval2026-Task_4-test-v1/test_track_a.jsonl"
OUTPUT_JSONL = "data/SemEval2026-Task_4-test-v1/test_track_a.jsonl_NE_masked.jsonl"
SPACY_MODEL = "en_core_web_sm"
TARGET_ENTS = {"PERSON", "GPE", "LOC", "ORG"}

# =========================
# LOAD SPACY
# =========================
nlp = spacy.load(SPACY_MODEL)


def replace_ner(text: str) -> str:
    """
    Replace named entities by their labels.
    """
    doc = nlp(text)
    new_text = text

    # Remplacement en partant de la fin pour éviter les décalages d'offset
    for ent in sorted(doc.ents, key=lambda e: e.start_char, reverse=True):
        if ent.label_ in TARGET_ENTS:
            new_text = (
                new_text[:ent.start_char]
                + ent.label_
                + new_text[ent.end_char:]
            )

    return new_text


def process_jsonl(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line_id, line in enumerate(fin, 1):
            if not line.strip():
                continue

            data = json.loads(line)

            # Cas 1 : plusieurs champs texte
            for key in ["anchor_text", "text_a", "text_b", "text"]:
                if key in data and isinstance(data[key], str):
                    data[key] = replace_ner(data[key])

            fout.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"✅ Fichier traité : {output_path}")


if __name__ == "__main__":
    process_jsonl(INPUT_JSONL, OUTPUT_JSONL)
