import pandas as pd
import spacy

nlp = spacy.load("en_core_web_sm")

ENTITY_LABELS = {"PERSON", "GPE", "LOC", "ORG"}


def replace_entities(text):
    if not isinstance(text, str):
        return text

    doc = nlp(text)
    new_text = text
    for ent in sorted(doc.ents, key=lambda e: e.start_char, reverse=True):
        if ent.label_ in ENTITY_LABELS:
            new_text = (
                new_text[: ent.start_char] + ent.label_ + new_text[ent.end_char :]
            )
    return new_text


def process_csv(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    for col in df.columns:
        df[col] = df[col].apply(replace_entities)

    df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    process_csv(
        input_csv="plots_es_en.autotrain.csv",
        output_csv="plots_es_en.autotrain-NE-masked.csv",
    )
