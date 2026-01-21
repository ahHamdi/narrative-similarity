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
                new_text[:ent.start_char]
                + ent.label_
                + new_text[ent.end_char:]
            )
    return new_text


def process_csv(input_csv, output_csv, text_columns):
    df = pd.read_csv(input_csv)

    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].apply(replace_entities)

    df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    process_csv(
        input_csv="large_resource_for_finetuning.csv",
        output_csv="large_resource_for_finetuning_NE_masked.csv",
        text_columns=["text"]
    )
