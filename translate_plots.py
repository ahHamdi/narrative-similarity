import os
import csv
import pandas as pd
import torch
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm

INPUT_CSV = "films_plots_ready_for_translation.csv"
OUTPUT_CSV = "films_plots_translated.csv"

TEXT_COL = "plot_other"
LANG_COL = "lang"

BATCH_SIZE = 4
MAX_LENGTH = 512
SAVE_EVERY = 25

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_MAP = {
    "fr": "Helsinki-NLP/opus-mt-fr-en",
    "es": "Helsinki-NLP/opus-mt-es-en",
    "pt": "Helsinki-NLP/opus-mt-ROMANCE-en",
    "de": "Helsinki-NLP/opus-mt-de-en",
    "it": "Helsinki-NLP/opus-mt-it-en",
}


def load_csv_safe(path):
    """Lecture CSV tolérante + réparation"""
    try:
        return pd.read_csv(path)
    except Exception:
        print("CSV cassé — tentative de réparation...")
        return pd.read_csv(path, engine="python", on_bad_lines="skip")


def save_csv_safe(df, path):
    df.to_csv(path, index=False, quoting=csv.QUOTE_ALL)


def load_existing_output():
    if os.path.exists(OUTPUT_CSV):
        print("Reprise depuis un fichier existant")
        return load_csv_safe(OUTPUT_CSV)
    return None


def translate_batch(texts, tokenizer, model):
    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(**inputs)

    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def main():
    print(f"Device : {DEVICE}")

    df = load_csv_safe(INPUT_CSV)

    df = df.dropna(subset=["film", "plot_en", TEXT_COL, LANG_COL])
    df = df.reset_index(drop=True)

    df_done = load_existing_output()
    done_keys = set()

    if df_done is not None:
        df_done["key"] = df_done["film"] + "||" + df_done[LANG_COL]
        done_keys = set(df_done["key"])

    if "plot_translated" not in df.columns:
        df["plot_translated"] = ""

    for lang, model_name in MODEL_MAP.items():
        print(f"\nTraduction {lang} → EN")

        mask = df[LANG_COL] == lang
        if not mask.any():
            continue

        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name).to(DEVICE)
        model.eval()

        lang_df = df[mask]

        texts = []
        indices = []

        for idx, row in tqdm(lang_df.iterrows(), total=len(lang_df)):
            key = f"{row['film']}||{row[LANG_COL]}"
            if key in done_keys:
                continue

            texts.append(row[TEXT_COL])
            indices.append(idx)

            if len(texts) == BATCH_SIZE:
                translations = translate_batch(texts, tokenizer, model)
                for i, t in zip(indices, translations):
                    df.at[i, "plot_translated"] = t

                texts, indices = [], []

        if texts:
            translations = translate_batch(texts, tokenizer, model)
            for i, t in zip(indices, translations):
                df.at[i, "plot_translated"] = t

        save_csv_safe(df, OUTPUT_CSV)
        print("Checkpoint sauvegardé")

        del model
        torch.cuda.empty_cache()

    save_csv_safe(df, OUTPUT_CSV)
    print("\nTraduction terminée")
    print(f"Fichier : {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
