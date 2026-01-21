#!/usr/bin/env python3
import argparse
import os
import random

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Model choices (short keys -> HF model names)
# -----------------------------
MODEL_CHOICES = {
    "minilm6": "sentence-transformers/all-MiniLM-L6-v2",
    "minilm12": "sentence-transformers/all-MiniLM-L12-v2",
    "movie_plots": "Jgmorenof/movie-plots-pt-en-all-MiniLM-L6-v2",
    "movie_plots_masked": "Jgmorenof/movie-plots-pt-en-maskedentities-all-MiniLM-L6-v2",
    "ahmed_masked_l12": "ahmedHamdi/movie-plots-pt-en-maskedentities-all-MiniLM-L12-v2",
    "ahmed_autotrain_masked": "ahmedHamdi/text_similarity_movie_plot_en_autotrain_all-MiniLM-L12-v2_masked_NEs",
    "story_l12": "ahmedHamdi/story-similarity-MiniLM-L12",
    "story_l12_masked": "ahmedHamdi/story-similarity-MiniLM-L12-NE-Masked",
    "story_v2_l12": "ahmedHamdi/story-similarity-V2-MiniLM-L12",
    "story_v2_l12_masked": "ahmedHamdi/story-similarity-V2-MiniLM-L12-NE-Masked",
}


def embeddings_predict(row, model: SentenceTransformer) -> bool:
    anchor = row["anchor_text"]
    text_a = row["text_a"]
    text_b = row["text_b"]

    emb_anchor = model.encode(anchor, convert_to_numpy=True)
    emb_a = model.encode(text_a, convert_to_numpy=True)
    emb_b = model.encode(text_b, convert_to_numpy=True)

    sim_a = cosine_similarity([emb_anchor], [emb_a])[0][0]
    sim_b = cosine_similarity([emb_anchor], [emb_b])[0][0]

    return sim_a > sim_b


if __name__ == "__main__":

    p = argparse.ArgumentParser(
        description="Track A: embedding-based similarity baseline"
    )

    p.add_argument(
        "--input",
        default="dev_track_a.jsonl",
        help="Input JSONL file",
    )

    p.add_argument(
        "--output-name",
        default="track_a_predictions.jsonl",
        help="Output filename (saved under data/output/)",
    )

    p.add_argument(
        "--model",
        choices=MODEL_CHOICES.keys(),
        default="ahmed_autotrain_masked",
        help="Model key to use",
    )

    p.add_argument(
        "--baseline",
        choices=["embeddings", "random"],
        default="embeddings",
        help="Baseline type",
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    model_name = MODEL_CHOICES[args.model]
    print(f"Using model: {model_name}")
    print(f"Baseline: {args.baseline}")

    output_dir = "data/output"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, args.output_name)

    # -----------------------------
    # Load model ONCE
    # -----------------------------
    model = SentenceTransformer(model_name)

    # -----------------------------
    # Load data
    # -----------------------------
    df = pd.read_json(args.input, lines=True)

    # -----------------------------
    # Predict
    # -----------------------------
    if args.baseline == "embeddings":
        df["predicted_text_a_is_closer"] = df.apply(
            lambda r: embeddings_predict(r, model),
            axis=1,
        )
    else:
        df["predicted_text_a_is_closer"] = [
            random.choice([True, False]) for _ in range(len(df))
        ]

    # -----------------------------
    # Evaluate (if gold present)
    # -----------------------------
    if "text_a_is_closer" in df.columns:
        accuracy = (df["predicted_text_a_is_closer"] == df["text_a_is_closer"]).mean()
        print(f"Accuracy {model_name}: {accuracy:.3f}")

    # -----------------------------
    # Write output
    # -----------------------------
    df["text_a_is_closer"] = df["predicted_text_a_is_closer"]
    df.drop(columns=["predicted_text_a_is_closer"], inplace=True)

    df.to_json(output_path, orient="records", lines=True)
    print(f"Predictions saved to {output_path}")
