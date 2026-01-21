import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import json
import random

# -----------------------------
# Paramètres
# -----------------------------
#MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
#MODEL_NAME = "sentence-transformers/all-MiniLM-L12-v2"
#MODEL_NAME = "Jgmorenof/movie-plots-pt-en-all-MiniLM-L6-v2"
#MODEL_NAME = "Jgmorenof/movie-plots-pt-en-maskedentities-all-MiniLM-L6-v2"
#MODEL_NAME = "ahmedHamdi/movie-plots-pt-en-maskedentities-all-MiniLM-L12-v2"
MODEL_NAME = "ahmedHamdi/text_similarity_movie_plot_en_autotrain_all-MiniLM-L12-v2_masked_NEs"
#MODEL_NAME = "ahmedHamdi/story-similarity-MiniLM-L12"
#MODEL_NAME = "ahmedHamdi/story-similarity-MiniLM-L12-NE-Masked"
#MODEL_NAME = "ahmedHamdi/story-similarity-V2-MiniLM-L12"
#MODEL_NAME = "ahmedHamdi/story-similarity-V2-MiniLM-L12-NE-Masked"
INPUT_FILE = "dev_track_a.jsonl"
OUTPUT_FILE = "output/track_a_embeddings_en_pt_plots.jsonl"
baseline = "embeddings"  # ou "random"

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# -----------------------------
# Charger le modèle
# -----------------------------
model = SentenceTransformer(MODEL_NAME)

# -----------------------------
# Charger le dataset
# -----------------------------
df = pd.read_json(INPUT_FILE, lines=True)

# -----------------------------
# Fonction de prédiction
# -----------------------------
def embeddings_predict(row):
    anchor = row["anchor_text"]
    text_a = row["text_a"]
    text_b = row["text_b"]

    # calculer les embeddings
    emb_anchor = model.encode(anchor, convert_to_numpy=True)
    emb_a = model.encode(text_a, convert_to_numpy=True)
    emb_b = model.encode(text_b, convert_to_numpy=True)

    # calculer similarité cosinus
    sim_a = cosine_similarity([emb_anchor], [emb_a])[0][0]
    sim_b = cosine_similarity([emb_anchor], [emb_b])[0][0]

    return sim_a > sim_b  # True si A est plus proche, False sinon

# -----------------------------
# Faire les prédictions
# -----------------------------
if baseline == "embeddings":
    df["predicted_text_a_is_closer"] = df.apply(embeddings_predict, axis=1)
else:
    df["predicted_text_a_is_closer"] = df.apply(lambda _: random.choice([True, False]), axis=1)

# -----------------------------
# Évaluer et sauvegarder
# -----------------------------
if "text_a_is_closer" in df.columns:
    accuracy = (df["predicted_text_a_is_closer"] == df["text_a_is_closer"]).mean()
    print(f"Accuracy: {accuracy:.3f}")

# Réécrire le champ pour correspondre au format attendu
df["text_a_is_closer"] = df["predicted_text_a_is_closer"]
del df["predicted_text_a_is_closer"]

# Sauvegarder en JSONL
df.to_json(OUTPUT_FILE, orient="records", lines=True)
print(f"Prédictions sauvegardées dans {OUTPUT_FILE}")
