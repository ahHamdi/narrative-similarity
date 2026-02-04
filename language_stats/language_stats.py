import pandas as pd
import numpy as np
import re
from collections import Counter
from nltk.tokenize import word_tokenize
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt

nltk.download("punkt")
nltk.download("punkt_tab")

CSV_PATH = "plots_fr_en.autotrain_NE_masked.csv"
ENTITY_LABELS = {"PERSON", "LOC", "GPE", "ORG"}

df = pd.read_csv(CSV_PATH)
assert {"plots_en", "plots_fr_en"}.issubset(df.columns)


def tokenize(text):
    return word_tokenize(str(text))


def count_entities(tokens):
    return Counter(t for t in tokens if t in ENTITY_LABELS)


def corpus_statistics(texts):
    n_plots = len(texts)

    token_counts = []
    entity_counts = []
    entity_per_label = Counter()

    for text in texts:
        tokens = tokenize(text)
        token_counts.append(len(tokens))

        ents = count_entities(tokens)
        entity_counts.append(sum(ents.values()))
        entity_per_label.update(ents)

    return {
        "n_plots": n_plots,
        "avg_tokens_per_plot": np.mean(token_counts),
        "total_entities": sum(entity_counts),
        "avg_entities_per_plot": np.mean(entity_counts),
        "entities_by_label": dict(entity_per_label),
    }


stats_en = corpus_statistics(df["plots_en"])
stats_pt_en = corpus_statistics(df["plots_fr_en"])


def print_stats(title, stats):
    print(f"\n {title}")
    print("-" * 40)
    print(f"Number of plots: {stats['n_plots']}")
    print(f"Avg tokens / plot: {stats['avg_tokens_per_plot']:.2f}")
    print(f"Total NEs: {stats['total_entities']}")
    print(f"Avg NEs / plot: {stats['avg_entities_per_plot']:.2f}")
    print("Entities by label:")
    for k, v in stats["entities_by_label"].items():
        print(f"  {k}: {v}")


print_stats("English original plots", stats_en)
print_stats("French-English translated plots", stats_pt_en)


def clean_for_wordcloud(text):
    tokens = tokenize(text)
    tokens = [
        t.lower() for t in tokens if t.isalpha() and t.upper() not in ENTITY_LABELS
    ]
    return " ".join(tokens)


def generate_wordcloud(texts, title):
    full_text = " ".join(clean_for_wordcloud(t) for t in texts)

    wc = WordCloud(
        width=1200, height=600, background_color="white", max_words=200
    ).generate(full_text)

    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.show()


generate_wordcloud(df["plots_en"], "English Original Plots")
generate_wordcloud(df["plots_fr_en"], "French-English Translated Plots")
