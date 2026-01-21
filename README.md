# Narrative Similarity

![Python](https://img.shields.io/badge/Python-3.9%2B%20(tested%203.12)-blue)


## Track A – Embedding-Based Similarity Baseline

This script runs an embedding-based similarity baseline for **Track A** using
[SentenceTransformers](https://www.sbert.net/).

Given an `anchor_text` and two candidate texts (`text_a`, `text_b`), the script
predicts which candidate is closer to the anchor using cosine similarity of
sentence embeddings.

The output is written as **JSONL** and matches the expected evaluation format.

### Evaluation

```bash
python run_track_a.py \
  --model story_v2_l12_masked
```

To add a model, modify `MODEL_CHOICES` in `run_track_a.py`.