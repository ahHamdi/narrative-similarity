import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator
import torch

# =====================
# Load TEST dataset
# =====================
df = pd.read_csv("data/IR_testset.csv")

queries = {}
corpus = {}
relevant_docs = {}

doc_id = 0

for qid, row in df.iterrows():
    qid = str(qid)
    queries[qid] = row["query"]

    relevant_docs[qid] = set()

    for col in ["doc_en", "doc_fr", "doc_pt", "doc_es"]:
        corpus_id = str(doc_id)
        corpus[corpus_id] = row[col]
        relevant_docs[qid].add(corpus_id)
        doc_id += 1

print(f"Queries : {len(queries)}")
print(f"Docs : {len(corpus)}")
print(f"Relevant docs by query : 4")

models = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-MiniLM-L12-v2",
    "sentence-transformers/all-mpnet-base-v2"
]

device = "cuda" if torch.cuda.is_available() else "cpu"

for model_name in models:
    print(f"\nEvaluation model : {model_name}")
    
    model = SentenceTransformer(model_name, device=device)

    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="plots-ir-multidoc"
    )

    metrics = evaluator(model)
    print(metrics)
