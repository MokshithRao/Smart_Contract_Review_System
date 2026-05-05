import os
os.environ["USE_TF"] = "0"

import faiss
import json
import random
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

# paths
ROOT = Path(__file__).resolve().parent.parent
CORPUS_FILE = ROOT / "clean_clause_corpus.json"
INDEX_OUT   = ROOT / "rag" / "faiss_index.index"
META_OUT    = ROOT / "rag" / "clauses_metadata.json"

SUBSET_SIZE = 50000   # sample size (full corpus is too slow on CPU)
BATCH_SIZE  = 512
EMBED_DIM   = 384     # output dim of all-MiniLM-L6-v2

# load embedding model
print("Loading embedding model...", flush=True)
model = SentenceTransformer("all-MiniLM-L6-v2")

# faiss index (flat L2 = brute force, fine for 50k vectors)
index = faiss.IndexFlatL2(EMBED_DIM)

# load the cleaned clause corpus
print("Loading corpus...", flush=True)
with open(CORPUS_FILE, "r", encoding="utf-8") as f:
    corpus = json.load(f)
print(f"Full corpus size: {len(corpus)}", flush=True)

# take a random subset so this runs in ~20 min on CPU
random.seed(42)
if len(corpus) > SUBSET_SIZE:
    corpus = random.sample(corpus, SUBSET_SIZE)
    print(f"Using random sample of {SUBSET_SIZE} clauses", flush=True)

# embed and index in batches
batch = []
metadata = []

print("Embedding and indexing...\n", flush=True)

for i, entry in enumerate(corpus):
    batch.append(entry["clause"])
    metadata.append({
        "clause": entry["clause"],
        "source": entry.get("source", ""),
        "label":  entry.get("label", [])
    })

    if len(batch) == BATCH_SIZE:
        vecs = model.encode(batch, show_progress_bar=False)
        vecs = np.array(vecs).astype("float32")
        index.add(vecs)
        batch = []
        print(f"  {i+1}/{len(corpus)} done", flush=True)

# leftover batch
if batch:
    vecs = model.encode(batch, show_progress_bar=False)
    vecs = np.array(vecs).astype("float32")
    index.add(vecs)

print(f"\nTotal vectors indexed: {index.ntotal}", flush=True)

# save everything
faiss.write_index(index, str(INDEX_OUT))
with open(META_OUT, "w", encoding="utf-8") as f:
    json.dump(metadata, f)

print(f"Index saved to {INDEX_OUT}", flush=True)
print(f"Metadata saved to {META_OUT}", flush=True)
