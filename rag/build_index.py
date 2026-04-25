import os
os.environ["USE_TF"] = "0"

import faiss
import json
import random
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

# Paths
project_root = Path(__file__).resolve().parent.parent
input_file = project_root / "clean_clause_corpus.json"
index_file = project_root / "rag" / "faiss_index.index"
clauses_file = project_root / "rag" / "clauses_metadata.json"

# Load model
print("Loading embedding model...", flush=True)
model = SentenceTransformer("all-MiniLM-L6-v2")

# FAISS setup
dimension = 384
index = faiss.IndexFlatL2(dimension)

# Load corpus
print("Loading corpus...", flush=True)
with open(input_file, "r", encoding="utf-8") as f:
    corpus = json.load(f)

print(f"Full corpus: {len(corpus)} clauses", flush=True)

# Sample 50K clauses for CPU-friendly indexing
SUBSET_SIZE = 50000
random.seed(42)
if len(corpus) > SUBSET_SIZE:
    corpus = random.sample(corpus, SUBSET_SIZE)
    print(f"Sampled {SUBSET_SIZE} clauses for indexing", flush=True)

batch_size = 512
batch_texts = []
metadata = []

print("Starting embedding + indexing...\n", flush=True)

for i, entry in enumerate(corpus):
    clause = entry["clause"]
    batch_texts.append(clause)
    metadata.append({
        "clause": clause,
        "source": entry.get("source", ""),
        "label": entry.get("label", [])
    })

    if len(batch_texts) == batch_size:
        embeddings = model.encode(batch_texts, show_progress_bar=False)
        embeddings = np.array(embeddings).astype("float32")
        index.add(embeddings)
        batch_texts = []

        processed = i + 1
        print(f"Processed {processed}/{len(corpus)} clauses", flush=True)

# Handle remaining batch
if batch_texts:
    embeddings = model.encode(batch_texts, show_progress_bar=False)
    embeddings = np.array(embeddings).astype("float32")
    index.add(embeddings)

print(f"\nTotal indexed: {index.ntotal}", flush=True)

# Save FAISS index
faiss.write_index(index, str(index_file))

# Save metadata (needed for retrieval)
with open(clauses_file, "w", encoding="utf-8") as f:
    json.dump(metadata, f)

print(f"Saved FAISS index to {index_file}", flush=True)
print(f"Saved clause metadata to {clauses_file}", flush=True)