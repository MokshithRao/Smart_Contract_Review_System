import os
os.environ["USE_TF"] = "0"

import json
import faiss
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

# Load environment
load_dotenv()

# Paths
project_root = Path(__file__).resolve().parent.parent
index_file = project_root / "rag" / "faiss_index.index"
clauses_file = project_root / "rag" / "clauses_metadata.json"

# Load FAISS index and metadata
print("Loading FAISS index and metadata...")
index = faiss.read_index(str(index_file))
with open(clauses_file, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Load embedding model
print("Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Setup HuggingFace Inference client
hf_client = InferenceClient(token=os.getenv("HF_API_TOKEN"))
LLM_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"


def retrieve_similar_clauses(clause_text, top_k=5):
    """Retrieve top_k similar clauses from FAISS index."""
    query_embedding = embed_model.encode([clause_text])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(metadata):
            results.append({
                "clause": metadata[idx]["clause"],
                "source": metadata[idx]["source"],
                "label": metadata[idx]["label"],
                "distance": float(dist)
            })
    return results


def rewrite_clause(original_clause, clause_type, similar_clauses):
    """Use LLM to rewrite a clause in a fairer and clearer way."""
    # Format similar clauses for the prompt
    similar_text = ""
    for i, sc in enumerate(similar_clauses, 1):
        similar_text += f"\n{i}. \"{sc['clause']}\"\n   (Label: {', '.join(sc['label'])})\n"

    prompt = f"""You are a legal contract expert. Your task is to rewrite a contract clause to make it fairer, clearer, and more balanced for both parties.

**Original Clause:**
"{original_clause}"

**Clause Type:** {clause_type}

**Similar clauses from a corpus of standard legal contracts (use these as reference for fair language):**
{similar_text}

**Instructions:**
1. Rewrite the clause to be fair to both parties (mutual obligations where possible).
2. Use clear, plain language that non-lawyers can understand.
3. Keep the legal intent intact — do not remove necessary protections.
4. Use the similar clauses above as examples of standard, balanced language.
5. Return ONLY the rewritten clause text, nothing else.

**Rewritten Clause:**"""

    try:
        response = hf_client.chat_completion(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Error generating rewrite: {e}]"


# --- Test the module ---
if __name__ == "__main__":
    test_clause = "The company may terminate this agreement at any time without notice."
    clause_type = "termination for convenience"

    print(f"\nOriginal Clause:\n\"{test_clause}\"\n")
    print(f"Clause Type: {clause_type}\n")

    print("Retrieving similar clauses...")
    similar = retrieve_similar_clauses(test_clause, top_k=5)
    print(f"Found {len(similar)} similar clauses:\n")
    for i, s in enumerate(similar, 1):
        print(f"  {i}. [{', '.join(s['label'])}] {s['clause'][:100]}...")

    print("\nRewriting clause with LLM...")
    rewritten = rewrite_clause(test_clause, clause_type, similar)
    print(f"\nRewritten Clause:\n\"{rewritten}\"")
