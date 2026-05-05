import os
os.environ["USE_TF"] = "0"

import json
import faiss
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

load_dotenv()

# paths
ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = ROOT / "rag" / "faiss_index.index"
META_PATH  = ROOT / "rag" / "clauses_metadata.json"

# load faiss index + metadata
print("Loading FAISS index and metadata...")
faiss_index = faiss.read_index(str(INDEX_PATH))
with open(META_PATH, "r", encoding="utf-8") as f:
    clause_metadata = json.load(f)

# embedding model (same one used during indexing)
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# llm setup
llm = InferenceClient(token=os.getenv("HF_API_TOKEN"))
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"


def get_similar_clauses(clause_text, k=5):
    """Find k most similar clauses from the FAISS index"""
    vec = embedder.encode([clause_text])
    vec = np.array(vec).astype("float32")

    dists, idxs = faiss_index.search(vec, k)

    hits = []
    for d, i in zip(dists[0], idxs[0]):
        if i < len(clause_metadata):
            hits.append({
                "clause":  clause_metadata[i]["clause"],
                "source":  clause_metadata[i]["source"],
                "label":   clause_metadata[i]["label"],
                "distance": float(d)
            })
    return hits


def call_llm(prompt):
    """Send a prompt to the LLM and return the response text"""
    resp = llm.chat_completion(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512
    )
    return resp.choices[0].message.content.strip()


def rewrite_clause(original, clause_type, similar_clauses):
    """Rewrite a contract clause using RAG context from similar clauses"""

    # format the similar clauses as numbered examples
    examples = ""
    for i, sc in enumerate(similar_clauses, 1):
        labels = ", ".join(sc["label"])
        examples += f'\n{i}. "{sc["clause"]}"\n   (Label: {labels})\n'

    prompt = f"""You are a legal contract expert. Rewrite the following contract clause to make it fairer, clearer, and more balanced for both parties.

Original Clause:
"{original}"

Clause Type: {clause_type}

Similar clauses from standard legal contracts (use as reference for fair language):
{examples}

Instructions:
1. Rewrite the clause to be fair to both parties (mutual obligations where possible)
2. Use clear, plain language that non-lawyers can understand
3. Keep the legal intent intact
4. Use the similar clauses above as examples of balanced language
5. Return ONLY the rewritten clause text, nothing else

Rewritten Clause:"""

    try:
        return call_llm(prompt)
    except Exception as e:
        return f"[Rewrite failed: {e}]"


def explain_changes(original, rewritten, clause_type):
    """Compare original vs rewritten clause and explain what changed"""

    prompt = f"""You are a legal contract expert. Compare the two clauses below and explain what changed and why.

Clause Type: {clause_type}

Original Clause:
"{original}"

Rewritten Clause:
"{rewritten}"

Instructions:
1. List each change as a bullet point
2. For each change, explain why it makes the clause fairer or clearer
3. Highlight if one-sided terms were made mutual
4. Note if missing protections (like notice periods) were added
5. Keep it concise and easy for non-lawyers to understand

Changes:"""

    try:
        return call_llm(prompt)
    except Exception as e:
        return f"[Explanation failed: {e}]"


if __name__ == "__main__":
    # test clause
    test_clause = "The company may terminate this agreement at any time without notice."
    test_type = "termination for convenience"

    print(f'\nOriginal: "{test_clause}"')
    print(f"Type: {test_type}\n")

    # retrieve
    print("Retrieving similar clauses...")
    similar = get_similar_clauses(test_clause, k=5)
    for i, s in enumerate(similar, 1):
        print(f"  {i}. [{', '.join(s['label'])}] {s['clause'][:100]}...")

    # rewrite
    print("\nRewriting...")
    rewritten = rewrite_clause(test_clause, test_type, similar)
    print(f'\nRewritten: "{rewritten}"')

    # explain
    print("\nExplaining changes...")
    explanation = explain_changes(test_clause, rewritten, test_type)
    print(f"\n{explanation}")
