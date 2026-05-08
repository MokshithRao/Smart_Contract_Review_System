"""
Evaluates the correctness and fairness of rewritten contract clauses.
Correctness is measured via semantic similarity between the original and rewritten clauses using sentence-transformers.
Fairness is measured via keyword analysis of harmful vs. balanced terms.
The script updates the 'evaluation/rewrites_for_review.json' file in place.
"""

import json
import logging
from pathlib import Path

# Set up basic logging for progress and errors
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    logging.warning("Dependencies missing. Please install 'sentence-transformers' and 'scikit-learn'.")

# Initialize the embedding model globally
try:
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
except NameError:
    embedding_model = None

# Keyword groups for fairness evaluation
HARMFUL_PHRASES = [
    "sole discretion",
    "without notice",
    "immediate termination",
    "exclusive",
    "not liable",
    "unilateral",
    "at any time",
    "for any reason",
    "without prior written consent",
    "without prior written approval"
]

BALANCED_PHRASES = [
    "either party",
    "mutual",
    "reasonable notice",
    "written notice",
    "both parties",
    "each party",
    "commercially reasonable",
    "prior written notice"
]


def evaluate_correctness(original: str, rewritten: str) -> int:
    """
    Computes semantic similarity to measure correctness.
    
    Args:
        original (str): The original clause text.
        rewritten (str): The rewritten clause text.
        
    Returns:
        int: Correctness score mapped from 1 to 5.
    """
    if not embedding_model:
        return 1

    # Generate embeddings
    embeddings = embedding_model.encode([original, rewritten])
    
    # Compute cosine similarity
    sim_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    
    # Convert similarity score into a 1–5 correctness score
    if sim_score > 0.90:
        return 5
    elif 0.85 <= sim_score <= 0.90:
        return 4
    elif 0.75 <= sim_score < 0.85:
        return 3
    elif 0.65 <= sim_score < 0.75:
        return 2
    else:
        return 1


def evaluate_fairness(original: str, rewritten: str) -> int:
    """
    Calculates fairness by assessing reductions in one-sided contractual language.
    
    Args:
        original (str): The original clause text.
        rewritten (str): The rewritten clause text.
        
    Returns:
        int: Fairness score mapped from 1 to 5 based on keyword occurrences.
    """
    original_lower = original.lower()
    rewritten_lower = rewritten.lower()
    
    # Count harmful phrases in ORIGINAL clause
    harmful_count = sum(phrase in original_lower for phrase in HARMFUL_PHRASES)
    
    # Count balanced phrases in REWRITTEN clause
    balanced_count = sum(phrase in rewritten_lower for phrase in BALANCED_PHRASES)
    
    # Calculate fairness delta
    fairness_delta = balanced_count - harmful_count
    
    # Convert fairness delta into a 1–5 fairness score
    if fairness_delta >= 2:
        return 5
    elif fairness_delta == 1:
        return 4
    elif fairness_delta == 0:
        return 3
    elif fairness_delta == -1:
        return 2
    else:  # <= -2
        return 1


def process_evaluations(filepath: Path) -> None:
    """
    Loads the JSON file, runs the correctness and fairness evaluation, 
    and updates the JSON file while preserving existing data.
    
    Args:
        filepath (Path): Path to the rewrites JSON file.
    """
    if not filepath.exists():
        logging.error(f"File not found: {filepath}")
        return

    # 1. Load rewrites_for_review.json
    try:
        with filepath.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing JSON file {filepath}: {e}")
        return
        
    correctness_scores = []
    fairness_scores = []
    processed_count = 0

    logging.info("Starting evaluation process...")

    for idx, entry in enumerate(data):
        original = entry.get("original", "").strip()
        rewritten = entry.get("rewritten", "").strip()
        
        # Handle missing fields safely: Skip entries with missing original/rewritten text
        if not original or not rewritten:
            logging.info(f"Skipping entry ID {entry.get('id', idx)} due to missing original or rewritten text.")
            continue
            
        # Ensure scores dictionary exists
        if "scores" not in entry or not isinstance(entry["scores"], dict):
            entry["scores"] = {}

        # 2. Evaluate correctness
        c_score = evaluate_correctness(original, rewritten)
        entry["scores"]["correctness"] = c_score
        correctness_scores.append(c_score)
        
        # 3. Evaluate fairness
        f_score = evaluate_fairness(original, rewritten)
        entry["scores"]["fairness"] = f_score
        fairness_scores.append(f_score)
        
        processed_count += 1
        
    # 4 & 5. Update JSON scores and Save updated JSON
    try:
        with filepath.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logging.info("Evaluation complete. Automatically updated JSON file.")
    except IOError as e:
        logging.error(f"Failed to write updated JSON back to disk: {e}")

    # Print summary averages
    if processed_count > 0:
        avg_correctness = sum(correctness_scores) / processed_count
        avg_fairness = sum(fairness_scores) / processed_count
        
        print(f"\nProcessed {processed_count} rewrites")
        print(f"Average correctness: {avg_correctness:.1f}")
        print(f"Average fairness: {avg_fairness:.1f}")
    else:
        print("\nProcessed 0 rewrites. No valid entries were found.")


if __name__ == "__main__":
    target_file = Path(__file__).parent / "rewrites_for_review.json"
    process_evaluations(target_file)
