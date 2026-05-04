import os
os.environ["USE_TF"] = "0"

import json
import time
from pathlib import Path
from rag.rewrite import get_similar_clauses, rewrite_clause, explain_changes

OUTPUT = Path(__file__).resolve().parent.parent / "evaluation" / "rewrites_for_review.json"

# 20 test clauses across different types
test_clauses = [
    ("The company may terminate this agreement at any time without notice.",
     "termination for convenience"),

    ("The employee shall not work for any competing business for a period of 5 years after termination.",
     "non-compete"),

    ("The company's total liability shall not exceed the fees paid in the last month.",
     "cap on liability"),

    ("This agreement shall be governed exclusively by the laws of the State of Delaware, and any disputes must be resolved in Delaware courts only.",
     "governing law"),

    ("The licensee is granted a non-exclusive, non-transferable license to use the software.",
     "license grant"),

    ("The employee shall not solicit any customers of the company for a period of 3 years after leaving.",
     "no-solicit of customers"),

    ("The company shall have the right to audit the contractor's records at any time without prior notice.",
     "audit rights"),

    ("This agreement shall automatically renew for successive one-year terms unless the company provides notice of termination.",
     "renewal term"),

    ("The contractor shall maintain insurance coverage as determined solely by the company.",
     "insurance"),

    ("Neither party may assign this agreement without the prior written consent of the company.",
     "anti-assignment"),

    ("The employee agrees not to make any negative statements about the company at any time.",
     "non-disparagement"),

    ("All intellectual property created during the term of this agreement shall belong exclusively to the company.",
     "ip ownership assignment"),

    ("The company grants the licensee an irrevocable, perpetual license to use the technology.",
     "irrevocable or perpetual license"),

    ("The effective date of this agreement shall be the date of the last signature.",
     "effective date"),

    ("This agreement shall expire on December 31, 2025, with no option for renewal.",
     "expiration date"),

    ("The company may change the pricing at any time without notice to the other party.",
     "price restrictions"),

    ("The employee shall not solicit or hire any employees of the company for 2 years after termination.",
     "no-solicit of employees"),

    ("The service provider shall continue to provide transition services for 6 months after termination at no additional cost.",
     "post-termination services"),

    ("The company shall share 5% of net revenue with the partner, calculated solely by the company.",
     "revenue/profit sharing"),

    ("The vendor shall not sell more than 1000 units per quarter without prior written approval from the company.",
     "volume restriction"),
]

results = []

for i, (clause, ctype) in enumerate(test_clauses, 1):
    print(f"\n[{i}/20] Processing: {ctype}", flush=True)

    similar = get_similar_clauses(clause, k=5)
    rewritten = rewrite_clause(clause, ctype, similar)

    # small delay to avoid rate limits
    time.sleep(2)

    explanation = explain_changes(clause, rewritten, ctype)
    time.sleep(2)

    results.append({
        "id": i,
        "clause_type": ctype,
        "original": clause,
        "rewritten": rewritten,
        "explanation": explanation,
        "similar_clauses_used": [s["clause"][:150] for s in similar],
        "scores": {
            "correctness": None,
            "fairness": None,
            "clarity": None
        }
    })

    print(f"  Done.", flush=True)

# save results
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nSaved {len(results)} rewrites to {OUTPUT}")
print("Fill in the scores (1-5) for correctness, fairness, and clarity as a team.")
