import json
import re

input_file = "rag/LEDGAR_2016-2019_clean.jsonl"
output_file = "clean_clause_corpus.json"

corpus = []

def clean_text(text):
    text = re.sub(r'\s+', ' ', text) 
    return text.strip()

def is_valid_clause(text):
    words = text.split()
    return 20 <= len(words) <= 300

with open(input_file, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        try:
            data = json.loads(line)
            
            clause = data.get("provision", "")
            source = data.get("source", "")
            label = data.get("label", [])

            clause = clean_text(clause)

            if is_valid_clause(clause):
                corpus.append({
                    "clause": clause,
                    "source": source,
                    "label": label  
                })

        except:
            continue
        if i % 100000 == 0:
            print(f"Processed {i} lines...")

with open(output_file, "w") as f:
    json.dump(corpus, f, indent=2)

print(f"Final clauses: {len(corpus)}")