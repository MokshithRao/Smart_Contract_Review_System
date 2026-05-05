"""Run contract inference and export risky-clause predictions as JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from extraction.Text_extractor import extract_text
from main_pipeline import process_contract, process_contract_one_sided_high_risk


def main() -> None:
	parser = argparse.ArgumentParser(description="Predict risky clauses from a contract.")
	parser.add_argument("--input", required=True, help="Path to input contract file (.pdf or .docx)")
	parser.add_argument(
		"--output",
		default="data/processed/contract_predictions.json",
		help="Path to output JSON file",
	)
	args = parser.parse_args()

	input_path = Path(args.input)
	if not input_path.exists():
		raise FileNotFoundError(f"Input file not found: {input_path}")

	text = extract_text(str(input_path))
	if not text or text == "Unsupported file format":
		raise ValueError("Could not extract text. Use a PDF or DOCX file.")

	important_clauses_grouped = process_contract(text)
	one_sided_high_risk = process_contract_one_sided_high_risk(text)

	payload = {
		"input_file": str(input_path),
		"text_chars": len(text),
		"important_clause_count": sum(len(items) for items in important_clauses_grouped.values()),
		"important_clauses_grouped": important_clauses_grouped,
		"one_sided_high_risk_count": len(one_sided_high_risk),
		"one_sided_high_risk_clauses": one_sided_high_risk,
	}

	output_path = Path(args.output)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

	print(f"Input file: {input_path}")
	print(f"Text chars: {len(text)}")
	print(f"Important clauses: {payload['important_clause_count']}")
	print(f"One-sided HIGH-risk clauses: {payload['one_sided_high_risk_count']}")
	print(f"Saved JSON: {output_path}")


if __name__ == "__main__":
	main()
