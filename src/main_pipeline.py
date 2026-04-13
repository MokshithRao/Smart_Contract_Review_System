"""Main pipeline for end-to-end contract clause risk analysis."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from inference.clause_classifier import predict_batch, split_into_clauses
from risk_filter import filter_important_clauses


logger = logging.getLogger(__name__)


def process_contract(text: str) -> List[Dict[str, Any]]:
	"""Run the full contract analysis pipeline and return important clauses.

	Steps:
	1) Split contract text into clauses.
	2) Classify all clauses using batch inference.
	3) Filter to high/medium risk clauses.
	"""

	if not text or not text.strip():
		logger.warning("Received empty contract text. Returning empty result.")
		return []

	clauses = split_into_clauses(text)
	if not clauses:
		logger.warning("No clauses were produced from input text. Returning empty result.")
		return []

	logger.info("Processing contract with %d clause(s).", len(clauses))
	predictions = predict_batch(clauses)
	if not predictions:
		logger.warning("Classifier returned no predictions. Returning empty result.")
		return []

	important_clauses = filter_important_clauses(predictions)
	logger.info("Pipeline identified %d important clause(s).", len(important_clauses))
	return important_clauses


if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

	sample_contract_text = (
		"Either party may terminate this agreement for convenience with thirty days notice. "
		"Vendor shall maintain cyber liability insurance at all times. "
		"This agreement is governed by the laws of California.\n"
		"The customer has audit rights on reasonable notice."
	)

	results = process_contract(sample_contract_text)
	print("Important Clauses:")
	print(json.dumps(results, indent=2))
