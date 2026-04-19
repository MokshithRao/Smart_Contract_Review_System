"""Main pipeline for end-to-end contract clause risk analysis."""

from __future__ import annotations

from collections import defaultdict
import json
import logging
from typing import Any, Dict, List

from inference.clause_classifier import predict_batch, split_into_clauses
from risk_filter import filter_important_clauses, filter_one_sided_high_risk_clauses


logger = logging.getLogger(__name__)


def group_by_label(clauses: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
	"""Group important clauses by predicted label for UI display."""

	grouped: defaultdict[str, List[Dict[str, Any]]] = defaultdict(list)
	for clause in clauses:
		label = str(clause.get("label", "unknown") or "unknown")
		grouped[label].append(clause)
	return dict(grouped)


def keep_top_k_per_label(grouped: Dict[str, List[Dict[str, Any]]], k: int = 2) -> Dict[str, List[Dict[str, Any]]]:
	"""Keep only the top-k clauses per label by confidence."""

	final: Dict[str, List[Dict[str, Any]]] = {}
	for label, items in grouped.items():
		sorted_items = sorted(items, key=lambda item: item.get("confidence", 0.0), reverse=True)
		final[label] = sorted_items[:k]
	return final


def process_contract(text: str) -> Dict[str, List[Dict[str, Any]]]:
	"""Run the full contract analysis pipeline and return important clauses.

	Steps:
	1) Split contract text into clauses.
	2) Merge short fragments into nearby clauses.
	2) Classify all clauses using batch inference.
	3) Filter to high/medium risk clauses.
	4) Group by label for UI consumption.
	"""

	if not text or not text.strip():
		logger.warning("Received empty contract text. Returning empty result.")
		return {}

	clauses = split_into_clauses(text)
	if not clauses:
		logger.warning("No clauses were produced from input text. Returning empty result.")
		return {}

	logger.info("Processing contract with %d clause(s).", len(clauses))
	predictions = predict_batch(clauses)
	if not predictions:
		logger.warning("Classifier returned no predictions. Returning empty result.")
		return {}

	important_clauses = filter_important_clauses(predictions)[:20]
	grouped_clauses = keep_top_k_per_label(group_by_label(important_clauses), k=2)
	logger.info("Pipeline identified %d important clause(s).", sum(len(items) for items in grouped_clauses.values()))
	return grouped_clauses


def process_contract_one_sided_high_risk(text: str, max_items: int = 20) -> List[Dict[str, Any]]:
	"""Return only one-sided HIGH-risk clauses for RAG rewriting.

	Steps:
	1) Split contract text into clause candidates.
	2) Classify each clause with the fine-tuned model.
	3) Keep only one-sided HIGH-risk clauses.
	4) Return top items ranked by bias/confidence.
	"""

	if not text or not text.strip():
		logger.warning("Received empty contract text. Returning empty result.")
		return []

	clauses = split_into_clauses(text)
	if not clauses:
		logger.warning("No clauses were produced from input text. Returning empty result.")
		return []

	logger.info("Running one-sided high-risk pipeline for %d clause(s).", len(clauses))
	predictions = predict_batch(clauses)
	if not predictions:
		logger.warning("Classifier returned no predictions. Returning empty result.")
		return []

	risky_clauses = filter_one_sided_high_risk_clauses(predictions)[:max_items]
	logger.info("Pipeline identified %d one-sided high-risk clause(s).", len(risky_clauses))
	return risky_clauses


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

	one_sided_results = process_contract_one_sided_high_risk(sample_contract_text)
	print("\nOne-Sided High-Risk Clauses:")
	print(json.dumps(one_sided_results, indent=2))
