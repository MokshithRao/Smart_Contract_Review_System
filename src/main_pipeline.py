"""Main pipeline for end-to-end contract clause risk analysis."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from src.inference.clause_classifier import predict_batch, split_into_clauses
from src.risk_filter import filter_important_clauses, filter_one_sided_high_risk_clauses

logger = logging.getLogger(__name__)


def _load_rag_helpers():
	"""Lazy import RAG helpers to avoid expensive startup overhead."""
	from rag.rewrite import explain_changes, get_similar_clauses, rewrite_clause

	return get_similar_clauses, rewrite_clause, explain_changes


def classify_contract(text: str) -> List[Dict[str, Any]]:
	"""Split raw contract text into clauses and classify each one."""

	if not text or not text.strip():
		logger.warning("Received empty contract text. Returning empty result.")
		return []

	clauses = split_into_clauses(text)
	if not clauses:
		logger.warning("No clauses were produced from input text. Returning empty result.")
		return []

	logger.info("Classifying contract with %d clause(s).", len(clauses))
	return predict_batch(clauses)


def process_contract(text: str) -> List[Dict[str, Any]]:
	"""Return important high/medium risk clauses from classified contract text."""

	predictions = classify_contract(text)
	if not predictions:
		return []

	important_clauses = filter_important_clauses(predictions)[:20]
	logger.info("Pipeline identified %d important clause(s).", len(important_clauses))
	return important_clauses


def build_rag_review(text: str, max_items: int = 3) -> List[Dict[str, Any]]:
	"""Produce RAG-based rewrites for top one-sided high-risk clauses."""

	predictions = classify_contract(text)
	if not predictions:
		return []

	risky_clauses = filter_one_sided_high_risk_clauses(predictions)[:max_items]
	if not risky_clauses:
		logger.info("No one-sided high-risk clauses found for RAG review; falling back to top important clauses.")
		risky_clauses = filter_important_clauses(predictions)[:max_items]
		if not risky_clauses:
			logger.info("No important clauses available for fallback RAG review.")
			return []

	try:
		get_similar_clauses, rewrite_clause, explain_changes = _load_rag_helpers()
	except Exception as error:
		logger.warning("RAG helpers unavailable: %s", error)
		return []

	reviews: List[Dict[str, Any]] = []
	for clause in risky_clauses:
		original = clause.get("text", "")
		clause_type = clause.get("label", "unknown")

		rewrite = ""
		explanation = ""
		similar_clauses = []

		try:
			similar_clauses = get_similar_clauses(original, k=5)
			rewrite = rewrite_clause(original, clause_type, similar_clauses)
			explanation = explain_changes(original, rewrite, clause_type)
		except Exception as error:
			logger.warning("RAG review failed for clause: %s", error)
			explanation = "RAG review could not be completed for this clause."

		reviews.append(
			{
				"original": original,
				"label": clause_type,
				"confidence": clause.get("confidence", 0.0),
				"risk": clause.get("risk", "HIGH"),
				"top_k": clause.get("top_k", []),
				"similar_clauses": similar_clauses,
				"rewrite": rewrite,
				"explanation": explanation,
			}
		)

	logger.info("Built %d RAG review(s).", len(reviews))
	return reviews


if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

	sample_contract_text = (
		"Either party may terminate this agreement for convenience with thirty days notice. "
		"Vendor shall maintain cyber liability insurance at all times. "
		"This agreement is governed by the laws of California.\n"
		"The customer has audit rights on reasonable notice."
	)

	all_predictions = classify_contract(sample_contract_text)
	print("All Clause Predictions:")
	print(any(all_predictions) and len(all_predictions))

	important = process_contract(sample_contract_text)
	print("\nImportant Clauses:")
	print(important)

	rage_reviews = build_rag_review(sample_contract_text)
	print("\nRAG Review Output:")
	print(rage_reviews)
