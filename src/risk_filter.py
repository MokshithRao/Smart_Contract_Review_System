"""Risk filtering utilities for clause classification outputs.

This module assigns risk levels to predicted clauses and returns only
high-importance items for downstream contract-review workflows.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
	from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency fallback path
	SentenceTransformer = None  # type: ignore[assignment]


# Category descriptions adapted from the provided category descriptions sheet.
CATEGORY_DESCRIPTIONS: Dict[str, str] = {
	"document name": "The name of the contract.",
	"parties": "The two or more parties who signed the contract.",
	"agreement date": "The date of the contract.",
	"effective date": "The date when the contract becomes effective.",
	"expiration date": "When the initial contract term expires.",
	"renewal term": "Renewal term after initial expiry, including auto-renewals.",
	"notice period to terminate renewal": "Required notice to stop renewal.",
	"governing law": "State or country law governing contract interpretation.",
	"most favored nation": "Right to receive better third-party terms.",
	"non-compete": "Restriction on competing activities.",
	"exclusivity": "Exclusive dealing or restriction on third-party dealings.",
	"no-solicit of customers": "Restriction on soliciting counterparty customers.",
	"competitive restriction exception": "Exception/carveout to competition restrictions.",
	"no-solicit of employees": "Restriction on soliciting/hiring counterparty workers.",
	"non-disparagement": "Requirement not to disparage the counterparty.",
	"termination for convenience": "Right to terminate without cause with notice.",
	"rofr/rofo/rofn": "Right of first refusal/offer/negotiation rights.",
	"change of control": "Rights triggered by merger, sale, or assignment by operation of law.",
	"anti-assignment": "Consent/notice requirements for assignment.",
	"revenue/profit sharing": "Requirement to share revenue or profits.",
	"price restrictions": "Limits on raising or reducing prices.",
	"minimum commitment": "Minimum buy/order or quantity obligation.",
	"volume restriction": "Fee/consent triggers above usage thresholds.",
	"ip ownership assignment": "IP created by one party becomes counterparty property.",
	"joint ip ownership": "Shared ownership of IP between parties.",
	"license grant": "License grant by one party to another.",
	"non-transferable license": "Limits on transferring granted license.",
	"affiliate license-licensor": "License includes licensor affiliates' IP.",
	"affiliate license-licensee": "License extends to licensee affiliates.",
	"unlimited/all-you-can-eat-license": "Enterprise or unlimited-use license.",
	"irrevocable or perpetual license": "Irrevocable or perpetual licensing rights.",
	"source code escrow": "Source code escrow obligations and release triggers.",
	"post-termination services": "Post-termination obligations and transition duties.",
	"audit rights": "Right to inspect books/records/locations for compliance.",
	"uncapped liability": "Liability exposure without cap.",
	"cap on liability": "Liability cap or limit on recovery/time to claim.",
	"liquidated damages": "Pre-agreed damages or termination fee.",
	"warranty duration": "Length of product/service warranty period.",
	"insurance": "Required insurance coverage obligations.",
	"covenant not to sue": "Restriction on challenging IP or bringing certain claims.",
	"third party beneficiary": "Non-signatory beneficiary rights.",
}


HIGH_RISK_LABELS = {
	"uncapped liability",
	"ip ownership assignment",
	"termination for convenience",
	"liquidated damages",
	"covenant not to sue",
}

MEDIUM_RISK_LABELS = {
	"cap on liability",
	"audit rights",
	"price restrictions",
	"minimum commitment",
	"volume restriction",
	"post-termination services",
	"no-solicit of customers",
	"no-solicit of employees",
	"most favored nation",
	"insurance",
	"rofr/rofo/rofn",
	"non-transferable license",
	"irrevocable or perpetual license",
	"third party beneficiary",
	"anti-assignment",
	"exclusivity",
	"non-compete",
}

HIGH_RISK_KEYWORDS = {
	"without notice",
	"immediate termination",
	"terminate for convenience",
	"sole discretion",
	"uncapped",
	"unlimited liability",
	"assign all rights",
	"exclusive",
	"non-compete",
	"liquidated damages",
	"injunctive relief",
}

MEDIUM_RISK_KEYWORDS = {
	"audit",
	"minimum purchase",
	"minimum commitment",
	"price increase",
	"price adjustment",
	"notice period",
	"renewal",
	"insurance",
	"indemnify",
	"cap on liability",
	"right of first refusal",
	"right of first offer",
}

ONE_SIDED_HIGH_RISK_PATTERNS = {
	"sole discretion": 2,
	"its sole discretion": 2,
	"absolute discretion": 2,
	"without notice": 2,
	"immediate termination": 2,
	"terminate for convenience": 2,
	"unilateral": 2,
	"shall remain with company": 2,
	"company is not liable": 2,
	"not liable": 2,
	"shall not": 1,
	"may not": 1,
	"without prior written approval": 1,
	"without prior written consent": 1,
	"at any time": 1,
	"for any reason": 1,
	"notwithstanding anything to the contrary": 1,
	"indemnify and hold harmless": 2,
	"waive": 1,
	"no liability": 2,
	"disclaims all": 2,
	"exclusive remedy": 1,
	"injunctive relief": 1,
}

BALANCED_LANGUAGE_PATTERNS = {
	"either party",
	"both parties",
	"mutual",
	"each party",
	"respectively",
	"by either party",
}

_RISK_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
_RISK_SCORE = {"HIGH": 2, "MEDIUM": 1, "LOW": 0}
SEMANTIC_DEDUP_THRESHOLD = 0.30
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


logger = logging.getLogger(__name__)


_EMBEDDING_MODEL: SentenceTransformer | None = None


def _normalize_for_similarity(text: str) -> str:
	"""Normalize text for semantic similarity comparisons."""

	normalized = text.lower()
	normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
	normalized = re.sub(r"\s+", " ", normalized).strip()
	return normalized


def _get_embedding_model() -> SentenceTransformer | None:
	"""Lazily load the embedding model used for semantic deduplication."""

	global _EMBEDDING_MODEL
	if SentenceTransformer is None:
		return None

	if _EMBEDDING_MODEL is None:
		_EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)

	return _EMBEDDING_MODEL


def _deduplicate_clauses_tfidf(clauses: List[Dict[str, Any]], normalized_texts: List[str]) -> List[Dict[str, Any]]:
	"""Fallback semantic deduplication using TF-IDF cosine similarity."""

	vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
	matrix = vectorizer.fit_transform(normalized_texts)

	kept_indices: List[int] = []
	for index in range(len(clauses)):
		if not kept_indices:
			kept_indices.append(index)
			continue

		same_label_kept = [kept for kept in kept_indices if clauses[kept].get("label") == clauses[index].get("label")]
		if not same_label_kept:
			kept_indices.append(index)
			continue

		similarities = cosine_similarity(matrix[index], matrix[same_label_kept])[0]
		if float(similarities.max()) >= SEMANTIC_DEDUP_THRESHOLD:
			continue

		kept_indices.append(index)

	return [clauses[index] for index in kept_indices]


def _deduplicate_clauses_embeddings(clauses: List[Dict[str, Any]], normalized_texts: List[str]) -> List[Dict[str, Any]] | None:
	"""Preferred semantic deduplication using sentence embeddings."""

	model = _get_embedding_model()
	if model is None:
		return None

	try:
		embeddings = model.encode(normalized_texts, normalize_embeddings=True)
	except Exception as exc:  # pragma: no cover - runtime fallback path
		logger.warning("Embedding-based dedup unavailable, falling back to TF-IDF: %s", exc)
		return None

	kept_indices: List[int] = []
	for index in range(len(clauses)):
		if not kept_indices:
			kept_indices.append(index)
			continue

		same_label_kept = [kept for kept in kept_indices if clauses[kept].get("label") == clauses[index].get("label")]
		if not same_label_kept:
			kept_indices.append(index)
			continue

		candidate = embeddings[index]
		existing = embeddings[same_label_kept]
		similarities = np.dot(existing, candidate)
		if float(np.max(similarities)) >= SEMANTIC_DEDUP_THRESHOLD:
			continue

		kept_indices.append(index)

	return [clauses[index] for index in kept_indices]


def _find_matching_keyword(lowered_text: str, keywords: set[str]) -> str | None:
	"""Return the first matching keyword found in pre-lowered text, if any."""

	for keyword in keywords:
		if keyword in lowered_text:
			return keyword
	return None


def classify_risk(clause: Dict[str, Any]) -> Dict[str, str]:
	"""Classify a clause's risk level based on label and keyword heuristics.

	Priority order:
	1) High-risk keywords
	2) Medium-risk keywords
	3) Label-based risk mapping
	"""

	label = str(clause.get("label", "")).strip().lower()
	text = str(clause.get("text", "")).strip()
	lowered = text.lower()

	high_keyword = _find_matching_keyword(lowered, HIGH_RISK_KEYWORDS)
	if high_keyword:
		return {
			"risk_level": "HIGH",
			"reason": f"Contains '{high_keyword}' (high-risk language)",
		}

	medium_keyword = _find_matching_keyword(lowered, MEDIUM_RISK_KEYWORDS)
	if medium_keyword:
		return {
			"risk_level": "MEDIUM",
			"reason": f"Contains '{medium_keyword}' (medium-risk language)",
		}

	if label in HIGH_RISK_LABELS:
		risk_level = "HIGH"
		reason = f"High-risk label: '{label}'"
	elif label in MEDIUM_RISK_LABELS:
		risk_level = "MEDIUM"
		reason = f"Medium-risk label: '{label}'"
	else:
		risk_level = "LOW"
		reason = f"Label '{label or 'unknown'}' is not in configured risk lists"

	description = CATEGORY_DESCRIPTIONS.get(label)
	if description:
		reason = f"{reason}. {description}"

	return {"risk_level": risk_level, "reason": reason}


def detect_one_sided_bias(text: str) -> Dict[str, Any]:
	"""Score whether a clause appears one-sided and return supporting triggers."""

	cleaned = (text or "").strip()
	if not cleaned:
		return {"is_one_sided": False, "bias_score": 0, "triggers": []}

	lowered = cleaned.lower()
	triggers: List[str] = []
	bias_score = 0

	for phrase, weight in ONE_SIDED_HIGH_RISK_PATTERNS.items():
		if phrase in lowered:
			triggers.append(phrase)
			bias_score += weight

	if any(phrase in lowered for phrase in BALANCED_LANGUAGE_PATTERNS):
		bias_score -= 2

	# A simple lexical check: more obligations than rights can indicate one-sided burden.
	obligation_hits = len(re.findall(r"\\b(shall|must|required to|obligated to)\\b", lowered))
	right_hits = len(re.findall(r"\\b(may|entitled to|has the right to)\\b", lowered))
	if obligation_hits > right_hits:
		bias_score += 1

	if "only" in lowered:
		bias_score += 1

	return {
		"is_one_sided": bias_score >= 2,
		"bias_score": bias_score,
		"triggers": sorted(set(triggers)),
	}


def deduplicate_clauses(clauses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
	"""Deduplicate near-duplicate clauses using cosine similarity of TF-IDF vectors."""

	if len(clauses) <= 1:
		return clauses

	normalized_texts = [_normalize_for_similarity(str(clause.get("text", ""))) for clause in clauses]
	if not any(normalized_texts):
		return clauses

	embedding_deduped = _deduplicate_clauses_embeddings(clauses, normalized_texts)
	if embedding_deduped is not None:
		return embedding_deduped

	return _deduplicate_clauses_tfidf(clauses, normalized_texts)


def filter_important_clauses(clauses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
	"""Keep only high/medium risk clauses and sort by risk then confidence."""

	enriched: List[Dict[str, Any]] = []
	for clause in clauses:
		text = str(clause.get("text", "")).strip()
		if not text:
			continue

		confidence = float(clause.get("confidence", 0.0) or 0.0)
		if confidence < 0.65:
			continue

		risk_meta = classify_risk(clause)
		risk_level = risk_meta["risk_level"]
		if risk_level not in {"HIGH", "MEDIUM"}:
			continue

		enriched.append(
			{
				"text": text,
				"label": str(clause.get("label", "")),
				"risk": risk_level,
				"risk_score": _RISK_SCORE[risk_level],
				"confidence": confidence,
				"reason": risk_meta["reason"],
				"top_k": clause.get("top_k", []),
			}
		)

	enriched.sort(key=lambda item: (_RISK_ORDER[item["risk"]], -item["confidence"]))
	return deduplicate_clauses(enriched)


def filter_one_sided_high_risk_clauses(
	clauses: List[Dict[str, Any]],
	min_confidence: float = 0.65,
) -> List[Dict[str, Any]]:
	"""Keep only HIGH-risk clauses that are likely one-sided.

	This is the preferred filter for downstream RAG rewriting where we only
	want potentially unfair clauses to be rewritten into balanced language.
	"""

	enriched: List[Dict[str, Any]] = []
	for clause in clauses:
		text = str(clause.get("text", "")).strip()
		if not text:
			continue

		confidence = float(clause.get("confidence", 0.0) or 0.0)
		if confidence < min_confidence:
			continue

		risk_meta = classify_risk(clause)
		if risk_meta.get("risk_level") != "HIGH":
			continue

		bias_meta = detect_one_sided_bias(text)
		if not bias_meta["is_one_sided"]:
			continue

		reason = risk_meta["reason"]
		if bias_meta["triggers"]:
			reason = f"{reason}. One-sided triggers: {', '.join(bias_meta['triggers'])}"

		enriched.append(
			{
				"text": text,
				"label": str(clause.get("label", "")),
				"risk": "HIGH",
				"risk_score": _RISK_SCORE["HIGH"],
				"confidence": confidence,
				"bias_score": int(bias_meta["bias_score"]),
				"one_sided": True,
				"reason": reason,
				"top_k": clause.get("top_k", []),
			}
		)

	enriched.sort(key=lambda item: (-item["bias_score"], -item["confidence"]))
	return deduplicate_clauses(enriched)
