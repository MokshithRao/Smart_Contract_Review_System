"""Risk filtering utilities for clause classification outputs.

This module assigns risk levels to predicted clauses and returns only
high-importance items for downstream contract-review workflows.
"""

from __future__ import annotations

from typing import Any, Dict, List


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
	"change of control",
	"anti-assignment",
	"non-compete",
	"exclusivity",
	"liquidated damages",
	"covenant not to sue",
	"source code escrow",
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

_RISK_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
_RISK_SCORE = {"HIGH": 2, "MEDIUM": 1, "LOW": 0}


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


def filter_important_clauses(clauses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
	"""Keep only high/medium risk clauses and sort by risk then confidence."""

	enriched: List[Dict[str, Any]] = []
	for clause in clauses:
		text = str(clause.get("text", "")).strip()
		if not text:
			continue

		confidence = float(clause.get("confidence", 0.0) or 0.0)
		if confidence < 0.4:
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
			}
		)

	enriched.sort(key=lambda item: (_RISK_ORDER[item["risk"]], -item["confidence"]))
	return enriched
