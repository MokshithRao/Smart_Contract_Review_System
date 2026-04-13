"""Clause classification utilities for LegalBERT LoRA inference."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer


torch.set_grad_enabled(False)


logger = logging.getLogger(__name__)


BASE_MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
REPO_ROOT = Path(__file__).resolve().parents[2]
ADAPTER_DIR = REPO_ROOT / "models" / "legalbert-lora"
LABEL_MAP_PATH = REPO_ROOT / "configs" / "label_mapping.json"
DEFAULT_TOP_K = 3
MAX_LENGTH = 256
CONFIDENCE_THRESHOLD = 0.4


_MODEL = None
_TOKENIZER = None
_ID2LABEL: Dict[int, str] | None = None
_DEVICE: torch.device | None = None


def _empty_prediction(text: str | None = None) -> Dict[str, Any]:
	"""Return a safe default prediction for empty input."""

	prediction = {"label": "uncertain", "confidence": 0.0, "top_k": []}
	if text is not None:
		prediction["text"] = text
	return prediction


def _load_label_mapping(label_map_path: Path = LABEL_MAP_PATH) -> Dict[int, str]:
	"""Load the integer-to-label mapping used by the classifier."""

	if not label_map_path.exists():
		raise FileNotFoundError(f"Label mapping not found: {label_map_path}")

	with label_map_path.open("r", encoding="utf-8") as handle:
		label_mapping = json.load(handle)

	if not isinstance(label_mapping, dict):
		raise ValueError("Label mapping file must contain a JSON object.")

	id2label: Dict[int, str] = {}
	for label, index in label_mapping.items():
		if not isinstance(label, str):
			raise ValueError("Label names in label mapping must be strings.")
		if not isinstance(index, int):
			raise ValueError("Label ids in label mapping must be integers.")
		id2label[index] = label

	if not id2label:
		raise ValueError("Label mapping is empty.")

	return id2label


def _ensure_model_loaded() -> Tuple[Any, Any, Dict[int, str], torch.device]:
	"""Load and cache the model, tokenizer, label mapping, and device."""

	global _MODEL, _TOKENIZER, _ID2LABEL, _DEVICE

	if _MODEL is not None and _TOKENIZER is not None and _ID2LABEL is not None and _DEVICE is not None:
		return _MODEL, _TOKENIZER, _ID2LABEL, _DEVICE

	model, tokenizer, id2label, device = load_model()
	_MODEL = model
	_TOKENIZER = tokenizer
	_ID2LABEL = id2label
	_DEVICE = device
	return model, tokenizer, id2label, device


def _resize_sequence_classification_head(model: Any, num_labels: int) -> Any:
	"""Resize the classifier head to match the fine-tuned label space."""

	classifier = getattr(model, "classifier", None)
	if classifier is None:
		return model

	hidden_size = getattr(model.config, "hidden_size", None)
	if hidden_size is None and hasattr(classifier, "in_features"):
		hidden_size = classifier.in_features
	if hidden_size is None:
		return model

	dropout = getattr(model, "dropout", None)
	classifier_dropout = getattr(model.config, "classifier_dropout", None)
	if dropout is not None and classifier_dropout is not None:
		dropout.p = classifier_dropout

	model.classifier = torch.nn.Linear(hidden_size, num_labels)
	model.config.num_labels = num_labels
	return model


def _format_top_k(probabilities: torch.Tensor, id2label: Dict[int, str], top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
	"""Convert probability scores into a JSON-friendly ranked label list."""

	top_k = max(1, min(top_k, probabilities.numel()))
	scores, indices = torch.topk(probabilities, k=top_k)

	ranked: List[Dict[str, Any]] = []
	for score, index in zip(scores.tolist(), indices.tolist()):
		ranked.append({"label": id2label[int(index)], "score": float(score)})
	return ranked


def _predict_probabilities(texts: List[str]) -> Tuple[torch.Tensor, Dict[int, str]]:
	"""Run batched inference and return class probabilities for each text."""

	model, tokenizer, id2label, device = _ensure_model_loaded()

	cleaned_texts = [text if text is not None else "" for text in texts]
	encoded_inputs = tokenizer(
		cleaned_texts,
		padding=True,
		truncation=True,
		return_tensors="pt",
		max_length=MAX_LENGTH,
	)
	encoded_inputs = {key: value.to(device) for key, value in encoded_inputs.items()}

	logger.debug("Running batch inference for %d text(s).", len(cleaned_texts))
	with torch.no_grad():
		outputs = model(**encoded_inputs)
		logits = outputs.logits
		probabilities = torch.softmax(logits, dim=-1)

	return probabilities.cpu(), id2label


def load_model() -> Tuple[Any, Any, Dict[int, str], torch.device]:
	"""Load the base LegalBERT model, apply the LoRA adapter, and prepare the tokenizer."""

	if not ADAPTER_DIR.exists():
		raise FileNotFoundError(f"LoRA adapter directory not found: {ADAPTER_DIR}")

	logger.info("Loading clause classification model from %s.", ADAPTER_DIR)
	id2label = _load_label_mapping()
	label2id = {label: index for index, label in id2label.items()}

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
	base_model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_NAME)
	base_model = _resize_sequence_classification_head(base_model, len(id2label))
	base_model.config.num_labels = len(id2label)
	base_model.config.id2label = id2label
	base_model.config.label2id = label2id
	model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
	model.to(device)
	model.eval()

	if hasattr(model, "config"):
		model.config.num_labels = len(id2label)
		model.config.id2label = id2label
		model.config.label2id = label2id

	logger.info("Model loaded on %s with %d labels.", device, len(id2label))
	return model, tokenizer, id2label, device


def predict_clause(text: str) -> Dict[str, Any]:
	"""Predict the most likely clause label for a single input text."""

	if not text or not text.strip():
		return _empty_prediction()

	logger.debug("Predicting single clause.")
	probabilities, id2label = _predict_probabilities([text])
	row = probabilities[0]
	top_k = _format_top_k(row, id2label, DEFAULT_TOP_K)
	label = top_k[0]["label"]
	confidence = float(top_k[0]["score"])

	if confidence < CONFIDENCE_THRESHOLD:
		label = "uncertain"

	return {
		"label": label,
		"confidence": confidence,
		"top_k": top_k,
	}


def predict_batch(texts: List[str]) -> List[Dict[str, Any]]:
	"""Predict clause labels for a batch of texts."""

	if not texts:
		return []

	logger.debug("Predicting batch of %d text(s).", len(texts))
	non_empty_indices = [index for index, text in enumerate(texts) if text and text.strip()]
	if not non_empty_indices:
		return [_empty_prediction(text) for text in texts]

	probabilities, id2label = _predict_probabilities([texts[index] for index in non_empty_indices])

	predictions: List[Dict[str, Any]] = [_empty_prediction(text) for text in texts]
	for position, row in zip(non_empty_indices, probabilities):
		text = texts[position]

		top_k = _format_top_k(row, id2label, DEFAULT_TOP_K)
		label = top_k[0]["label"]
		confidence = float(top_k[0]["score"])

		if confidence < CONFIDENCE_THRESHOLD:
			label = "uncertain"

		predictions[position] = {
			"text": text,
			"label": label,
			"confidence": confidence,
			"top_k": top_k,
		}

	return predictions


def split_into_clauses(text: str) -> List[str]:
	"""Split text into clause-like segments using sentence and newline boundaries."""

	if not text:
		return []

	parts = re.split(r"\.\s+|\n+", text)
	return [part.strip() for part in parts if part and part.strip()]


if __name__ == "__main__":
	sample_contract_text = (
		"The company may terminate this agreement for convenience upon thirty days' written notice. "
		"The parties agree that the governing law shall be the laws of New York.\n"
		"The license granted hereunder is non-transferable."
	)

	try:
		load_model()
		clauses = split_into_clauses(sample_contract_text)
		results = predict_batch(clauses)
		print(json.dumps(results, indent=2))
	except Exception as exc:
		print(json.dumps({"error": str(exc)}, indent=2))
