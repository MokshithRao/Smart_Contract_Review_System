---
base_model: nlpaueb/legal-bert-base-uncased
library_name: peft
language:
- en
tags:
- legal
- contract
- text-classification
- lora
- peft
- bert
- cuad
- base_model:adapter:nlpaueb/legal-bert-base-uncased
license: mit
---

# LegalBERT · Contract Clause Classifier (LoRA)

A LoRA adapter fine-tuned on top of [`nlpaueb/legal-bert-base-uncased`](https://huggingface.co/nlpaueb/legal-bert-base-uncased) for multi-class contract clause classification across all **41 CUAD clause types**.

The model significantly outperforms a random baseline (accuracy 1.9% → **69.8%**, macro F1 0.005 → **0.650**) after 5 epochs of training.

---

## Model Details

| Property | Value |
|---|---|
| **Base model** | `nlpaueb/legal-bert-base-uncased` |
| **Adapter type** | LoRA (PEFT) |
| **Task** | Multi-class sequence classification |
| **Classes** | 41 CUAD clause types |
| **LoRA rank (r)** | 16 |
| **LoRA alpha** | 32 |
| **LoRA dropout** | 0.1 |
| **Target modules** | `query`, `value` |
| **Max sequence length** | 512 tokens |
| **PEFT version** | 0.18.1 |

---

## Training

The adapter was trained for **5 epochs** on the [CUAD dataset](https://huggingface.co/datasets/theatticusproject/cuad-qa), which contains expert-labelled contract clauses across 41 legal categories.

### Training curve

| Epoch | Train Loss | Val Loss | Accuracy | Macro F1 |
|---|---|---|---|---|
| 1 | 6.257 | 4.771 | 39.1% | 0.290 |
| 2 | 3.233 | 2.846 | 60.6% | 0.535 |
| 3 | 2.426 | 2.376 | 67.6% | 0.621 |
| 4 | 2.165 | 2.198 | 69.4% | 0.644 |
| 5 | 2.060 | 2.147 | **69.8%** | **0.650** |

### Baseline comparison

| Metric | Baseline (majority class) | This model |
|---|---|---|
| Accuracy | 1.9% | **69.8%** |
| Macro F1 | 0.005 | **0.650** |

---

## Usage

This is a PEFT LoRA adapter — you need to load it on top of the base model using the `peft` library.

### Installation

```bash
pip install transformers peft
```

### Inference

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

base_model_id = "nlpaueb/legal-bert-base-uncased"
adapter_id = "Mokshith31/legalbert-contract-clause-classification"

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForSequenceClassification.from_pretrained(
    base_model_id,
    num_labels=41
)
model = PeftModel.from_pretrained(base_model, adapter_id)
model.eval()

clause = "Either party may terminate this Agreement upon 30 days written notice."

inputs = tokenizer(clause, return_tensors="pt", truncation=True, max_length=512)
outputs = model(**inputs)
predicted_class = outputs.logits.argmax(dim=-1).item()
print(f"Predicted label: {predicted_class}")
```

### With pipeline (merged weights)

You can also merge the adapter into the base model and use the standard `pipeline` API:

```python
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

base = AutoModelForSequenceClassification.from_pretrained(
    "nlpaueb/legal-bert-base-uncased", num_labels=41
)
model = PeftModel.from_pretrained(base, "Mokshith31/legalbert-contract-clause-classification")
model = model.merge_and_unload()  # fuse LoRA weights

tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
```

---

## CUAD Label Schema

The model predicts one of the following 41 clause categories from the [CUAD dataset](https://www.atticusprojectai.org/cuad):

| ID | Clause Type |
|---|---|
| 0 | Document Name |
| 1 | Parties |
| 2 | Agreement Date |
| 3 | Effective Date |
| 4 | Expiration Date |
| 5 | Renewal Term |
| 6 | Notice Period to Terminate Renewal |
| 7 | Governing Law |
| 8 | Most Favored Nation |
| 9 | Non-Compete |
| 10 | Exclusivity |
| 11 | No-Solicit of Customers |
| 12 | No-Solicit of Employees |
| 13 | Non-Disparagement |
| 14 | Termination for Convenience |
| 15 | ROFR / ROFO / ROFN |
| 16 | Change of Control |
| 17 | Anti-Assignment |
| 18 | Revenue / Profit Sharing |
| 19 | Price Restriction |
| 20 | Minimum Commitment |
| 21 | Volume Restriction |
| 22 | IP Ownership Assignment |
| 23 | Joint IP Ownership |
| 24 | License Grant |
| 25 | Non-Transferable License |
| 26 | Affiliate License-Licensor |
| 27 | Affiliate License-Licensee |
| 28 | Unlimited / All-You-Can-Eat License |
| 29 | Irrevocable or Perpetual License |
| 30 | Source Code Escrow |
| 31 | Post-Termination Services |
| 32 | Audit Rights |
| 33 | Uncapped Liability |
| 34 | Cap on Liability |
| 35 | Liquidated Damages |
| 36 | Warranty Duration |
| 37 | Insurance |
| 38 | Covenant Not to Sue |
| 39 | Third Party Beneficiary |
| 40 | Other |

---

## Limitations and Bias

- Trained exclusively on **English-language** commercial contracts from the CUAD dataset. Performance may degrade on other legal domains (e.g. employment, real estate) or non-US contract styles.
- Some CUAD classes have very **few training examples** (e.g. class 2 has only 1 support sample), which leads to lower per-class performance on rare clause types.
- The model is **not a substitute for legal advice**. Predictions should be reviewed by qualified professionals before use in any legal workflow.
- Class imbalance in the CUAD dataset means the model may favour more common clause types.

---

## Citation

If you use this model, please cite the original CUAD dataset:

```bibtex
@article{hendrycks2021cuad,
  title={CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review},
  author={Hendrycks, Dan and Burns, Collin and Chen, Anya and Ball, Spencer},
  journal={arXiv preprint arXiv:2103.06268},
  year={2021}
}
```

And the LegalBERT base model:

```bibtex
@inproceedings{chalkidis-etal-2020-legal,
  title={LEGAL-BERT: The Muppets straight out of Law School},
  author={Chalkidis, Ilias and Fergadiotis, Manos and Malakasiotis, Prodromos and Aletras, Nikolaos and Androutsopoulos, Ion},
  booktitle={Findings of EMNLP},
  year={2020}
}
```

---

### Framework versions

- Transformers
- PEFT 0.18.1
