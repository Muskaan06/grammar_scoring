# 📘 **Grammar Scoring Using Transformer Regression**

## 📌 Overview

This project implements an **end-to-end grammar scoring system** using a **fine-tuned Transformer model**.
Given a text input (e.g., ASR transcript), the model predicts a **continuous grammar quality score**, suitable for:

* Spoken language assessment
* Answer quality scoring
* Grammar proficiency evaluation
* Automated feedback systems

The model is trained as a **regression task** using a Transformer encoder (DeBERTa-v3).

---

## 📂 Project Workflow

The notebook follows a clear and modular pipeline:

### **1. Dataset Loading**

* Training and test datasets are provided as separate CSVs.
* Each CSV contains:

  * `text` — the input sentence/transcript
  * `score` — the target grammar score (float)

### **2. Preprocessing**

* Columns are renamed (`score → labels`) to align with HuggingFace Trainer requirements.
* Tokenization uses a pretrained Transformer tokenizer with:

  * padding to max length
  * truncation
  * max length = 128 tokens
* Datasets are converted to HuggingFace Dataset objects and formatted for PyTorch.

---

## 🤖 **Model Architecture**

The model uses:

* A pretrained Transformer encoder (`microsoft/deberta-v3-base` by default)
* A regression head (`num_labels = 1`)
* `problem_type = "regression"` to force the model to output continuous values

This architecture allows the model to learn nuanced syntactic and semantic patterns that correlate with grammatical correctness.

---

## 🏋️ **Training Setup**

Training is performed using a custom HuggingFace `Trainer` class that computes **RMSE (Root Mean Squared Error)** as the loss function.

Key elements:

* **Custom RMSETrainer** overrides `compute_loss`
* Only training is performed (no evaluation step)
* Checkpoints saved periodically
* Logging of training progress
* GPU acceleration if available

TrainingArguments define:

* epochs
* batch size
* learning rate
* save/log intervals

---

## 📈 **Why RMSE Loss?**

Grammar scoring is a **continuous** prediction task.
RMSE:

* treats large errors more severely
* is more interpretable than MSE
* aligns well with scoring rubrics

Using RMSE directly during training helps stabilize gradient flow for regression tasks.

---

## 🔍 **Usage: Predicting Grammar Scores**

Once trained, the model can score custom sentences:

```python
score_text("He go to school yesterday")
# → returns a grammar score (float)
```

Batch scoring is also supported.

---

## 💾 **Saving & Loading the Model**

The fine-tuned model and tokenizer are saved automatically:

```
grammar_model_final/
├── config.json
├── pytorch_model.bin
├── tokenizer.json
└── tokenizer_config.json
```

To load the model:

```python
model = AutoModelForSequenceClassification.from_pretrained("grammar_model_final")
tokenizer = AutoTokenizer.from_pretrained("grammar_model_final")
```

