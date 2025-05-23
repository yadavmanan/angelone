# LLM Evaluation & Hallucination Detection Framework

## Overview
This repository provides a framework for evaluating the performance and factual correctness of large language models (LLMs), particularly OpenAI's GPT-4o and Mistral. It includes tools for:

- **Comparing LLM outputs** using multi-dimensional evaluation.
- **Detecting hallucinations** in generated text.
- **Benchmarking different prompt styles** for QA tasks.

The implementation is modular, with a clear separation between:
- Evaluation logic
- Hallucination detection
- Prompt optimization

---

## Repository Structure
| File             | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| `one.py`         | Evaluates GPT-4o and Mistral outputs based on relevance, coherence, factual accuracy, tone, and intent. |
| `two.py`         | Detects hallucinated claims in model outputs using reference data (Wikipedia) and a RoBERTa-based NLI model. |
| `three.py`       | Tests various prompt templates using semantic similarity to benchmark answer quality and select the best format. |
| `.env`           | Environment variables for API keys (not committed).                        |
| `requirements.txt` | Python dependencies.                                                     |

---

## Installation

### Clone the Repository
```bash
git clone https://github.com/yourusername/llm-eval-framework.git
cd llm-eval-framework
```

Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Set up your .env file with the following keys:
```
OPENAI_API_KEY=your_openai_key
MISTRAL_API_KEY=your_mistral_key
GOOGLE_FACTCHECK_API_KEY=your_google_fact_check_key
```

## Assumptions
- Wikipedia articles provide reliable ground truth for hallucination detection.
- Embedding similarity is a valid proxy for semantic relevance.
- Token overlap can approximate intent alignment.
- RoBERTa (MNLI) is sufficiently accurate for factual support verification.
- Evaluation metrics are consistent with a moderate temperature setting (0.7).

## Evaluation Metrics
| Metric            | Description                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| Factual Accuracy  | Google Fact Check API & RoBERTa NLI verdicts                                |
| Relevance         | Embedding similarity with prompt                                            |
| Coherence         | Sentence-to-token ratio                                                     |
| Tone & Style      | Polarity and subjectivity via TextBlob                                      |
| Intent Alignment  | Token overlap between prompt and response                                   |
| Semantic Match    | Cosine similarity to expected answers                                       |

## How to Run
1. Compare GPT vs Mistral
```bash
python one.py
```

2. Detect Hallucinations in Output
```bash
python two.py
```

3. Benchmark Prompt Templates
```bash
python three.py
```

## Tools & Models Used
- **LLMs**: GPT-4o (OpenAI), Mistral
- **Embeddings**: openai/embedding-3-large, sentence-transformers/all-MiniLM-L6-v2
- **NLI Model**: roberta-large-mnli from HuggingFace
- **Tokenization & Sentiment**: nltk, textblob
- **Fact Verification**: Google Fact Check Tools API

