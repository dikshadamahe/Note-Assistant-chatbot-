# Intelligent Notes Assistant Chatbot (BYOP Ready)

A production-oriented CLI chatbot that answers questions from your notes using **BERT sentence embeddings** and semantic search.

## Why This Project

This project solves a practical BYOP problem: finding accurate answers from long note documents without manually scanning them. It is designed for coursework evaluation with:

- Explainable NLP analysis mode
- Quantitative evaluation pipeline
- Clean modular Python structure
- Terminal-only, reproducible workflow

## Key Upgrades

### 1. BERT-Based Semantic Search (Task 1)

Retrieval is upgraded from TF-IDF search to **Sentence Transformers embeddings**:

- Model: `all-MiniLM-L6-v2`
- Notes are split into sentences and embedded once
- Embeddings are cached on disk in `.cache/embeddings/`
- Query embedding is generated at runtime
- Cosine similarity ranks top-matching sentences
- Existing CLI flow remains unchanged

### 2. Evaluation System (Task 2)

A new `evaluation.py` module provides benchmark evaluation:

- 30 fixed QA test cases
- Metrics:
  - Accuracy
  - Approximate token-overlap precision
  - Approximate token-overlap recall
  - Approximate F1
- Failed-case breakdown:
  - Question
  - Expected answer
  - Predicted answer
  - Similarity + precision/recall
- Saves report to `evaluation_report.txt`

### 3. Better Dataset Handling (Task 3)

The project now validates datasets more robustly:

- Invalid path checks
- Empty-file checks
- Readability checks
- UTF-8 decoding validation
- Cleaner sentence splitting and heading/noise filtering

### 4. Code Quality + Error Handling (Task 4)

- Syntax/indentation issues fixed
- Invalid tokens removed
- Improved docstrings and modular design
- Defensive checks for missing/empty input

### 5. Performance Optimization (Task 5)

- Embeddings are cached to avoid recomputation
- Fast cosine scoring using NumPy arrays
- Query-time retrieval remains lightweight

### 6. Query Logging (Task 6 - Bonus)

User queries are logged to `chatbot_queries.log` with:

- Query text
- Best similarity score
- Top-3 similarity scores
- Confidence label
- Runtime latency (ms)

## Project Structure

```text
notes-assistant-chatbot/
├── main.py
├── chatbot.py
├── preprocessing.py
├── utils.py
├── evaluation.py
├── requirements.txt
├── README.md
├── data/
│   ├── notes.txt
│   └── startup_metrics_notes.txt
└── .cache/
    └── embeddings/
```

## Installation

```bash
cd notes-assistant-chatbot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the Chatbot

Interactive mode:

```bash
python main.py
```

Use a custom dataset:

```bash
python main.py --notes-file data/startup_metrics_notes.txt
```

Tune similarity threshold:

```bash
python main.py --notes-file data/notes.txt --threshold 0.25
```

## Run Evaluation

```bash
python evaluation.py
```

Or via main entrypoint:

```bash
python main.py --evaluate
```

Custom dataset + report file:

```bash
python main.py --evaluate --notes-file data/notes.txt --report-file evaluation_report.txt
```

Custom threshold in evaluation:

```bash
python evaluation.py --notes-file data/notes.txt --report-file evaluation_report.txt --threshold 0.2
```

## Evaluation Results (Latest Run)

Latest benchmark run on `data/notes.txt`:

- Total Questions: 30
- Accuracy: 33.33%
- Avg Precision (approx): 35.38%
- Avg Recall (approx): 33.21%
- Approx F1: 34.26%
- Failed Cases: 20

Note: these values depend heavily on dataset wording and sentence quality. You can improve scores by refining notes content and evaluation prompts.

## Example Output

### Ask Question

```text
Question: What is Natural Language Processing (NLP)?
Answer: Natural Language Processing (NLP) is a subfield of Artificial Intelligence (AI)...
Similarity Score: 82.16%
Confidence: High
```

### Evaluation

```text
NOTES ASSISTANT EVALUATION REPORT
Total Test Questions: 30
Accuracy: 33.33%
Average Precision (approx): 35.38%
Average Recall (approx): 33.21%
F1 (approx): 34.26%
```

## BYOP Submission Readiness

This project is submission-ready because it is:

- Fully terminal runnable (`python main.py`)
- Easy to set up (`pip install -r requirements.txt`)
- No GUI dependency
- Quantitatively evaluable
- Modular and maintainable

## Future Improvements

- Cross-encoder reranking for top-k results
- Domain-specific embedding models
- Better automatic threshold tuning
- Optional history-aware multi-turn context
- CSV/JSON evaluation export for report automation
