
# RAG Decision Lab – Multilingual Retrieval System

This project implements a simple **Retrieval-Augmented Generation (RAG)** pipeline using multiple embedding models and **FAISS** for semantic search.

It is designed to compare how different embedding models perform on the same dataset and query.

---

## Overview

The system:

* Reads Markdown documents
* Splits them into meaningful chunks
* Converts text into embeddings using multiple models
* Indexes them with FAISS
* Retrieves the most relevant chunks for a query
* Saves results in an HTML file for comparison

---

## Features

* Paragraph-based chunking (~700 characters)
* Multiple embedding models support
* Multilingual retrieval (Arabic + English)
* FAISS similarity search (cosine similarity via normalization)
* HTML visualization of results
* Memory-efficient execution

---

## Project Structure

```
telecom-rag-assistant/
│
├── data/
│   └── data/              # Add your .md files here (not included in repo)
│
├── rag_decision_lab.py    # Main pipeline
├── results.html           # Generated output (ignored in git)
├── .gitignore
└── README.md
```

---

## Clone the Repository

```bash
git clone https://github.com/Merna-Hany12/Telecom-RAG-Assistant.git
cd Telecom-RAG-Assistant
```

---

## Installation

Make sure you have Python 3.9+ installed, then run:

```bash
pip install numpy faiss-cpu sentence-transformers
```

---

## Data Setup (Required)

The dataset is **not included** in this repository.

You must create it manually:

```bash
mkdir -p data/data
```

Then add your Markdown files:

```
data/data/file1.md
data/data/file2.md
```

---

## How to Run

```bash
python rag_decision_lab.py
```

After running, open:

```
results.html
```

---

## Example Query

```python
query = "ايه عروض النت ؟"
```

The system supports:

* Arabic
* English
* Mixed-language queries

---

## Embedding Models Used

| Model                                   | Description                                         |
| --------------------------------------- | --------------------------------------------------- |
| `intfloat/multilingual-e5-base`         | Strong retrieval model (requires prefix formatting) |
| `BAAI/bge-m3`                           | High-performance multilingual model                 |
| `paraphrase-multilingual-mpnet-base-v2` | Good semantic similarity baseline                   |

---

## How It Works

### 1. Chunking

* Splits text by paragraphs
* Combines them into chunks with a size limit (~700 characters)

### 2. Embeddings

* Converts each chunk into a dense vector
* Uses `normalize_embeddings=True` to enable cosine similarity

### 3. Indexing (FAISS)

* Uses `IndexFlatIP`
* Inner product becomes cosine similarity after normalization

### 4. Retrieval

* Encodes the query
* Retrieves top-k similar chunks
* Returns:

  * Text
  * Source file
  * Similarity score

### 5. Output

* Results saved in `results.html`
* Easy comparison between models

---

## Important Notes

### E5 Model Requirement

The E5 model requires prefixes:

```
"query: your question"
"passage: document text"
```

This improves retrieval accuracy.

---

### Why normalize_embeddings=True?

* Converts similarity to cosine similarity
* Improves performance across models
* Required for BGE and E5

---

### Why FAISS IndexFlatIP?

* Fast and simple
* No training required
* Works well with normalized vectors

---

## .gitignore Recommendation

Make sure you ignore unnecessary files:

```
data/
results.html
__pycache__/
*.pyc
```

---

## Common Issues

### Missing Data Folder

Error:

```
FileNotFoundError: data/data
```

Solution:

* Create the folder manually
* Add `.md` files

---

### Slow First Run

* Models are downloaded the first time
* This is normal

---
