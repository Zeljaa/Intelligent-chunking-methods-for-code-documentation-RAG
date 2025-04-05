# 🧠 Intelligent Chunking Methods for Code Documentation RAG  
**JetBrains Internship Entry Task**

---

## 📌 Overview  
This project explores how different chunking parameters and the number of retrieved chunks affect the **recall** and **precision** of retrieved data in a retrieval-augmented generation (RAG) system.

---

## 🎯 Objective  
Implement a full retrieval pipeline using open-source embedding models, and evaluate retrieval quality across varying:

- Chunking strategies  
- Chunk sizes  
- Number of retrieved chunks (top-k)

---

## 📚 Dataset  
- **Corpus:** `Chatlogs (chatlogs.md)`  
- **Queries & Golden Excerpts:** `questions_df.csv`  

---

## 🛠️ Dataset Preparation  
To enable experimentation, the dataset was processed in two distinct formats:

- **User-only messages**: Extracted and chunked only the `content` field from each message (excluding metadata).
- **Full chat format**: Retained the full dictionary, including fields like `content` and `role` (e.g., username or system/user tag), and chunked the serialized text.

Example of a raw entry:
```python
{
  "content": "These instructions apply to section-based themes (Responsive 6.0+, Retina 4.0+, Parallax 3.0+ Turbo 2.0+, Mobilia 5.0+)...",
  "role": "user"
}

```
---

## 🧩 Chunking Strategy  
Used the [**`FixedTokenChunker`**](https://github.com/brandonstarxel/chunking_evaluation/blob/main/chunking_evaluation/chunking/fixed_token_chunker.py) implementation, as described in the reference paper.

This method splits text into **fixed-length token chunks** using a tokenizer. Internally, it uses the **`cl100k_base` tokenizer**, which is the same tokenizer used by **OpenAI's ChatGPT-3.5-turbo** and **GPT-4-turbo**. This ensures consistency with real-world production systems and accurate token-level chunking.

The following configurations were explored:

- **Chunk Sizes (in tokens)**:
  - 10  
  - 50  
  - 100  
  - 150  
  - 200  
  - 250

- **Number of Retrieved Chunks (`k`)**:
  - 1  
  - 3  
  - 5

---

## 🔍 Embedding Model  
Used the [**`all-MiniLM-L6-v2`**](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) model via the `SentenceTransformers` library.

- Generated dense embeddings for all corpus chunks and queries.
- Used **cosine similarity** to compute distances between query and chunk embeddings.

---

## 📊 Evaluation Metrics  
Evaluated retrieval performance using:

- **Precision**
- **Recall**
- **F1 Score**

All scores are **averaged across 55 evaluation queries**, with **standard deviation** also reported to measure variance.
