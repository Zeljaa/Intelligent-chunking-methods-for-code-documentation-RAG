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


## Results

| DataSet Preparation         | Chunk Size | K  | Recall (avg ± std)         | Precision (avg ± std)      | F1 (avg ± std)           |
|-----------------------------|------------|----|----------------------------|----------------------------|--------------------------|
| **User-only messages**       | 10         | 1  | 0.054 ± 0.098              | 0.287 ± 0.437              | 0.088 ± 0.149            |
| **User-only messages**       | 10         | 3  | 0.114 ± 0.148              | 0.207 ± 0.224              | 0.137 ± 0.154            |
| **User-only messages**       | 10         | 5  | 0.161 ± 0.190              | 0.169 ± 0.167              | 0.149 ± 0.141            |
| **User-only messages**       | 50         | 1  | 0.374 ± 0.339              | 0.442 ± 0.396              | 0.376 ± 0.324            |
| **User-only messages**       | 50         | 3  | 0.637 ± 0.355              | 0.260 ± 0.169              | 0.350 ± 0.208            |
| **User-only messages**       | 50         | 5  | 0.733 ± 0.335              | 0.194 ± 0.112              | 0.293 ± 0.156            |
| **User-only messages**       | 100        | 1  | 0.468 ± 0.421              | 0.293 ± 0.278              | 0.344 ± 0.314            |
| **User-only messages**       | 100        | 3  | 0.772 ± 0.343              | 0.211 ± 0.148              | 0.310 ± 0.180            |
| **User-only messages**       | 100        | 5  | 0.848 ± 0.296              | 0.142 ± 0.106              | 0.232 ± 0.137            |
| **User-only messages**       | 150        | 1  | 0.517 ± 0.438              | 0.238 ± 0.242              | 0.307 ± 0.285            |
| **User-only messages**       | 150        | 3  | 0.815 ± 0.327              | 0.166 ± 0.129              | 0.259 ± 0.168            |
| **User-only messages**       | 150        | 5  | 0.886 ± 0.267              | 0.112 ± 0.098              | 0.188 ± 0.133            |
| **User-only messages**       | 200        | 1  | 0.589 ± 0.430              | 0.216 ± 0.202              | 0.294 ± 0.237            |
| **User-only messages**       | 200        | 3  | 0.893 ± 0.272              | 0.151 ± 0.130              | 0.238 ± 0.159            |
| **User-only messages**       | 200        | 5  | 0.937 ± 0.204              | 0.092 ± 0.083              | 0.160 ± 0.117            |
| **User-only messages**       | 250        | 1  | 0.560 ± 0.463              | 0.172 ± 0.200              | 0.247 ± 0.250            |
| **User-only messages**       | 250        | 3  | 0.880 ± 0.283              | 0.138 ± 0.138              | 0.220 ± 0.169            |
| **User-only messages**       | 250        | 5  | 0.946 ± 0.185              | 0.083 ± 0.079              | 0.145 ± 0.108            |

