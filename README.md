# 🧠 Intelligent Chunking Methods for Code Documentation RAG  
**JetBrains Internship Entry Task**

---

## 📌 Overview  
This project explores how different chunking parameters and the number of retrieved chunks affect the **recall** and **precision** of retrieved data in a retrieval-augmented generation (RAG) system. All experiments were performed on the **Chatlogs** dataset.

---

## 🎯 Objective  
Implement a full retrieval pipeline using open-source embedding models, and evaluate retrieval quality across varying:

- Chunking strategies  
- Chunk sizes  
- Number of retrieved chunks (top-k)

---

## 📚 Dataset  
- **Corpus:** `Chatlogs`  
- **Queries & Golden Excerpts:** `questions_df.csv`  

---

## 🛠️ Dataset Preparation  
To enable experimentation, the dataset was processed in two different formats:

- **User-only messages**: Chunks consist only of messages sent by users.  
- **Full chat format**: Chunks contain both the message and the sender’s username.  

Queries were filtered and aligned to match the processed corpus for fair evaluation.

---

## 🧩 Chunking Strategy  
Implemented the **`FixedTokenChunker`** as described in the referenced paper.  
The following configurations were explored:

- **Chunk Sizes**:
  - 10  
  - 50  
  - 100  
  - 150  
  - 200  
  - 250 tokens

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
