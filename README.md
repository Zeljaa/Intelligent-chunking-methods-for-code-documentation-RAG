# Intelligent Chunking Methods for Code Documentation RAG  
**JetBrains Internship Entry Task**

---

## 📌 Overview  
This project explores how different **chunking parameters** and the **number of retrieved chunks (top-k)** affect **recall** and **precision** in a retrieval-augmented generation (RAG) system. The goal is to evaluate which configurations lead to the most effective information retrieval, particularly for downstream language model tasks.

---

## 🎯 Objective  
Implement a complete retrieval pipeline and analyze the influence of:

- Chunk size  (`10`, `50`, `100`, `150`, `200`, `250` tokens)
- Number of retrieved chunks (`k` ∈ {1, 3, 5})  

on retrieval performance, using open-source tools and models.

---

## 📚 Dataset  
- **Corpus**: `chatlogs.md` – a markdown file containing real-world technical chat messages.  
- **Evaluation Set**: `questions_df.csv` – a collection of questions and their corresponding golden answer spans, used to compute evaluation metrics.

---

## ⚙️ Retrieval Pipeline  
The retrieval process follows a simple yet effective architecture:

1. **Chunking**  
   The entire corpus is split into **fixed-length token chunks** using the [`FixedTokenChunker`](https://github.com/brandonstarxel/chunking_evaluation/blob/main/chunking_evaluation/chunking/fixed_token_chunker.py), which relies on the `cl100k_base` tokenizer (compatible with OpenAI's GPT models). Explored chunk sizes include:  
   `10`, `50`, `100`, `150`, `200`, `250` tokens.

2. **Embedding**  
   Each chunk, as well as every evaluation query, is embedded using the **`all-MiniLM-L6-v2`** model via the `SentenceTransformers` library.  
   - Embeddings are compared using **cosine similarity**.

3. **Retrieval**  
   For each query, the top `k` most similar chunks are retrieved (`k ∈ {1, 3, 5}`) based on embedding similarity.

4. **Evaluation**  
   Retrieved chunks are compared against the golden answers to compute:
   - **Recall**: proportion of relevant chunks found  
   - **Precision**: proportion of retrieved chunks that are relevant  
   - **F1 Score**: harmonic mean of precision and recall  
   Scores are averaged across **55 queries**, with standard deviations reported to assess consistency.
