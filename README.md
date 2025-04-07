# Intelligent Chunking Methods for Code Documentation RAG  
**JetBrains Internship Entry Task**



## 📌 Overview  
This project explores how different **chunking parameters** and the **number of retrieved chunks (top-k)** affect **recall** and **precision** in a retrieval-augmented generation (RAG) system. The goal is to evaluate which configurations lead to the most effective information retrieval, particularly for downstream language model tasks.



## 🎯 Objective  
Implement a complete retrieval pipeline and analyze the influence of:

- Chunk size  (`10`, `50`, `100`, `150`, `200`, `250` tokens)
- Number of retrieved chunks (`k` ∈ {1, 3, 5})  

on retrieval performance, using open-source tools and models.


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


---

## 📈 Results & Key Findings

To evaluate performance across configurations, two heatmaps were generated — one for **average precision** and one for **average recall**, across all combinations of chunk sizes and top-k values.

### 🔹 Precision Observations  
From the precision heatmap, a clear trend emerged:  
For **every tested chunk size**, the **highest precision** was achieved when **k = 1**.  
Among all configurations, the best result was:

> **Chunk Size: 50, k = 1 → Precision = 34.56%**
![eval_whole_result_avg_precision_heatmap](https://github.com/user-attachments/assets/9be2ce4b-f216-4a45-ad6f-2bbcfc1d1c97)

This highlights the importance of conservative retrieval in precision-oriented tasks — fewer chunks retrieved often means fewer irrelevant results.

---

### 🔹 Recall Observations  
Recall increased as **both chunk size and k value** increased — which is expected, as larger or more numerous chunks increase the chance of hitting relevant content.
![eval_whole_result_avg_recall_heatmap](https://github.com/user-attachments/assets/dd3b8c75-5e78-4237-a0b7-e955f7c4400b)

However, this metric alone can be misleading due to the increase in retrieved content length.

To normalize for chunk volume, we calculated **Recall per Token**, i.e., how much relevant information we retrieved relative to the number of tokens retrieved.

#### 📊 Top 5 Configurations by Recall per Token

| Chunk Size | k Value | Recall per Token |
|------------|---------|------------------|
| 50         | 1       | **6.81**          |
| 10         | 1       | 5.47              |
| 100        | 1       | 4.79              |
| 10         | 3       | 3.98              |
| 10         | 5       | 3.59              |

> It's particularly interesting that **Chunk Size = 50, k = 1** again outperforms all other settings, this time in terms of **recall efficiency**.

---

### 🔹 Overall Sweet Spot – AUC Comparison  
When combining **precision and recall**, we computed the **Area Under Curve (AUC)** to estimate overall performance balance.

- AUC also suggests that **k = 1** provides the best balance of both metrics.
- This aligns with the precision and recall (avg & per-token) results.
![advanced_analysis_pr_curves_by_k](https://github.com/user-attachments/assets/46a99e93-e05f-45a8-8457-53614fed8d10)
### 🔹 Complementary View – AUC by Chunk Size

We also flipped the view:  
> What if we fix **chunk size** and observe AUC across different **k** values?

This reveals how chunk granularity alone impacts performance.

<p align="center">
  <img src="https://github.com/user-attachments/assets/55b62d6e-c42a-4669-8617-8f9fb5140330" width="45%" />
  <img src="https://github.com/user-attachments/assets/958b2e3f-61ae-48b8-b16c-85883e465db7" width="45%" />
</p>


🧠 **Insight**: Chunk sizes around **50–100 tokens** strike the best balance — long enough for context, short enough to avoid noise. Paired with **k = 1**, this setup proved most effective for this corpus and question set.

---
## 🧹 What Happens When We Clean the Corpus?

Initially, the corpus included full chat messages with `"user"` and `"assistant"` roles, structured like this:

```json
[
  {"content": "These instructions apply to section-based themes...", "role": "user"},
  {"content": "This feature only applies to Collection pages...", "role": "assistant"},
  ...
]
```
However, golden answers consist of solo content only — individual utterances, either from users or assistants — and do not contain metadata like roles or JSON structures.

This discrepancy introduces noise during chunking, as unrelated user queries and assistant replies can be merged into the same chunk. To mitigate this, we introduced a preprocessing step to strip the corpus down to only the raw "content" values, producing a clean, flattened text stream more aligned with the golden answer format.

💡 Result?
--
After re-running the entire evaluation on the cleaned corpus, the top-performing configuration showed a dramatic improvement in precision:

🔼 Precision improved from 34.56% → 44.19% for Chunk Size = 50, k = 1
![eval_result_avg_precision_heatmap](https://github.com/user-attachments/assets/08214291-1d9d-40f8-8cfc-24e35a6421e8)

This clearly demonstrates that removing structural and conversational noise from the corpus leads to significantly better retrieval performance, especially in precision-focused setups.

This result reinforces a core RAG insight:

Even basic preprocessing — when aligned with the evaluation objective — can lead to meaningful downstream improvements.

------------

📌 *More in-depth analysis is available in the repository*, including:
- Full precision/recall/F1 tables (averages and standard deviations)  
- Per-question breakdowns  
- Additional exploratory metrics  

🔍 **Note:**  
Advanced analytics were performed using the **entire (uncleaned) corpus**.  
In the `basic_analytics` folder:  
- Files prefixed with `eval_whole_results_` refer to results **on the full corpus**  
- Files prefixed with `eval_results_` refer to results **on the cleaned corpus**

