# 📌 Vector Database Performance Comparison  
### ChromaDB vs Pinecone for Document Retrieval

---

## 📖 Overview

This project presents a benchmarking framework to compare vector databases and retrieval algorithms for large-scale document retrieval.

It evaluates:

- Traditional lexical search (BM25)  
- Vector-based semantic search (Brute Force & HNSW)

across:

- ChromaDB (Local, Open Source)  
- Pinecone (Managed Cloud, Serverless)

The goal is to analyze latency vs retrieval quality trade-offs for different embedding models and indexing strategies.

---

## 📂 Dataset

- Source: PDF Documents (Technical / Educational / Resume-based)  
- Number of PDFs: ~1000  
- Average Pages per PDF: 5–10  
- Total Text Chunks: ~25,000  
- Extraction Tool: pypdf  

PDFs are excluded from the repository due to size constraints.

---

## 🧠 Embedding Models

| Model Name | Vector Dimension | Type |
|------------|------------------|------|
| all-MiniLM-L6-v2 | 384 | Lightweight, Fast |
| all-mpnet-base-v2 | 768 | High Quality |
| text-embedding-3-small | 1536 | OpenAI (Optional) |

---

## 🗄️ Vector Databases

| Database | Type | Deployment |
|----------|------|------------|
| ChromaDB | Open Source | Local |
| Pinecone | Managed | Cloud (Serverless) |

---

## 🔍 Retrieval Algorithms

| Algorithm | Category |
|-----------|----------|
| BM25 | Lexical Search |
| Brute Force | Exact Vector Search |
| HNSW | Approximate Nearest Neighbor |

---

## ⚙️ Evaluation Configuration

- Queries: 20 Natural Language Queries  
- Top-K: 5  
- Similarity Metric: Cosine Similarity  

---

## 📈 Key Findings

- BM25 lacks semantic understanding.  
- HNSW provides best performance.  
- ChromaDB is ideal for local testing.  
- Pinecone enables scalable cloud search.  

---

## 🚀 Installation

pip install langchain chromadb pinecone-client sentence-transformers openai pandas scikit-learn rank-bm25

---

## ▶️ Run

python main.py

---

## 📁 Output

Results saved in final_results.csv

---

## 🎯 Conclusion

HNSW-based vector search offers best trade-off between speed and quality.
