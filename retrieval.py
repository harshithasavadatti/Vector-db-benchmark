# retrieval.py

import time
import numpy as np

from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity

from config import TOP_K


# -------- BM25 --------

def setup_bm25(texts):

    tok = [t.lower().split() for t in texts]

    return BM25Okapi(tok)


def search_bm25(bm25, q):

    s = time.time()

    scores = bm25.get_scores(q.lower().split())

    top = np.argsort(scores)[::-1][:TOP_K]

    return top, (time.time()-s)*1000


# -------- BRUTE --------

def search_brute(emb, q):

    s = time.time()

    sim = cosine_similarity([q], emb)[0]

    top = np.argsort(sim)[::-1][:TOP_K]

    return top, (time.time()-s)*1000


# -------- CHROMA --------

def search_chroma(col, q):

    s = time.time()

    r = col.query(
        query_embeddings=[q.tolist()],
        n_results=TOP_K
    )

    return r["ids"][0], (time.time()-s)*1000


# -------- PINECONE --------

def search_pinecone(index, q):

    s = time.time()

    r = index.query(
        vector=q.tolist(),
        top_k=TOP_K
    )

    ids = [m["id"] for m in r["matches"]]

    return ids, (time.time()-s)*1000
