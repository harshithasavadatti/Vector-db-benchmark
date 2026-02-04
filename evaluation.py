# evaluation.py

import pandas as pd

from retrieval import *
from relevance import is_relevant


def evaluate(queries, model, emb, texts, bm25, chroma, pine):

    rows = []

    for q in queries:

        qe = model.encode(q)

        b_ids, t1 = search_bm25(bm25, q)
        r_ids, t2 = search_brute(emb, qe)
        c_ids, t3 = search_chroma(chroma, qe)
        p_ids, t4 = search_pinecone(pine, qe)


        def precision(ids):

            return sum(
                is_relevant(q, texts[int(i)])
                for i in ids
            ) / 5


        rows.append([
            q,
            t1, precision(b_ids),
            t2, precision(r_ids),
            t3, precision(c_ids),
            t4, precision(p_ids)
        ])


    return pd.DataFrame(
        rows,
        columns=[
            "Query",

            "BM25(ms)", "BM25@5",
            "Brute(ms)", "Brute@5",
            "Chroma(ms)", "Chroma@5",
            "Pinecone(ms)", "Pinecone@5"
        ]
    )
