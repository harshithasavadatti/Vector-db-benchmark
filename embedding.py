# embedding.py

import os
import pickle

from sentence_transformers import SentenceTransformer
from openai import OpenAI

from config import OPENAI_API_KEY


def load_models():

    openai_client = None

    if OPENAI_API_KEY:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)

    return {
        "minilm": SentenceTransformer("all-MiniLM-L6-v2"),
        "mpnet": SentenceTransformer("all-mpnet-base-v2"),
        "openai": openai_client
    }


def embed_st(texts, model, name):

    cache = f"{name}_embeddings.pkl"

    if os.path.exists(cache):

        print("Loading cached:", name)

        with open(cache, "rb") as f:
            return pickle.load(f)

    print("Generating embeddings:", name)

    emb = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True
    )

    with open(cache, "wb") as f:
        pickle.dump(emb, f)

    return emb


def embed_openai(texts, client):

    cache = "openai_embeddings.pkl"

    if os.path.exists(cache):

        print("Loading cached: openai")

        with open(cache, "rb") as f:
            return pickle.load(f)

    print("Generating OpenAI embeddings")

    vectors = []

    for i in range(0, len(texts), 100):

        batch = texts[i:i+100]

        res = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )

        for r in res.data:
            vectors.append(r.embedding)

    with open(cache, "wb") as f:
        pickle.dump(vectors, f)

    return vectors
