# databases.py

import chromadb
from chromadb.config import Settings

from pinecone import Pinecone, ServerlessSpec

from config import CHROMA_PATH, PINECONE_API_KEY


# ---------------- CHROMA ----------------

def setup_chroma(model_name):

    client = chromadb.Client(
        Settings(
            persist_directory=CHROMA_PATH,
            anonymized_telemetry=False
        )
    )

    name = f"docs_{model_name}"

    col = client.get_or_create_collection(name)

    print("Using Chroma:", name)

    return col


def store_chroma(col, texts, emb, batch=1000):

    ids = [str(i) for i in range(len(texts))]

    if col.count() > 0:
        print("Chroma already populated")
        return

    for i in range(0, len(texts), batch):

        end = i + batch

        col.add(
            documents=texts[i:end],
            embeddings=emb[i:end].tolist(),
            ids=ids[i:end]
        )

        print("Chroma:", min(end, len(texts)))


# ---------------- PINECONE ----------------

def setup_pinecone(dim):

    pc = Pinecone(api_key=PINECONE_API_KEY)

    name = f"vector-db-{dim}"

    if name not in pc.list_indexes().names():

        pc.create_index(
            name=name,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

        print("Created Pinecone:", name)

    return pc.Index(name)


def store_pinecone(index, emb, texts, batch=100):

    if index.describe_index_stats()["total_vector_count"] > 0:
        print("Pinecone already populated")
        return

    vectors = []

    for i, e in enumerate(emb):

        vectors.append({
            "id": str(i),
            "values": e.tolist(),
            "metadata": {"text": texts[i]}
        })

    for i in range(0, len(vectors), batch):

        end = i + batch

        index.upsert(vectors=vectors[i:end])

        print("Pinecone:", min(end, len(vectors)))
