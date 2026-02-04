# main.py

import pandas as pd

from loader import *
from embedding import *
from databases import *
from retrieval import setup_bm25
from evaluation import evaluate


def main():

    # Load PDFs
    docs = load_pdfs()
    texts = split_documents(docs)

    # BM25
    bm25 = setup_bm25(texts)

    # Queries
    queries = [

        "Python developer with machine learning experience",
        "Data scientist skilled in NLP and deep learning",
        "Backend engineer with Django and REST API",
        "AI engineer with TensorFlow and PyTorch",
        "Software developer with strong DSA skills",

        "Fresher software engineer with internship experience",
        "Candidate with two years experience in data analytics",
        "Entry level data scientist with projects",
        "Full stack developer with React and Node.js",
        "Professional with AWS cloud experience",

        "Computer science graduate in AI",
        "BTech graduate with programming skills",
        "Engineering student with research background",
        "Graduate specialized in data science",

        "Resume for ML engineer role",
        "Profile for data analyst job",
        "Backend Python developer profile",
        "Business intelligence analyst profile",

        "Experience with SQL Power BI Tableau",
        "Docker Kubernetes DevOps experience"
    ]


    models = load_models()

    all_results = []


    for name, model in models.items():

        print("\n===== MODEL:", name.upper(), "=====")


        # Embeddings
        if name == "openai":

            if model is None:
                continue

            emb = embed_openai(texts, model)
            dim = 1536

            q_model = models["minilm"]

        else:

            emb = embed_st(texts, model, name)
            dim = emb.shape[1]

            q_model = model


        # Databases
        chroma = setup_chroma(name)
        store_chroma(chroma, texts, emb)

        pine = setup_pinecone(dim)
        store_pinecone(pine, emb, texts)


        # Evaluate
        df = evaluate(
            queries,
            q_model,
            emb,
            texts,
            bm25,
            chroma,
            pine
        )

        df["Model"] = name

        all_results.append(df)


    final = pd.concat(all_results)

    final.to_csv("final_results.csv", index=False)

    print("\n===== FINAL RESULTS =====")
    print(final)


if __name__ == "__main__":
    main()
