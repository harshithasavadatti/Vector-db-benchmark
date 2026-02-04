# loader.py

import glob

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import PDF_FOLDER, CHUNK_SIZE, CHUNK_OVERLAP


def load_pdfs():

    docs = []

    for path in glob.glob(PDF_FOLDER + "/*.pdf"):

        loader = PyPDFLoader(path)
        pages = loader.load()

        docs.extend(pages)

    print("Loaded pages:", len(docs))

    return docs


def split_documents(docs):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(docs)

    texts = [c.page_content for c in chunks]

    print("Total chunks:", len(texts))

    return texts
