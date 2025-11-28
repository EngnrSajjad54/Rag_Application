import os
import shutil
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from config import CHROMA_PATH

def save_to_chroma(chunks, overwrite=True):
    if os.path.exists(CHROMA_PATH) and overwrite:
        shutil.rmtree(CHROMA_PATH)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(),
        persist_directory=CHROMA_PATH
    )
    vectordb.persist()
    print(f"Saved {len(chunks)} chunks with metadata to ChromaDB at {CHROMA_PATH}")
    return vectordb

def load_chroma():
    if not os.path.exists(CHROMA_PATH):
        print("ChromaDB not found. Please generate the vectorstore first.")
        return None

    vectordb = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=OpenAIEmbeddings()
    )
    return vectordb
