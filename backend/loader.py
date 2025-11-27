from langchain_community.document_loaders import PyPDFLoader
from config import DATA_PATH

def load_pdf():
    loader = PyPDFLoader(DATA_PATH)
    return loader.load()
