import os
from dotenv import load_dotenv

load_dotenv()
# ---- SETTINGS ----
DATA_PATH = r"C:\Rag_Application\Rag_Application\SajjadHussain..AI.pdf"
CHROMA_PATH = r"C:\Rag_Application\Rag_Application\chroma_db"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 40