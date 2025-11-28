from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config import CHUNK_SIZE, CHUNK_OVERLAP
import time
import os

def split_documents(documents: list[Document]):
    new_docs = []

    for doc in documents:
        # FILE NAME extract
        file_name = os.path.basename(doc.metadata.get("source", "unknown_file"))

        # PAGE NUMBER (agar PDF loader de raha ho)
        page_number = doc.metadata.get("page", None)

        # TIMESTAMP
        timestamp = time.time()

        # BASIC METADATA ADD
        new_metadata = {
            "source": file_name,
            "page_number": page_number,
            "timestamp": timestamp,
            "section": doc.metadata.get("section", None)  # optional
        }

        # Replace metadata before splitting
        updated_doc = Document(
            page_content=doc.page_content,
            metadata=new_metadata
        )
        new_docs.append(updated_doc)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True
    )

    chunks = splitter.split_documents(new_docs)

    print(f" Total Chunks Created: {len(chunks)}")
    return chunks
