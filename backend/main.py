from rag_query import rag_query
from loader import document_loader
from spliter import split_documents
from vector_store import save_to_chroma

#  Initialize vector store
def generate_vectorstore():
    print(" Loading PDF...")
    documents = document_loader()

    print(" Splitting into chunks...")
    chunks = split_documents(documents)

    print(" Saving into Chroma...")
    save_to_chroma(chunks)

    print(" Vector DB ready!")

#  chat loop
def chat_loop():
    print("\n RAG Chat Started (type 'exit' to quit)\n")
    while True:
        query = input("Enter your query here: ")
        
        if query.lower() == "exit":
            break

        answer = rag_query(query)
        print("\nAI:", answer, "\n")

generate_vectorstore()
chat_loop()
