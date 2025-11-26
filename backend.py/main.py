from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
import shutil
import os 

load_dotenv()
Data_path= "C:\Rag_Application\Rag_Application\SajjadHussain..AI.pdf"
# docuement loader
def document_loader():
    loader = PyPDFLoader(Data_path)
    return loader.load()
documents = document_loader()
# print(documents[0])
# text splitter
def text_splitter(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=40, length_function=len, add_start_index=True)
    chunk = text_splitter.split_documents(documents)
    print(f"Number of chunks: {len(chunk)}")
    documents = chunk[0]
    print(f"Document content: {documents.page_content}")
    print(f"Document metadata: {documents.metadata}")
    return chunk

Chroma_path = "C:/Rag_Application/Rag_Application/chroma_db"
def save_to_chroma(chunks: list[Document]):
    # clear out the existing chroma database folder
    if os.path.exists(Chroma_path):
        shutil.rmtree(Chroma_path)
        # create a new chroma database
    vectordb = Chroma.from_documents(documents=chunks, embedding=OpenAIEmbeddings(), persist_directory=Chroma_path)
    vectordb.persist()
    
    print(f"Saved {len(chunks)} chunks to ChromaDB at {Chroma_path}")

def generate_data_store():
    documents = document_loader()
    chunks = text_splitter(documents)
    save_to_chroma(chunks)
generate_data_store()

query = "Please tell me the key responsibilities of Sajjad Hussain as mentioned in the document?"

prompt = PromptTemplate(
    template="Answer the question based on the context below.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:",
    input_variables=["context", "question"]
)

def query_rag(query):
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory=Chroma_path, embedding_function=embeddings)
    result = vectordb.similarity_search(query, k=4)
    
    if len(result) == 0:
        print("No relevant documents found")
    context = "\n\n".join([doc.page_content for doc in result])
    final_prompt = prompt.format(context=context, question=query)
    llm = ChatOpenAI(temperature=0)
    response = llm.invoke(final_prompt)
    source = [doc.metadata for doc in result]
    formatted_response = {
        "response": response,
        "source": source
    }
    return formatted_response

query_rag(query)

