from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from vector_store import load_chroma
from chat_memory import add_user_message, add_ai_message, get_history_text

#  prompt template
prompt = PromptTemplate(
    template="You are a helpful assistant. Use the chat history: {history} and the following context: {context} to answer the user question: {question}. Provide a clear helpful answer.",
    input_variables=["history", "context", "question"]
)

# rag query function
def rag_query(question):
    add_user_message(question)

    vectordb = load_chroma()
    results = vectordb.similarity_search(question, k=4)

    if not results:
        print(f"No relevant document context found.")

    context = "\n\n".join(doc.page_content for doc in results)
    history_text = get_history_text()

    final_prompt = prompt.format(history=history_text, context=context, question=question)

    llm = ChatOpenAI(temperature=0)
    response = llm.invoke(final_prompt)

    add_ai_message(response.content)
    return response.content