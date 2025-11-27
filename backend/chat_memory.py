from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

chat_history = [
    SystemMessage(content="You are a helpful assistant.")
]

def save_message_to_file(role, text):
    with open("chat_history.txt", "a") as f:
        f.write(f"{role}: {text}\n")

def add_user_message(text):
    chat_history.append(HumanMessage(content=text))
    save_message_to_file("User", text)



def add_ai_message(text):
    chat_history.append(AIMessage(content=text))
    save_message_to_file("AI", text)

def get_history_text():
    return "\n".join([msg.content for msg in chat_history])
