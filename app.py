import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
import chatbot
from dotenv import load_dotenv
import nest_asyncio
import asyncio

# Apply nest_asyncio patch
nest_asyncio.apply()

# Ensure an event loop is running
if not asyncio.get_event_loop().is_running():
    asyncio.set_event_loop(asyncio.new_event_loop())


VECTORDB_PATH = "local_db"

@st.cache_resource
def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # vector_db = Chroma(persist_directory=VECTORDB_PATH, embedding_function=embeddings)
    # return vector_db
    vector_db = FAISS.load_local(VECTORDB_PATH, embeddings, allow_dangerous_deserialization=True)
    return vector_db

load_dotenv()
vector_store = get_vectorstore()


def main():
    st.title("TDX Chatbot")

    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Please enter your query")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

    try:
        result = chatbot.query_chatbot(prompt, vector_store)
        st.chat_message('assistant').markdown(result)
        st.session_state.messages.append({'role':'assistant', 'content': result})
    except Exception as e:
        st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()