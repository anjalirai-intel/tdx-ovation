import asyncio
import traceback

try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

try:
    import fiximports
except ImportError:
    pass

import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
import chatbot
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings

VECTORDB_PATH = "local_db"

@st.cache_resource
def get_vectorstore():
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    embeddings = AzureOpenAIEmbeddings(
        model=os.environ["OPENAI_EMBEDDING_MODEL"],
        api_key=os.environ["AZURE_OPENAI_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_URL"],
        api_version=os.environ["OPENAI_EMBEDDING_VERSION"],
        chunk_size=100
    )
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
        if not prompt.strip():
            st.error("Query cannot be empty. Please enter a valid query.")
            return

        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        try:
            result = chatbot.query_chatbot(prompt, vector_store)
            st.chat_message('assistant').markdown(result)
            st.session_state.messages.append({'role':'assistant', 'content': result})
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.error(traceback.format_exc())


if __name__ == "__main__":
    main()