import traceback
import sqlite3
import os
import streamlit as st
from langchain_community.vectorstores import FAISS
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
        chunk_size=1000
    )
    vector_db = FAISS.load_local(VECTORDB_PATH, embeddings, allow_dangerous_deserialization=True)
    return vector_db

load_dotenv()
vector_store = get_vectorstore()

# Initialize SQLite database
conn = sqlite3.connect('conversations.db')
c = conn.cursor()
c.execute('''
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
''')
c.execute('''
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    sender TEXT,
    message TEXT,
    FOREIGN KEY (conversation_id) REFERENCES conversations (id)
)
''')
conn.commit()

# Function to save conversation and message to the database
def save_conversation(topic):
    c.execute('''
    INSERT INTO conversations (topic)
    VALUES (?)
    ''', (topic,))
    conn.commit()
    return c.lastrowid

def save_message(conversation_id, sender, message):
    c.execute('''
    INSERT INTO messages (conversation_id, sender, message)
    VALUES (?, ?, ?)
    ''', (conversation_id, sender, message))
    conn.commit()

# Function to delete a conversation
def delete_conversation(conversation_id):
    c.execute('DELETE FROM messages WHERE conversation_id = ?', (conversation_id,))
    c.execute('DELETE FROM conversations WHERE id = ?', (conversation_id,))
    conn.commit()


def main():
    st.title("TDX Chatbot")

    # Start new conversation
    new_conversation = st.sidebar.text_input("New Conversation Topic")
    if st.sidebar.button("Start New Conversation"):
        if new_conversation:
            conversation_id = save_conversation(new_conversation)
            st.session_state.conversation_id = conversation_id
            st.session_state.topic = new_conversation
            st.session_state.messages = []

    # Delete conversation
    conversation_to_delete = st.sidebar.selectbox(
        "Select Conversation to Delete",
        [None] + [row[0] for row in c.execute('SELECT topic FROM conversations').fetchall()])

    if st.sidebar.button("Delete Selected Conversation"):
        if conversation_to_delete:
            conversation_id_to_delete = c.execute(
                'SELECT id FROM conversations WHERE topic = ?', (conversation_to_delete,)).fetchone()[0]
            delete_conversation(conversation_id_to_delete)

    # Display past conversations
    st.sidebar.write("---")
    st.sidebar.write("Common TDX FAQ's")
    past_conversations = c.execute('SELECT id, topic, timestamp FROM conversations ORDER BY timestamp DESC').fetchall()
    for conversation_id, topic, timestamp in past_conversations:
        if st.sidebar.button(f"{topic}", key=f"load_{conversation_id}"):
            st.session_state.conversation_id = conversation_id
            st.session_state.topic = topic
            st.session_state.messages = [
                {'role': row[0], 'content': row[1]}
                for row in c.execute(
                'SELECT sender, message FROM messages WHERE conversation_id = ? ORDER BY timestamp',
                (conversation_id,)).fetchall()
            ]

    # Main chat area
    if 'conversation_id' not in st.session_state:
        st.session_state.conversation_id = None
        st.session_state.topic = None
        st.session_state.messages = []

    if st.session_state.conversation_id:
        st.header(f"Topic: {st.session_state.topic}")

        for msg in st.session_state.messages:
            st.chat_message(msg['role'].lower()).markdown(msg['content'])

        prompt = st.chat_input("Please enter your query")

        if prompt:
            if not prompt.strip():
                st.error("Query cannot be empty. Please enter a valid query.")
                return

            st.chat_message('user').markdown(prompt)
            st.session_state.messages.append({'role': 'user', 'content': prompt})
            save_message(st.session_state.conversation_id, "User", prompt)

            try:
                result = chatbot.query_chatbot(prompt, vector_store)
                st.chat_message('assistant').markdown(result)
                st.session_state.messages.append({'role': 'assistant', 'content': result})
                save_message(st.session_state.conversation_id, "assistant", result)
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.error(traceback.format_exc())

    else:
        st.header("Start a new conversation or select an existing one from the sidebar.")


if __name__ == "__main__":
    main()