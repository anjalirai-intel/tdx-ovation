import os
import chatbot

if not os.path.exists(chatbot.VECTORDB_PATH):
    extracted_data = chatbot.load_pdf_files()
    chunk_data = chatbot.create_chunks(extracted_data)
    chatbot.create_vector_store(chunk_data)