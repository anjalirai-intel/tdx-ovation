import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


load_dotenv()

os.environ["https_proxy"] = "http://proxy-dmz.intel.com:912"
os.environ["http_proxy"] = "http://proxy-dmz.intel.com:911"

VECTORDB_PATH = "local_db"
DATA_FOLDER = "data"
HF_TOKEN=os.environ.get("HF_TOKEN")

PROMPT_TEMPLATE = """
    You are a very helpful assistant which helps customer resolves their query related to
    TDX(Trust Domain Extensions). With your help customer can ramp up quickly. 
    If you don't know the answer, just say that you are not aware of it and if they would
    like to send email to respective person for further support

    Context: {context}
    Question: {question}
"""

def load_pdf_files():
    loader = DirectoryLoader(DATA_FOLDER, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

def create_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # embedding_model = OpenAIEmbeddings(
    #     model="text-embedding-ada-002",
    #     openai_api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    #     openai_api_base=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    #     openai_api_type="azure",
    #     openai_api_version="2023-03-15-preview"  # Ensure this matches your Azure deployment
    # )
    vector_db = FAISS.from_documents(text_chunks, embeddings)
    vector_db.save_local(VECTORDB_PATH)
    # vector_db = Chroma.from_documents(text_chunks, embeddings, persist_directory=VECTORDB_PATH)
    # vector_db.persist()

def custom_prompt(template):
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    return prompt

def load_llm():
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        task="text-generation",
        temperature=0.5,
        model_kwargs={
            "token": HF_TOKEN,
            "max_length": 512
        }
    )
    return llm

def query_chatbot(prompt, vector_store):
    qa_chain = RetrievalQA.from_chain_type(
        llm=load_llm(),
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=False,
        chain_type_kwargs={'prompt': custom_prompt(PROMPT_TEMPLATE)}
    )

    response = qa_chain.invoke({'query': prompt})
    return response['result']