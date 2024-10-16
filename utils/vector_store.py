# utils/vector_store.py

import os
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_vector_store(document_text):
    # Initialize Embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

    # Split the document into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(document_text)

    # Create embeddings and store in FAISS
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store

def save_vector_store(vector_store, folder_path):
    """
    Save the vector store to disk using FAISS's built-in methods.

    Args:
        vector_store (FAISS): The vector store to save.
        folder_path (str): The folder path where the FAISS index will be saved.
    """
    # Ensure the directory exists
    os.makedirs(folder_path, exist_ok=True)

    # Save the vector store
    vector_store.save_local(folder_path)

def load_vector_store(folder_path):
    """
    Load the FAISS vector store from disk.

    Args:
        folder_path (str): The folder path from where the FAISS index will be loaded.

    Returns:
        vector_store (FAISS): The reconstructed FAISS vector store.
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("The OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    
    # Set allow_dangerous_deserialization=True to enable loading the pickle file
    vector_store = FAISS.load_local(folder_path, embeddings, allow_dangerous_deserialization=True)
    return vector_store