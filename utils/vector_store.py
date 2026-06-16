import os
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter

def create_vector_store(document_text, embeddings):
    """
    Split document into chunks, embed each chunk,
    and store in a FAISS vector store.
    """
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(document_text)
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store

def save_vector_store(vector_store, folder_path):
    """Save FAISS vector store to disk."""
    os.makedirs(folder_path, exist_ok=True)
    vector_store.save_local(folder_path)

def load_vector_store(folder_path, embeddings):
    """Load FAISS vector store from disk."""
    vector_store = FAISS.load_local(
        folder_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vector_store