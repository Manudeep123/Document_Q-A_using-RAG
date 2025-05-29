from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
import os

def create_vector_store(docs, storage_type="faiss", persist_dir=None):
    """Create a vector store from documents."""
    embeddings = OpenAIEmbeddings()
    
    if storage_type.lower() == "faiss":
        vector_store = FAISS.from_documents(docs, embeddings)
        if persist_dir:
            vector_store.save_local(persist_dir)
    elif storage_type.lower() == "chroma":
        vector_store = Chroma.from_documents(
            docs, embeddings, persist_directory=persist_dir
        )
    else:
        raise ValueError("Invalid storage type. Choose 'faiss' or 'chroma'.")
    
    return vector_store  # Return the created vector store

def load_vector_store(storage_type="faiss", persist_dir=None):
    """Load an existing vector store from disk."""
    if not persist_dir:
        raise ValueError("persist_dir must be provided to load a vector store.")
    
    embeddings = OpenAIEmbeddings()
    
    if storage_type.lower() == "faiss":
        return FAISS.load_local(persist_dir, embeddings)
    elif storage_type.lower() == "chroma":
        return Chroma(
            persist_directory=persist_dir, 
            embedding_function=embeddings
        )
    else:
        raise ValueError("Invalid storage type. Choose 'faiss' or 'chroma'.")