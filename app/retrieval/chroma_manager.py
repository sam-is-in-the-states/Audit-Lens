from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def get_vector_db(path):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return Chroma(
        persist_directory=path,
        embedding_function=embeddings
    )
