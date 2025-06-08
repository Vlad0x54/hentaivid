# hentaivid/index.py 
import faiss
import numpy as np
from typing import List, Optional
from loguru import logger

def create_index(embeddings: np.ndarray) -> Optional[faiss.IndexFlatL2]:
    """Creates a FAISS index from a list of embeddings."""
    if embeddings.size == 0:
        logger.warning("Embeddings array is empty. Cannot create index.")
        return None
    
    try:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        return index
    except Exception as e:
        logger.error(f"Error creating FAISS index: {e}")
        return None 

def save_index(index: faiss.Index, file_path: str):
    """Saves a FAISS index to a file."""
    try:
        faiss.write_index(index, file_path)
        logger.info(f"FAISS index saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving FAISS index to {file_path}: {e}")

def load_index(file_path: str) -> Optional[faiss.Index]:
    """Loads a FAISS index from a file."""
    try:
        index = faiss.read_index(file_path)
        logger.info(f"FAISS index loaded from {file_path}")
        return index
    except Exception as e:
        logger.error(f"Error loading FAISS index from {file_path}: {e}")
        return None

def search_index(index: faiss.Index, query_embedding: np.ndarray, k: int) -> Optional[List[int]]:
    """Searches a FAISS index for the k-nearest neighbors."""
    if query_embedding.size == 0:
        logger.warning("Query embedding is empty. Cannot search.")
        return None
        
    try:
        distances, indices = index.search(query_embedding, k)
        return indices[0].tolist()
    except Exception as e:
        logger.error(f"Error searching FAISS index: {e}")
        return None 