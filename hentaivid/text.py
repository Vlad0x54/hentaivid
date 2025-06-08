# hentaivid/text.py 
import os
from typing import List
from sentence_transformers import SentenceTransformer
from loguru import logger

def load_text(file_path: str) -> str:
    """Loads text from a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return ""
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {e}")
        return ""

def chunk_text(text: str, chunk_size: int = 256, overlap: int = 32) -> List[str]:
    """Splits text into chunks of a specified size with overlap."""
    if not text:
        return []
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def get_embeddings(chunks: List[str], model_name: str = "all-MiniLM-L6-v2") -> List[List[float]]:
    """Generates embeddings for a list of text chunks."""
    if not chunks:
        return []
    
    logger.info(f"Loading sentence-transformer model: {model_name}")
    model = SentenceTransformer(model_name)
    logger.info("Generating embeddings...")
    embeddings = model.encode(chunks, show_progress_bar=True)
    return embeddings.tolist() 