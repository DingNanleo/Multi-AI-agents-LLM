# rag/embedder.py
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class MedicalEmbedder:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the embedder with a pretrained model"""
        self.model = SentenceTransformer(model_name)
        
    def embed_documents(self, documents: List[str], batch_size: int = 8) -> np.ndarray:
        """
        Embed a list of documents with batch processing
        
        Args:
            documents: List of text documents to embed
            batch_size: Number of documents to process at once (default: 8)
            
        Returns:
            numpy.ndarray: Array of document embeddings
        """
        return self.model.encode(
            documents,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query
        
        Args:
            query: Text query to embed
            
        Returns:
            numpy.ndarray: Query embedding vector
        """
        return self.model.encode(query, convert_to_numpy=True)