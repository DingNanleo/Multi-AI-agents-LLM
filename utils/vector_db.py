# utils/vector_db.py
import json
import os
from typing import Dict, List
import numpy as np
# method 1：TfidfVectorizer  
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.metrics.pairwise import cosine_similarity
# method 2: deepseek embedding
#from embedder import DeepSeekEmbedder

class VectorDatabase:
    def __init__(
            self, 
            name: str, 
            # embedder： DeepSeekEmbedder,
            storage_path: str = "vector_db_storage"):
        self.name = name
        self.storage_path = storage_path
        #self..embedder = embedder
        #self.embeddings = []
        self.data = []
        self.vectorizer = TfidfVectorizer()
        self.vectors = None     
        os.makedirs(self.storage_path, exist_ok=True)
        self._load()

    def _load(self):
        """Load data from disk"""
        file_path = os.path.join(self.storage_path, f"{self.name}.json")
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                self.data = json.load(f)
                #self.embeddings = [np.array(embedding) for embedding in saved_data.get('embeddings', [])]
            self._update_vectors()

    def _save(self):
        """Save data to disk"""
        file_path = os.path.join(self.storage_path, f"{self.name}.json")
        with open(file_path, 'w') as f:
            json.dump(self.data, f)

    def _update_vectors(self):
        """Update vector representations of the data"""
        if self.data:
            texts = [self._get_text(record) for record in self.data]
            self.vectors = self.vectorizer.fit_transform(texts)
        else:
            self.vectors = None

    def _get_text(self, record: Dict) -> str:
        """Extract searchable text from a record"""
        return " ".join([
            str(record.get("Question", "")),
            str(record.get("Answer", "")),
            str(record.get("Reasoning", "")),
            str(record.get("Analysis_Process", ""))
        ])

    def store(self, record: Dict):
        """Store a new record in the database"""
        self.data.append(record)
        self._update_vectors()
        self._save()

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search for similar records using cosine similarity"""
        if not self.data:
            return []
            
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.vectors)
        sorted_indices = np.argsort(similarities[0])[::-1]
        
        return [self.data[i] for i in sorted_indices[:top_k]]

    def clear(self):
        """Clear all data in this database"""
        self.data = []
        self.vectors = None
        self._save()