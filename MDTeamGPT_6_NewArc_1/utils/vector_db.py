# utils/vector_db.py
import json
import os
from typing import Dict, List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.metrics.pairwise import cosine_similarity

class VectorDatabase:
    def __init__(
            self,
            name: str, 
            storage_path: str = "vector_db_storage"
    ):
        """
        Initialize a vector database with empty JSON files
        
        Args:
            name: Identifier for this database (CorrectKB or ChainKB)
            storage_path: Directory path for persistent storage
        """
        self.name = name
        self.storage_path = storage_path
        self.data = []
        self.vectorizer = TfidfVectorizer(
            min_df=1,
            stop_words=None,
            token_pattern=r'(?u)\b\w+\b'
        )
        self.vectors = None
        
        # Ensure storage directory exists
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Initialize empty JSON file if doesn't exist
        #self._initialize_empty_file()
        self._create_fresh_json_file()
        self._load()

    def _create_fresh_json_file(self):
        """Create new empty JSON file (simplified)"""
        file_path = os.path.join(self.storage_path, f"{self.name}.json")
        with open(file_path, 'w') as f:
            json.dump([], f)  # Empty array as initial state

    def _load(self):
        """Load data from existing JSON file"""
        file_path = os.path.join(self.storage_path, f"{self.name}.json")
        #print(f"file path:{file_path}")
        try:
            with open(file_path, 'r') as f:
                self.data = json.load(f)
            self._update_vectors()
        except Exception as e:
            print(f"Warning: Failed to load {self.name}: {str(e)}")
            self.data = []

    def _save(self):
        """Save current data to JSON file"""
        file_path = os.path.join(self.storage_path, f"{self.name}.json")
        with open(file_path, 'w') as f:
            json.dump(self.data, f, indent=2)

    def _update_vectors(self):
        """Update vector representations only if data exists"""
        if self.data:
            texts = [self._get_text(record) for record in self.data]
            self.vectors = self.vectorizer.fit_transform(texts)
        else:
            # Initialize with empty vectors but maintain the file
            self.vectors = None

    def _get_text(self, record: Dict) -> str:
        """Extract searchable text from a record"""
        return " ".join([
            str(record.get("Question", "")),
            str(record.get("Answer", "")),
            str(record.get("Reasoning", "")),
            str(record.get("Analysis_Process", ""))
        ]).strip()

    def store(self, record: Dict):
        """Store a new record and update the JSON file"""
        self.data.append(record)
        self._update_vectors()
        self._save()

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search with empty result handling"""
        if not self.data:
            return []
            
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.vectors)
        sorted_indices = np.argsort(similarities[0])[::-1]
        return [self.data[i] for i in sorted_indices[:top_k]]

    def clear(self):
        """Clear data but maintain the JSON file"""
        self.data = []
        self.vectors = None
        self._save()

    def size(self) -> int:
        return len(self.data)