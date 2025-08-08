# utils/deepseek_embedder.py
import requests
import numpy as np
from typing import List

class DeepSeekEmbedder:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com/v1/embeddings"
        
    def get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings from DeepSeek API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "input": texts,
            "model": "deepseek-embedding"  # Confirm the exact model name
        }
        
        response = requests.post(self.base_url, json=payload, headers=headers)
        response.raise_for_status()
        
        embeddings = [np.array(item['embedding']) for item in response.json()['data']]
        return embeddings