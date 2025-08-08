# rag/vector_store.py
import numpy as np
import faiss
import json
from pathlib import Path
from typing import List, Dict, Optional,Callable
import pickle
import time

class VectorStore:
    def __init__(self, dimension: int = 384, persist_dir: Optional[Path] = None):
        self.dimension = dimension
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(dimension))
        self.metadata_map = {}  # {doc_id: metadata}
        self.next_id = 0
        self.persist_dir = persist_dir
        
    def add_documents(self, embeddings: np.ndarray, documents: List[Dict]):
        """Add documents with metadata to the vector store"""
        ids = np.arange(self.next_id, self.next_id + len(embeddings))
        self.index.add_with_ids(embeddings, ids)
        
        # Store metadata with ID mapping
        for doc_id, doc in zip(ids, documents):
            self.metadata_map[int(doc_id)] = {
                "content": doc["content"],
                "metadata": doc.get("metadata", {})
            }
        
        self.next_id += len(embeddings)
    
    def save(self, path: Path):
        """Save the vector store to disk"""
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        
        # Save metadata separately
        with open(path / "metadata.pkl", 'wb') as f:
            pickle.dump(self.metadata_map, f)
    
    @classmethod
    def load(cls, path: Path):
        """Load vector store from disk"""
        if not path.exists():
            raise FileNotFoundError(f"Vector store directory {path} does not exist")
            
        # Load FAISS index
        index = faiss.read_index(str(path / "index.faiss"))
        
        # Load metadata (changed from documents.pkl to metadata.pkl)
        with open(path / "metadata.pkl", 'rb') as f:
            metadata_map = pickle.load(f)
        
        # Create new instance
        store = cls(dimension=index.d)
        store.index = index
        store.metadata_map = metadata_map
        store.next_id = max(metadata_map.keys()) + 1 if metadata_map else 0
        
        return store

    def similarity_search(
        self, 
        query_embedding: np.ndarray, 
        k: int = 3,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> List[Dict]:
        
        print("start similarity search.....")
         # Add strict validation
        try:
            assert isinstance(query_embedding, np.ndarray), "Query must be numpy array"
            assert query_embedding.dtype == np.float32, f"Expected float32, got {query_embedding.dtype}"
            assert query_embedding.shape == (self.dimension,), \
                f"Query shape mismatch. Expected ({self.dimension},), got {query_embedding.shape}"
            print("[Validation] All input checks passed")
        except AssertionError as e:
            print(f"[Validation Failed] {str(e)}")
            raise

        """Enhanced search with progress reporting"""
        try:
            query_embedding = query_embedding.reshape(1, -1).astype('float32')
            print(f"[Reshaped] Query now: {query_embedding.shape}, {query_embedding.dtype}")
        except Exception as e:
            print(f"[Reshape Error] {str(e)}")
            raise

        # FAISS search with progress reporting
        if progress_callback:
            progress_callback(10)  # Initial progress
            try:
                distances, indices = self.index.search(query_embedding, k)
                progress_callback(100)
            except Exception as e:
                progress_callback(0)
                raise
        
        # Process results (existing code)
        results = []
        try:
            print("[Processing] Building results...")
            for i, idx in enumerate(indices[0]):
                if idx >= 0:
                    print(f"[Processing] Index {i}: doc_id {idx} (distance: {distances[0][i]:.4f})", end=" ")
                    if idx in self.metadata_map:
                        doc_data = self.metadata_map[int(idx)]
                        results.append({
                            "content": doc_data["content"],
                            "score": float(distances[0][i]),
                            "metadata": doc_data["metadata"]
                        })
                        print("✅ Added to results")
                    else:
                        print("❌ Missing metadata")
                else:
                    print(f"[Processing] Index {i}: Invalid index ({idx})")

            print(f"[Processing] Final results count: {len(results)}")
        except Exception as e:
            print(f"[Processing Error] {str(e)}")
            raise

        print("=== DEBUG END ===")
        
        return results


    def _debug_index_search(self, query_embedding: np.ndarray, k: int):
        """Debug method for FAISS searches (now properly takes self)"""
        print("\n=== DEBUG: FAISS Index ===")
        print(f"Index type: {type(self.index)}")
        print(f"Total vectors: {self.index.ntotal}")
        print(f"Dimension: {self.index.d}")
        print(f"Is trained: {self.index.is_trained}")
        
        print(f"\nQuery shape: {query_embedding.shape}")
        print(f"Query dtype: {query_embedding.dtype}")
        print(f"Sample values: {query_embedding[0, :5]}...")
        
        print("\nTrying small search (k=1)...")
        distances, indices = self.index.search(query_embedding, 1)
        print(f"Small search success - distances: {distances}, indices: {indices}")
        
        print(f"\nTrying full search (k={k})...")
        distances, indices = self.index.search(query_embedding, k)
        print(f"Search successful - distances shape: {distances.shape}, indices shape: {indices.shape}")
        
        return distances, indices

def test_similarity_search():
    """Standalone test function for VectorStore's similarity_search"""
    
    # Create a test instance of the actual VectorStore class
    searcher = VectorStore(dimension=384)
    
    # Create test data
    np.random.seed(42)
    test_embeddings = np.random.rand(100, 384).astype('float32')
    test_documents = [
        {"content": f"Test document {i}", "metadata": {"doc_id": i}}
        for i in range(100)
    ]
    
    # Add documents to the index
    searcher.add_documents(test_embeddings, test_documents)
    
    # Define progress callback
    def progress_callback(progress: float):
        print(f"Progress: {progress:.1f}%")
    
    print("\n=== Starting similarity_search tests ===")
    
    try:
        # Test 1: Basic search
        print("\n[Test 1] Basic search (k=3)")
        query = np.random.rand(384).astype('float32')
        results = searcher.similarity_search(query, k=3)
        print(f"Found {len(results)} results")
        for i, res in enumerate(results, 1):
            print(f"{i}. Score: {res['score']:.4f} | Content: {res['content'][:50]}...")
        
        # Test 2: With progress callback
        print("\n[Test 2] With progress callback")
        results = searcher.similarity_search(query, k=3, progress_callback=progress_callback)
        
        # Test 3: Edge case (k > ntotal)
        print("\n[Test 3] k > total documents")
        results = searcher.similarity_search(query, k=150)
        print(f"Returned {len(results)} results (max possible is 100)")
        
        # Test 4: Empty query
        print("\n[Test 4] Empty query vector")
        empty_query = np.zeros(384, dtype='float32')
        results = searcher.similarity_search(empty_query, k=3)
        
        print("\n=== All tests passed successfully ===")
        return True
    
    except Exception as e:
        print(f"\n!!! Test failed: {str(e)}")
        return False
        
def main():
    test_similarity_search()

if __name__ == "__main__":
    main()
