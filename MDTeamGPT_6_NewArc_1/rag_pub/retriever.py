# rag/retriever.py
from typing import List, Dict
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from .vector_store import VectorStore
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

@classmethod
def load(cls, persist_dir: str):
    """Load a pre-built vector store and create a retriever"""
    vector_store = VectorStore.load(persist_dir)
    return cls(vector_store)


class MedicalRetriever:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.bm25 = None
        self.corpus = []
        
    def initialize_bm25(self, documents: List[str]):
        """Initialize BM25 with tokenized documents"""
        if not documents:
            logger.warning("No documents provided for BM25 initialization")
            return
            
        self.corpus = documents
        tokenized_corpus = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to 0-1 range"""
        if np.max(scores) - np.min(scores) == 0:
            return np.ones_like(scores)
        return (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

    def hybrid_retrieve(self, query: str, top_k: int = 3,similarity_threshold:float=None) -> List[Dict]:
        """
        Hybrid retrieval combining semantic and keyword search with score fusion
        """
        print(f"\n=== HYBRID RETRIEVAL DEBUG ===")
        print(f"Initial Query: '{query}'")

        def progress_callback(percent: float):
            print(f"Search progress: {percent:.0f}%", end='\r')

        # Semantic search focuses on the meaning behind the query rather than just matching keywords.
        # uses an embedder (likely a neural network like Sentence-BERT) to convert the query into a dense vector (embedding).
        # The system then compares this embedding against pre-stored document embeddings 
        # in the vector database (e.g., FAISS, Pinecone) using cosine similarity or other distance metrics.
        print("\n[1/3] Starting semantic searching ...")
        try:
            query_embedding = self.embedder.encode(query)
            print(f"Query embedding shape: {query_embedding.shape}")
            semantic_results = self.vector_store.similarity_search(query_embedding, k=top_k, progress_callback=progress_callback)
            print(f"Found {len(semantic_results)} semantic results")
            if semantic_results:
                print(f"Top semantic result: {semantic_results[0]['content'][:100]}... (score: {semantic_results[0]['score']:.2f})")
        except Exception as e:
            print(f"Semantic search failed: {str(e)}")
            semantic_results = []
    

        print("\n[2/3] Starting keyword search...")
        # Keyword search (BM25) Keyword search compares text (tokens).
        # BM25 is a probabilistic ranking algorithm for keyword-based search. 
        # It scores documents based on:
        # Term frequency (how often keywords appear in the document)
        # Inverse document frequency (how rare the keywords are across all documents)
        # Document length normalization (shorter docs get a slight boost)
        # In your code, the query is split into tokens (words), and BM25 calculates scores for each document in the corpus.
        keyword_results = []
        if self.bm25:
            try:
                tokenized_query = query.split()
                print(f"Tokenized query: {tokenized_query}")
                bm25_scores = self.bm25.get_scores(tokenized_query)
                top_indices = np.argsort(bm25_scores)[-top_k*2:][::-1]
                print(f"Top BM25 indices: {top_indices}")

                keyword_results = [{
                    "content": self.corpus[i],
                    "score": float(bm25_scores[i]),
                    "metadata": {"source": "BM25"}  # Placeholder
                } for i in top_indices if i < len(self.corpus)]

                print(f"Found {len(keyword_results)} keyword results")
                if keyword_results:
                    print(f"Top keyword result: {keyword_results[0]['content'][:100]}... (score: {keyword_results[0]['score']:.2f})")
            except Exception as e:
                print(f"Keyword search failed: {str(e)}")
        else:
            print("BM25 not initialized - skipping keyword search")
    
        # Combine and rerank results
        print("\n[3/3] Combining and ranking results...")
        all_results = semantic_results + keyword_results
        print(f"Total combined results before dedupe: {len(all_results)}")
    
        # Group by content to deduplicate
        content_map = defaultdict(list)
        for result in all_results:
            content_map[result["content"]].append(result)
        print(f"Unique results after dedupe: {len(content_map)}")

        # Fusion scoring: average of normalized scores
        final_results = []
        for content, matches in content_map.items():
            semantic_score = next((r["score"] for r in matches if "metadata" in r and r["metadata"].get("source") != "BM25"), 0)
            keyword_score = next((r["score"] for r in matches if "metadata" in r and r["metadata"].get("source") == "BM25"), 0)
            
            # Normalize and combine scores
            norm_semantic = self.normalize_scores(np.array([semantic_score]))[0]
            norm_keyword = self.normalize_scores(np.array([keyword_score]))[0]
            combined_score = (norm_semantic + norm_keyword) / 2
            
            final_results.append({
                "content": content,
                "score": combined_score,
                "metadata": matches[0].get("metadata", {}),
                "scores": {
                    "semantic": float(semantic_score),
                    "keyword": float(keyword_score)
                }
            })
        
        # Sort by combined score
        final_results.sort(key=lambda x: x["score"], reverse=True)
        print("\nTop 3 final results:")
        for i, res in enumerate(final_results[:3], 1):
            print(f"{i}. Score: {res['score']:.2f} (Semantic: {res['scores']['semantic']:.2f}, Keyword: {res['scores']['keyword']:.2f})")
            print(f"   Content: {res['content'][:100]}...")
            print(f"   Source: {res['metadata'].get('source', 'Unknown')}\n")
        

        if similarity_threshold is not None:
            final_results = [res for res in final_results if res['score'] >= similarity_threshold]
    
        print("\nSearch complete!")
        return final_results[:top_k]

    def retrieve_medical_context(self, question: str) -> Dict:
        """Retrieve context with source filtering"""
        results = self.hybrid_retrieve(question, top_k=5)
        
        # Organize by source
        medline = [r for r in results if r["metadata"].get("source") == "MedlinePlus"]
        pubmed = [r for r in results if r["metadata"].get("source") == "PubMedQA"]
        
        return {
            "medline_knowledge": medline[:2],
            "pubmed_evidence": pubmed[:1],
            "other_context": [r for r in results if r not in medline + pubmed]
        }