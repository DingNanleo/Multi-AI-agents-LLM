# retriever.py
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
import re

class EvidenceRetriever:
    def __init__(self, index_path="index.faiss", pkl_path="index.pkl", embedding_model="sentence-transformers/all-mpnet-base-v2"):
        """
        Initialize the EvidenceRetriever with FAISS index and document store.
        """
        # 1. First verify embedding model
        print(f"🔍 Using embedding model: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        # Test embedding dimensions
        test_embedding = self.embeddings.embed_query("test")
        print(f"📏 Embedding dimension: {len(test_embedding)}")
        
        # 2. Verify index files
        if not os.path.exists(index_path) or not os.path.exists(pkl_path):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            index_path = os.path.join(current_dir, index_path)
            pkl_path = os.path.join(current_dir, pkl_path)
            
            if not os.path.exists(index_path) or not os.path.exists(pkl_path):
                raise FileNotFoundError(f"Index files not found at {index_path} and {pkl_path}")
        
        print(f"📂 Loading index from: {index_path}")
        print(f"📄 Loading metadata from: {pkl_path}")
        
        # 3. Load with additional checks
        try:
            self.db = FAISS.load_local(
                folder_path=os.path.dirname(index_path),
                embeddings=self.embeddings,
                index_name=os.path.basename(index_path).replace(".faiss", ""),
                allow_dangerous_deserialization=True
            )
            print("✅ Index loaded successfully")
            # Verify index dimensions
            if hasattr(self.db, 'index') and hasattr(self.db.index, 'd'):
                print(f"🔢 Index dimension: {self.db.index.d}")
        except Exception as e:
            print(f"❌ Failed to load index: {str(e)}")
            raise
    
    def _format_evidence_content(self, content):
        """Format the evidence content for better readability in prompts"""
        # 1. Remove excessive line breaks and normalize whitespace
        # print("\n=== 格式化前原始内容 ===")
        # print(content)

        content = ' '.join(content.split())
        
        # 2. Fix bullet points and lists formatting
        content = re.sub(r'(\w)\n(\w)', r'\1 \2', content)  # Join broken lines
        content = re.sub(r'\.\s([A-Z])', r'. \1', content)  # Add space after period before capital
        
        # 3. Remove redundant section headers if needed
        content = re.sub(r'SECTION\s+\d+:\s*', '', content, flags=re.IGNORECASE)
        
        # 4. Ensure proper sentence spacing
        content = re.sub(r'\.([A-Za-z])', r'. \1', content)
        
        # print("\n=== 格式化后内容 ===")
        # print(content)

        return content.strip()

    def _format_evidence_output(self, evidence_list):
        """Convert the list of evidence dictionaries into formatted text blocks"""
        formatted_output = ""
        for i, evidence in enumerate(evidence_list, 1):
            formatted_output += (
                f"Evidence {i} (Confidence: {evidence['confidence']:.2f}, "
                f"Source: {evidence['source']}):\n"
                f"{evidence['content']}\n\n"
            )
        return formatted_output.strip()

    def retrieve_evidence(self, question, specialty=None, k=3):
        """Retrieve relevant medical evidence."""
        try:
            # Verify question embedding
            question_embedding = self.embeddings.embed_query(question)
            print(f"\n{'='*20} Question embedding dimension: {len(question_embedding)}{'='*20}")
            
            results = self.db.similarity_search(question, k=k*5)
            print(f"Found {len(results)} Medical Knowledge Chunk(Before {specialty} Filtering)")
            
            if specialty:
                results = [doc for doc in results 
                         if doc.metadata.get('specialty', '').lower() == specialty.lower()]
                print(f"Found {len(results)} Medical Knowledge Chunk(After{specialty} Filtering)")

            # return [{
            #     "content": doc.page_content,
            #     "source": doc.metadata.get('source', 'Unknown'),
            #     "specialty": doc.metadata.get('specialty', 'Unknown'),
            #     "confidence": 1/(i+1)
            # } for i, doc in enumerate(results[:k])]
            # Format each document's content before returning

            # Format each document's content
            evidence_list = []
            for i, doc in enumerate(results[:k]):
                formatted_content = self._format_evidence_content(doc.page_content)
                evidence_list.append({
                    "content": formatted_content,
                    "source": doc.metadata.get('source', 'Unknown'),
                    "specialty": doc.metadata.get('specialty', 'Unknown'),
                    "confidence": 1/(i+1)  # Simple confidence score based on rank
                })
            print(f"Return {len(evidence_list)} Medical Knowledge Chunk ")  
            # Return in requested format
            return self._format_evidence_output(evidence_list)
            
        except Exception as e:
            print(f"❌ Retrieval error: {str(e)}")
            raise

if __name__ == "__main__":
    # 1. First confirm where your index files are located
    print("Current working directory:", os.getcwd())
    
    # 2. Try with explicit paths
    retriever = EvidenceRetriever(
        index_path="data/EM_textbook/FAISS_index/index.faiss",
        pkl_path="data/EM_textbook/FAISS_index/index.pkl",
        embedding_model="BAAI/bge-small-en-v1.5"  # Must match original
    )
    
    # Example query
    question = "A 35-year-old man arrives at the emergency department within minutes after a head-on motor vehicle accident. He suffered from blunt abdominal trauma, several lacerations to his face as well as lacerations to his upper and lower extremities. The patient is afebrile, blood pressure is 45/25 mmHg and pulse is 160/minute. A CBC is obtained and is most likely to demonstrate which of the following?"
    #specialty = "Trauma Surgery"
    specialty = "Emergency" 
    print(f"\nTesting with query: '{question}'")
    
    #results = retriever.retrieve_evidence(question=question, k=1)
    results = retriever.retrieve_evidence(question=question, specialty=specialty, k=3)
    print(f"Found {len(results)} results")
    print(results)

