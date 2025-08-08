from abc import ABC, abstractmethod
from utils.api_client import DeepSeekClient
from utils.api_client2 import GroqClient
from typing import Union, Dict, Optional
import json
from rag_pub.retriever import MedicalRetriever

class BaseAgent(ABC):
    def __init__(self, name: str, role: str, historical_pool=None, retriever: Optional[MedicalRetriever] = None):
        self.name = name
        self.role = role
        #self.api = GroqClient()
        self.api = DeepSeekClient()
        self.historical_pool = historical_pool
        self.retriever = retriever
        
    @abstractmethod
    def perform_task(self, *args, **kwargs):
        pass
    
    def call_llm(self, prompt, require_json=False, **kwargs):
        max_context = 128000
        prompt_token_estimate = len(prompt) // 4
        remaining_tokens = max_context - prompt_token_estimate
        
        kwargs.setdefault('max_tokens', min(remaining_tokens, 4000))
        kwargs.setdefault('temperature', 0.2)
        
        # Add require_json handling
        response = self.api.call(
            prompt,
            require_json=require_json,
            **kwargs
        )
        
        # Handle error responses
        if isinstance(response, str) and response.startswith('{"error":'):
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                pass
        return response

    def retrieve_medical_context(self, query: str) -> Dict:
        """Enhanced retrieval with better context organization"""
        if not self.retriever:
            return {
                "context": "",
                "sources": [],
                "retrieval_scores": []
            }
            
        results = self.retriever.retrieve_medical_context(query)
        
        # Format context sections
        context_parts = []
        if results["medline_knowledge"]:
            context_parts.append("## General Medical Knowledge\n" + 
                               "\n".join([r["content"] for r in results["medline_knowledge"]]))
        
        if results["pubmed_evidence"]:
            context_parts.append("## Research Evidence\n" + 
                               "\n".join([r["content"] for r in results["pubmed_evidence"]]))
        
        if results["other_context"]:
            context_parts.append("## Additional Context\n" + 
                               "\n".join([r["content"] for r in results["other_context"]]))
        
        return {
            "context": "\n\n".join(context_parts),
            "sources": [r["metadata"] for r in results["medline_knowledge"] + results["pubmed_evidence"]],
            "retrieval_scores": [r["score"] for r in results["medline_knowledge"] + results["pubmed_evidence"]]
        }





# import torch

# class BaseAgent:
#     def __init__(self):
#         self.model_name = "medalpaca/medalpaca-7b"
#         self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
#         self.model = AutoModelForCausalLM.from_pretrained(
#             self.model_name,
#             torch_dtype=torch.float16,  # FP16 for efficiency
#             device_map="auto"            # Auto GPU/CPU placement
#         )
    
#     def generate_response(self, prompt):
#         inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
#         outputs = self.model.generate(**inputs, max_new_tokens=200)
#         return self.tokenizer.decode(outputs[0], skip_special_tokens=True)



