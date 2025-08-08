from abc import ABC, abstractmethod
from utils.api_client import DeepSeekClient
from utils.api_client2 import GroqClient
from typing import Union
import json


class BaseAgent(ABC):
    def __init__(self, name, role):
        self.name = name
        self.role = role
        #self.api = GroqClient()
        self.api = DeepSeekClient()    
        
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

    # def call_llm(self, prompt,  require_json=False, **kwargs):
    #     max_context = 128000  # DeepSeek-V3's limit
    #     prompt_token_estimate = len(prompt) // 4  # Rough estimate (1 token ≈ 4 chars)
    #     remaining_tokens = max_context - prompt_token_estimate
        
    #     kwargs.setdefault('max_tokens', min(remaining_tokens, 4000))  # Cap at 4K unless needed
    #     kwargs.setdefault('temperature', 0.2)
    #     if require_json:
    #         kwargs['response_format'] = {"type": "json_object"}

    #     return self.api.call(prompt, **kwargs)



    # def call_llm(self, prompt: str, require_json: bool = False, **kwargs) -> Union[str, dict]:
   
    #     # Set common parameters
    #     kwargs.setdefault('temperature', 0.7)
    #     kwargs.setdefault('max_tokens', 1000)
        
    #     try:
    #         # Get raw response
    #         raw_response = self.api.call(prompt, **kwargs)
            
    #         # If JSON was requested
    #         if require_json:
    #             if isinstance(raw_response, dict):  # If API natively returns JSON
    #                 return raw_response
    #             try:
    #                 # Try direct parse first
    #                 return json.loads(raw_response)
    #             except json.JSONDecodeError:
    #                 # Clean markdown if present
    #                 clean_response = raw_response.replace('```json', '').replace('```', '').strip()
    #                 return json.loads(clean_response)
            
    #         return raw_response
            
    #     except Exception as e:
    #         print(f"LLM call failed: {str(e)}")
    #         raise
        

