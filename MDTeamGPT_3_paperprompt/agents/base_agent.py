from abc import ABC, abstractmethod
from utils.api_client import DeepSeekClient
from utils.api_client2 import GroqClient


class BaseAgent(ABC):
    def __init__(self, name, role):
        self.name = name
        self.role = role
        #self.api = GroqClient()
        self.api = DeepSeekClient()    
           
    @abstractmethod
    def perform_task(self, *args, **kwargs):
        pass
        
    def call_llm(self, prompt, **kwargs):
        max_context = 128000  # DeepSeek-V3's limit
        prompt_token_estimate = len(prompt) // 4  # Rough estimate (1 token ≈ 4 chars)
        remaining_tokens = max_context - prompt_token_estimate
        
        kwargs.setdefault('max_tokens', min(remaining_tokens, 4000))  # Cap at 4K unless needed
        kwargs.setdefault('temperature', 0.7)
        return self.api.call(prompt, **kwargs)
    

