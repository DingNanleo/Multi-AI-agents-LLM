from abc import ABC, abstractmethod
from utils.api_client import DeepSeekClient
from utils.api_client2 import GroqClient

class BaseAgent(ABC):
    def __init__(self, name, role):
        self.name = name
        self.role = role
        self.api = DeepSeekClient()
        #self.api = GroqClient()
        
    @abstractmethod
    def perform_task(self, *args, **kwargs):
        pass
        
    def call_llm(self, prompt, **kwargs):
        # Enforce default limits if not specified
        kwargs.setdefault('max_tokens', 2000)  # Default limit
        kwargs.setdefault('temperature', 0.0)
    
        # Truncate prompt if too long (avoid API errors)
        truncated_prompt = prompt[:8000] if len(prompt) > 8000 else prompt
        return self.api.call(truncated_prompt, **kwargs)