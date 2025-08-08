import os
import time
import requests
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

class GroqClient:
    def __init__(self, max_retries: int = 5, initial_timeout: int = 15):
        self.api_key = os.getenv("Groqcloud_API_KEY")
        self.base_url = os.getenv("Groqcloud_API_URL")
        self.max_retries = max_retries
        self.initial_timeout = initial_timeout
        
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")

    def call(self, prompt: str, model: str = "llama3-70b-8192", temperature: float = 0.7, max_tokens: int = 1000) -> Optional[str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 1,
            "stream": False,
            "stop": None
        }
        
        endpoint = f"{self.base_url}chat/completions"
        last_error = None
        
        for attempt in range(self.max_retries):
            current_timeout = self.initial_timeout * (attempt + 1)
            try:
                response = requests.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                    timeout=current_timeout
                )
                response.raise_for_status()
                
                data = response.json()
                return data["choices"][0]["message"]["content"]
                
            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = min(2 ** attempt, 60)
                    print(f"Attempt {attempt + 1}/{self.max_retries} failed. Retrying in {wait_time}s... Error: {str(e)}")
                    time.sleep(wait_time)
                else:
                    # Print detailed error info for debugging
                    if hasattr(e, 'response') and e.response:
                        print(f"API Error Details: {e.response.text}")
        
        raise Exception(f"API call failed after {self.max_retries} attempts. Last error: {str(last_error)}")