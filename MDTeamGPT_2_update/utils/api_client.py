import os
import time
import json
import requests
from dotenv import load_dotenv
from urllib.parse import urljoin
from typing import Optional

load_dotenv()

class DeepSeekClient:
    def __init__(self, max_retries: int = 5, initial_timeout: int = 15):
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.base_url = os.getenv("DEEPSEEK_API_URL")
        self.max_retries = max_retries
        self.initial_timeout = initial_timeout
        
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment variables")
        if not self.base_url:
            raise ValueError("DEEPSEEK_API_URL not found in environment variables")
        
        if not self.base_url.endswith('/'):
            self.base_url += '/'

    def call(self, prompt: str, model: str = "deepseek-chat", temperature: float = 0.7, max_tokens: int = 1000) -> Optional[str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": min(max_tokens, 1000),  # Enforce lower limit
            "stream": False  # Disabled for compatibility
        }
        
        endpoint = urljoin(self.base_url, "v1/chat/completions")
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
                
                # Handle both streaming and non-streaming responses
                if 'text/event-stream' in response.headers.get('content-type', ''):
                    return self._handle_streaming_response(response)
                return response.json()["choices"][0]["message"]["content"]
                
            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = min(2 ** attempt, 60)  # Exponential backoff with 60s cap
                    print(f"Attempt {attempt + 1}/{self.max_retries} failed. Retrying in {wait_time}s... Error: {str(e)}")
                    time.sleep(wait_time)
        
        raise Exception(f"API call failed after {self.max_retries} attempts. Last error: {str(last_error)}")

    def _handle_streaming_response(self, response) -> str:
        """Handle streaming responses if enabled"""
        full_content = ""
        for line in response.iter_lines():
            if line:
                decoded = line.decode('utf-8')
                if decoded.startswith('data:'):
                    try:
                        data = json.loads(decoded[5:])
                        if 'choices' in data:
                            full_content += data['choices'][0].get('delta', {}).get('content', '')
                    except json.JSONDecodeError:
                        continue
        return full_content