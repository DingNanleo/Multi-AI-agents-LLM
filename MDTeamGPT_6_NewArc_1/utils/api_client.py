import os
import time
import json
import requests
from dotenv import load_dotenv
from urllib.parse import urljoin
from typing import Optional,Union

load_dotenv()

class DeepSeekClient:
    def __init__(self, max_retries: int = 10, initial_timeout: int = 15):
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.base_url = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/")
        self.max_retries = max_retries
        self.initial_timeout = initial_timeout
        
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment variables")
        if not self.base_url.endswith('/'):
            self.base_url += '/'


    def call(self, prompt: str, model: str = "deepseek-chat", temperature: float = 0.7,
         max_tokens: int = 1000, require_json: bool = False, **kwargs) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Parse system message if present
        messages = []
        if prompt.strip().startswith("You are a"):
            # Split into system and user messages
            parts = prompt.split("\n\n", 1)
            messages.append({"role": "system", "content": parts[0]})
            user_content = parts[1] if len(parts) > 1 else ""
        else:
            user_content = prompt
        
        if user_content:
            messages.append({"role": "user", "content": user_content})
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": max(0, min(temperature, 2)),
            "max_tokens": min(max_tokens, 4000),
            **kwargs
        }
        
        # Special handling for JSON responses
        if require_json:
            payload["response_format"] = {"type": "json_object"}
            # Ensure the first message tells the model to respond with JSON
            if messages and "respond in JSON" not in messages[0]["content"]:
                messages[0]["content"] += "\n\nIMPORTANT: Respond in valid JSON format only."
        
        endpoint = urljoin(self.base_url, "v1/chat/completions")
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.initial_timeout * (attempt + 1)
                )
                
                if response.status_code == 400:
                    error_msg = response.json().get('error', {}).get('message', '')
                    if "messages" in error_msg:
                        # Fix messages format if that's the error
                        payload["messages"] = [{"role": "user", "content": prompt}]
                        continue
                    
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
                
            except requests.exceptions.RequestException as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    return json.dumps({
                        "api_error": True,
                        "message": str(e),
                        "status_code": getattr(e.response, 'status_code', None)
                    })
                time.sleep(min(2 ** attempt, 60))


    # def call(self, prompt: str, model: str = "deepseek-chat", temperature: float = 0.7, 
    #          max_tokens: int = 1000, **kwargs) -> str:
    #     """
    #     Modified version that:
    #     1. Validates payload before sending
    #     2. Handles 422 errors specifically
    #     3. Provides better error messages
    #     """
    #     headers = {
    #         "Authorization": f"Bearer {self.api_key}",
    #         "Content-Type": "application/json"
    #     }
        
    #     # Build payload with validation
    #     payload = {
    #         "model": model,
    #         "messages": [{"role": "user", "content": prompt}],
    #         "temperature": max(0, min(temperature, 2)),  # Clamped 0-2
    #         "max_tokens": min(max_tokens, 4000),  # Enforce reasonable limit
    #         **kwargs
    #     }
        
    #     # Remove None values to avoid API errors
    #     payload = {k: v for k, v in payload.items() if v is not None}
        
    #     endpoint = urljoin(self.base_url, "v1/chat/completions")
        
    #     for attempt in range(self.max_retries):
    #         try:
    #             response = requests.post(
    #                 endpoint,
    #                 headers=headers,
    #                 json=payload,
    #                 timeout=self.initial_timeout * (attempt + 1)
    #             )
                
    #             # Special handling for 422 errors
    #             if response.status_code == 422:
    #                 error_msg = response.json().get('error', {}).get('message', 'Unknown error')
    #                 print(f"Validation error: {error_msg}")
    #                 # Remove potentially problematic fields
    #                 payload.pop('response_format', None)
    #                 payload.pop('functions', None)
    #                 continue
                
    #             response.raise_for_status()
    #             return response.json()["choices"][0]["message"]["content"]
                
    #         except requests.exceptions.RequestException as e:
    #             print(f"Attempt {attempt + 1} failed: {str(e)}")
    #             if attempt == self.max_retries - 1:
    #                 raise Exception(f"API call failed after {self.max_retries} attempts. Last error: {str(e)}")
    #             time.sleep(min(2 ** attempt, 60))

# class DeepSeekClient:
#     def __init__(self, max_retries: int = 10, initial_timeout: int = 15):
#         self.api_key = os.getenv("DEEPSEEK_API_KEY")
#         self.base_url = os.getenv("DEEPSEEK_API_URL")
#         self.max_retries = max_retries
#         self.initial_timeout = initial_timeout
        
#         if not self.api_key:
#             raise ValueError("DEEPSEEK_API_KEY not found in environment variables")
#         if not self.base_url:
#             raise ValueError("DEEPSEEK_API_URL not found in environment variables")
        
#         if not self.base_url.endswith('/'):
#             self.base_url += '/'

#     # def call(self, prompt: str, model: str = "deepseek-chat", temperature: float = 0.7, max_tokens: int = 1000) -> Optional[str]:
#     def call(self, 
#             prompt: str, 
#             model: str = "deepseek-chat", 
#             temperature: float = 0.7, 
#             max_tokens: int = 1000,
#             response_format: Optional[dict] = None,
#             **kwargs) -> Union[str, dict]:
        
#         headers = {
#             "Authorization": f"Bearer {self.api_key}",
#             "Content-Type": "application/json",
#             "Accept": "application/json"
#         }

#         payload = {
#             "model": model,
#             "messages": [
#                 {
#                     "role": "system",
#                     "content": "Respond with pure JSON. No markdown or extra text." 
#                     if response_format == {"type": "json_object"} else None
#                 },
#                 {"role": "user", "content": prompt}
#             ],
#             "temperature": temperature,
#             "max_tokens": min(max_tokens, 4000),  # More reasonable limit
#             "stream": False,
#             **kwargs
#         }

#         if response_format:
#             payload["response_format"] = response_format

#         # payload = {
#         #     "model": model,
#         #     "messages": [{"role": "user", "content": prompt}],
#         #     "temperature": temperature,
#         #     "max_tokens": min(max_tokens, 1000),  # Enforce lower limit
#         #     "stream": False  # Disabled for compatibility
#         # }
        
#         endpoint = urljoin(self.base_url, "v1/chat/completions")
#         last_error = None
        
#         for attempt in range(self.max_retries):
#             try:
#                 response = requests.post(
#                     endpoint,
#                     headers=headers,
#                     json={k: v for k, v in payload.items() if v is not None},
#                     timeout=self.initial_timeout * (attempt + 1)
#                 )
#                 response.raise_for_status()
                
#                 data = response.json()
#                 content = data["choices"][0]["message"]["content"]
                
#                 # If we requested JSON, attempt to parse it
#                 if response_format == {"type": "json_object"}:
#                     try:
#                         return json.loads(content)
#                     except json.JSONDecodeError:
#                         # Fallback: Try cleaning markdown if present
#                         clean_content = content.replace('```json', '').replace('```', '').strip()
#                         return json.loads(clean_content)
                
#                 return content
                
#             except requests.exceptions.RequestException as e:
#                 last_error = e
#                 if attempt < self.max_retries - 1:
#                     wait_time = min(2 ** attempt, 60)
#                     print(f"Attempt {attempt + 1}/{self.max_retries} failed. Retrying in {wait_time}s... Error: {str(e)}")
#                     time.sleep(wait_time)
        
#         raise Exception(f"API call failed after {self.max_retries} attempts. Last error: {str(last_error)}")
    
