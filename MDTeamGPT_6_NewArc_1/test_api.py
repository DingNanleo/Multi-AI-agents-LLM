from utils.api_client import DeepSeekClient
from utils.api_client2 import GroqClient

#client = DeepSeekClient(max_retries=5)  # Quick test
client = GroqClient(max_retries=1)  # Quick test
try:
    print("Testing API connection...")
    response = client.call("Hello, world!", max_tokens=50)
    print("SUCCESS! API response:", response[:100] + "...")
except Exception as e:
    print("FAILED:", str(e))