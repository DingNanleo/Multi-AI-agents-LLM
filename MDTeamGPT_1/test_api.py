from utils.api_client import DeepSeekClient

client = DeepSeekClient(max_retries=1)  # Quick test
try:
    print("Testing API connection...")
    response = client.call("Hello, world!", max_tokens=50)
    print("SUCCESS! API response:", response[:100] + "...")
except Exception as e:
    print("FAILED:", str(e))