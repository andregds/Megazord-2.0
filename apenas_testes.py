import requests
print(requests.get("https://api.deepseek.com", timeout=10).status_code)
