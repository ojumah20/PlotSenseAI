import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API details
API_KEY = os.getenv("GROQ_API_KEY")
MODEL_ID = "llama-3.2-11b-vision-preview"  # Vision-enabled model
API_URL = f"https://api.groq.com/v1/chat/completions" #need to correct this

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Step 1: Test with a basic text prompt
payload = {
    "model": MODEL_ID,
    "messages": [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Explain the importance of data visualization in business."}
    ],
    "max_tokens": 200
}

response = requests.post(API_URL, headers=headers, json=payload)

if response.status_code == 200:
    result = response.json()
    print("AI Response:", result["choices"][0]["message"]["content"])
else:
    print(f"Error: {response.status_code}, {response.text}")
