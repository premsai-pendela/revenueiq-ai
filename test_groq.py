"""
Test Groq API Connection
"""
import os
from dotenv import load_dotenv
from groq import Groq

# Load API key from .env file
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Test with a simple prompt
print("🤖 Testing Groq API connection...\n")

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",  # ✅ UPDATED MODEL
    messages=[
        {
            "role": "system",
            "content": "You are a helpful AI assistant."
        },
        {
            "role": "user",
            "content": "Say 'Hello! Groq is working!' in a friendly way."
        }
    ],
    temperature=0.7,
    max_tokens=100
)

print("✅ Response from Groq:\n")
print(response.choices[0].message.content)
print("\n🎉 Success! Groq is connected and working!")
