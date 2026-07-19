import os
from groq import Groq

api_key = os.environ.get("GROQ_API_KEY")
print(f"API Key found: {api_key[:10]}..." if api_key else "NO API KEY!")

client = Groq(api_key=api_key)
response = client.chat.completions.create(
    model    = "llama-3.3-70b-versatile",
    messages = [{"role": "user", "content": "Say hello in one word"}],
    max_tokens = 10,
)
print("Groq response:", response.choices[0].message.content)