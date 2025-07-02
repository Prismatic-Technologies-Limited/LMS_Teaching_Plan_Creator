from dotenv import load_dotenv
import os
import requests

def call_groq(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
        "Content-Type": "application/json"
    }

    data = {
        "messages": [
            {"role": "system", "content": "You are a helpful academic assistant who designs teaching plans."},
            {"role": "user", "content": prompt}
        ],
        "model": os.getenv("GROQ_MODEL", "mixtral-8x7b-32768")
    }

    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)

    # 🔍 Print full response if 'choices' not in it
    try:
        response_json = response.json()
        return response_json["choices"][0]["message"]["content"]
    except Exception as e:
        print("⚠️ Groq API response error:", response.text)  # this will show the actual error message from Groq
        raise
