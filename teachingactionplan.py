from fastapi import FastAPI, Request, HTTPException
from dotenv import load_dotenv
import os
import requests
import json

load_dotenv()
app = FastAPI()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen-qwq-32b")  # safer default model

def build_prompt(data):
    return f"""
You are an expert teaching assistant AI.

Generate a lecture teaching plan, key concepts, student learning outcomes (SLOs), and suggestions to teach the following topic.

Lecture Details:
- Lecture Title: {data['lecture_title']}
- Subject: {data['subject']}
- Faculty: {data['faculty']}
- Target Year: {data['target_year']}
- Date: {data['date']}

Provide the output strictly in JSON format like:
{{
  "lecture_title": "...",
  "key_concepts": ["...", "..."],
  "student_learning_outcomes": ["...", "..."],
  "teaching_plan": ["...", "..."],
  "suggestions": ["...", "..."]
}}
"""

def call_groq(prompt):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=data
    )

    try:
        response_json = response.json()

        # Handle API errors
        if "error" in response_json:
            raise HTTPException(
                status_code=500,
                detail=f"Groq API Error: {response_json['error'].get('message', 'Unknown error')}"
            )

        # Safely extract choice
        choices = response_json.get("choices")
        if not choices or "message" not in choices[0]:
            raise HTTPException(
                status_code=500,
                detail="Invalid response from Groq API: 'choices[0].message' not found."
            )

        return choices[0]["message"]["content"]

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid JSON received from Groq API.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected Error: {str(e)}")

@app.post("/generate-plan/")
async def generate_generic_plan(request: Request):
    try:
        data = await request.json()
        prompt = build_prompt(data)
        response = call_groq(prompt)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
