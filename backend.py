from flask import Flask, request, jsonify
from google import genai
from google.genai import types
import os


app = Flask(__name__)

GEMINI_MODEL = "gemini-3.8-flash"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

gemini_client = None

if GEMINI_API_KEY:
    gemini_client = genai.Client(
        api_key=GEMINI_API_KEY
    )


def ask_gemini(prompt):

    if gemini_client is None:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured."
        )

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2
        )
    )

    if not response.text:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    return response.text.strip()


@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "status": "online",
        "message": "AI Study Assistant backend is running."
    })


@app.route("/ask", methods=["POST"])
def ask():

    try:

        data = request.get_json(
            silent=True
        ) or {}

        question = str(
            data.get("question", "")
        ).strip()

        subject = str(
            data.get("subject", "General")
        ).strip()

        answer_type = str(
            data.get("answer_type", "Detailed")
        ).strip()

        if not question:

            return jsonify({
                "success": False,
                "error": "Question is required."
            }), 400

        prompt = f"""
You are an accurate AI Study Assistant.

Subject:
{subject}

Answer type:
{answer_type}

Student question:
{question}

Give a clear, accurate and easy-to-understand answer.

LANGUAGE RULES:
- Reply in the same language used by the student.
- Tamil script question → simple natural Tamil.
- Tanglish question → simple Tanglish.
- English question → English.
- Do not unnecessarily translate the question.
- Keep important technical terms in English when appropriate.

ACCURACY RULES:
- Do not invent facts.
- For Python, follow actual Python behavior.
- For mathematics, calculate carefully.
- If code is provided, determine the actual output or error.
- If you are uncertain, say so instead of making up an answer.

If the answer type is Short:
Give a concise answer.

If the answer type is Detailed:
Give a clear explanation with useful details.
"""

        answer = ask_gemini(
            prompt
        )

        return jsonify({
            "success": True,
            "answer": answer
        })

    except Exception as error:

        return jsonify({
            "success": False,
            "error": str(error)
        }), 500


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=8000,
        debug=True
    )