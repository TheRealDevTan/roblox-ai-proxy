import os
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

API_URL = "https://router.huggingface.co/v1/chat/completions"
API_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HF_API_TOKEN")
MODEL = "openai/gpt-oss-120b:fastest"


@app.route("/chat", methods=["POST"])
def chat_proxy():
    try:
        data = request.get_json(silent=True) or {}

        system_prompt = data.get("system", "")
        history = data.get("history") or []
        user_message = data.get("message", "")

        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        for item in history:
            if isinstance(item, dict):
                messages.append({
                    "role": item.get("role", "user"),
                    "content": str(item.get("content", ""))
                })
            else:
                messages.append({
                    "role": "user",
                    "content": str(item)
                })

        messages.append({
            "role": "user",
            "content": user_message
        })

        if not API_TOKEN:
            return jsonify({
                "reply": "[PROXY ERROR] HF_TOKEN is missing on Render."
            }), 500

        response = requests.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {API_TOKEN}",
                "Content-Type": "application/json"
            },
            json={
                "model": MODEL,
                "messages": messages,
                "max_tokens": 120,
                "temperature": 0.7,
                "stream": False
            },
            timeout=30
        )

        if response.status_code != 200:
            print("[HF ERROR]", response.status_code, response.text[:500])
            return jsonify({
                "reply": f"[HF ERROR {response.status_code}] {response.text[:300]}"
            }), 200

        result = response.json()
        reply = result["choices"][0]["message"]["content"]

        return jsonify({"reply": reply}), 200

    except Exception as e:
        print("[PROXY CRASH]", repr(e))
        return jsonify({
            "reply": "[PROXY CRASH] Core routing failed."
        }), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", 5000))
    )
