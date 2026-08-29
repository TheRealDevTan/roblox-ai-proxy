import os
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

HF_TOKEN = os.getenv("HF_TOKEN")
HF_URL = "https://router.huggingface.co/v1/chat/completions"
MODEL = "openai/gpt-oss-120b:fastest"


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(silent=True) or {}

        system_prompt = data.get("system", "")
        history = data.get("history") or []
        user_message = data.get("message", "")

        if not HF_TOKEN:
            return jsonify({
                "reply": "[ERROR] HF_TOKEN is missing in Render."
            }), 500

        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": str(system_prompt)
            })

        for item in history:
            if isinstance(item, dict):
                messages.append({
                    "role": item.get("role", "user"),
                    "content": str(
                        item.get("content")
                        or item.get("message")
                        or item.get("text")
                        or ""
                    )
                })
            else:
                messages.append({
                    "role": "user",
                    "content": str(item)
                })

        messages.append({
            "role": "user",
            "content": str(user_message)
        })

        response = requests.post(
            HF_URL,
            headers={
                "Authorization": f"Bearer {HF_TOKEN}",
                "Content-Type": "application/json"
            },
            json={
                "model": MODEL,
                "messages": messages,
                "max_tokens": 120,
                "temperature": 0.7,
                "stream": False
            },
            timeout=60
        )

        if response.status_code != 200:
            print("[HUGGING FACE ERROR]", response.status_code)
            print(response.text[:500])

            return jsonify({
                "reply": f"[HF ERROR {response.status_code}] {response.text[:300]}"
            }), 200

        result = response.json()
        reply = result["choices"][0]["message"]["content"]

        return jsonify({
            "reply": reply
        }), 200

    except Exception as error:
        print("[PROXY CRASH]", repr(error))

        return jsonify({
            "reply": "[PROXY CRASH] The Python proxy failed."
        }), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
