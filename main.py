import os
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

# Free, open-source model endpoint (Hugging Face Serverless inference API)
API_URL = "https://huggingface.co"
API_TOKEN = os.getenv("HF_API_TOKEN") # Safely loads your API key from environment settings

@app.route('/chat', methods=['POST'])
def chat_proxy():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"reply": "[PROXY ERROR] No payload data received."}), 400
            
        system_prompt = data.get("system", "")
        chat_history = data.get("history",)
        new_user_message = data.get("message", "")
        
        # Format incoming fields into standard ChatML tags that the LLM understands
        formatted_prompt = f"<|system|>\n{system_prompt}\n"
        for exchange in chat_history:
            formatted_prompt += f"{exchange}\n"
        formatted_prompt += f"<|user|>\n{new_user_message}\n<|assistant|>\n"
        
        # Build network payload to fire out to Hugging Face
        headers = {
            "Authorization": f"Bearer {API_TOKEN}" if API_TOKEN else "",
            # FIX: Adding a standard User-Agent headers string prevents the 403 block
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        payload = {
            "inputs": formatted_prompt,
            "parameters": {"max_new_tokens": 120, "temperature": 0.7}
        }
        
        response = requests.post(API_URL, json=payload, headers=headers, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            generated_text = ""
            if isinstance(result, list) and len(result) > 0:
                generated_text = result.get("generated_text", "")
            elif isinstance(result, dict):
                generated_text = result.get("generated_text", "")
                
            # Clean off header instructions before passing text down to the player GUI
            clean_reply = generated_text.replace(formatted_prompt, "").strip()
            return jsonify({"reply": clean_reply}), 200
        else:
            return jsonify({"reply": f"[SERVER ERROR] Public LLM returned code: {response.status_code}"}), 200

    except Exception as e:
        return jsonify({"reply": f"[PROXY CRASH] Core routing failed: {str(e)}"}), 500

if __name__ == '__main__':
    # Listen dynamically on all network interfaces for deployment compliance
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", 5000)))
