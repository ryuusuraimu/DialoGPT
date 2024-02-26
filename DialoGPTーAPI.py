from flask import Flask, request, Response
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from googletrans import Translator
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

translator = Translator()
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium", padding_side="left")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

@app.route('/generate_response', methods=['GET'])
def generate_response():
    inp_ja = request.args.get('q')
    text_en = translator.translate(inp_ja, src='ja', dest='en').text
    new_user_input_ids = tokenizer.encode(text_en + tokenizer.eos_token, return_tensors='pt')
    chat_history_ids = model.generate(new_user_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(chat_history_ids[:, new_user_input_ids.shape[-1]:][0], skip_special_tokens=True)
    response_ja = translator.translate(response, src='en', dest='ja').text
    # JSONレスポンスをUTF-8でエンコードして返す、ensure_ascii=Falseを設定
    response_data = json.dumps({"response": response_ja}, ensure_ascii=False)
    return Response(response_data, mimetype="application/json; charset=utf-8")

if __name__ == '__main__':
    app.run(port=5004)
