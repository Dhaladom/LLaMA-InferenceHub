
from flask import Flask, request, jsonify
from model_backend import LlamaInferenceBackend
from transformers import GenerationConfig

model = LlamaInferenceBackend()

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.get_json()
    if 'prompt' not in data:
        return jsonify({'error': 'No prompt provided'}), 400
    if 'gen_config' not in data:
        gen_config = None
    else:
        gen_config = GenerationConfig._dict_from_json_file(data['gen_config'])

    prompt = data['prompt']

    try:
        output = model.generate(prompt, gen_config)
        return jsonify({'generated_text': output})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)