
from flask import Flask, request, jsonify, stream_with_context, Response
from model_backend_multiprocessing import model_worker
from transformers import GenerationConfig
from multiprocessing import Process, Queue, Event
import queue

app = Flask(__name__)
input_queue = Queue()
output_queue = Queue()
streamer_queue = Queue()
stopp_event = Event()

@app.route('/stopp_generation', methods=['POST'])
def stopp_generation():
    stopp_event.set()
    output = "Generation stopped."
    output_queue.put(output)
    return jsonify({'output': output}), 200

@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.get_json()
    if 'prompt' not in data:
        return jsonify({'error': 'No prompt provided'}), 400
    if 'gen_config' not in data:
        gen_config = None
    else:
        gen_config = GenerationConfig.from_dict(data['gen_config'])

    prompt = data['prompt']

    # try:
    #     input_queue.put((prompt, gen_config))

    #     while True:
    #         try:
    #             print(streamer_queue.get(timeout=2))
    #         except:
    #             break

    #     output = output_queue.get()
    #     return jsonify({'output': output}), 200
    # except Exception as e:
    #     return jsonify({'error': str(e)}), 500

    try:
        input_queue.put((prompt, gen_config))

        def generate_output():
            while True:
                try:
                    yield str(streamer_queue.get(timeout=2))
                except queue.Empty:
                    break
    
        return Response(stream_with_context(generate_output()), content_type='text/plain')
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    



if __name__ == "__main__":
    
    inference_process = Process(target=model_worker, args=(input_queue, output_queue, stopp_event, streamer_queue))
    inference_process.start()
    app.run(debug=False, host='0.0.0.0', port=5000)