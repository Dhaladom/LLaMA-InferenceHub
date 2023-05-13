# What is TALIS?

<u>T</u>riton <u>a</u>ccelerated <u>L</u>LaMA <u>i</u>nference <u>s</u>erver (**TALIS**) attempts to become a simple, fast and robust solution for serving LLaMA models via API with an emphasis on inference speed.

This is a super<sup>super</sup> early version of TALIS. Dont expect it to work. For now it supports:

- [x] Running GPTQ quantized 65B LLaMA models on 2x 24GB VRAM Nvidia GPUs on Linux.

# What can it do?
For now it enables 65B-LLaMA models to run primerily on dual RTX 3090 or RTX 4090 GPU's with decent speed. Some benchmarks my come soon, but the gist is it can run 65B-LLaMA models at over **10 tps** (tokens per second) on two RTX 4090's with a **max sequence length of 1525 tokens** on a Linux headless server.

# How to use
For now this is geared towards people familiar with Linux and Python. If you are not, you can still use it, but you will have to do some research on your own.

The requirements may or may not be correct. Sorry. (Reach out if you have issues.)

### (Very) Basic Example

The following will let you parse inputs to the model and get outputs from the model via the command line.

1. Specify the settings in the "load_config.py" file:
````
# example load_config.py

MODEL_DIR = "/path/to/your/model/dir"
CHECKPOINTS = "/path/to/your/checkpoints.safetensors"
WBITS = 4
GROUPSIZE = 128
GEN_CONFIG = "gen_default.json"
DEVICE_MAP = "device_map_standard.json"
````
2. Start the python script from within the repo directory:
````
python3 llama_inference.py
````

# What is Planned? (In order of priority)

- [ ] Provide an actual server and API
- [ ] Support more LLaMA model-sizes and GPU's
- [ ] Provide docker support
- [ ] Provide a simple web interface
- [ ] (maybe) substitute Huggingface libs for more lightweight solutions (watching [this](https://github.com/turboderp/exllama) closely)


# Acknowledgements
This code is based on [GPTQ](https://github.com/IST-DASLab/gptq) and [GPTQ-forLLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa).

Triton GPTQ kernel code is based on [GPTQ-triton](https://github.com/fpgaminer/GPTQ-triton).

The user GitHub user [emvw7yf](https://github.com/emvw7yf) who provided the [llama-accelerate-path](https://github.com/huggingface/accelerate/issues/1394) patch, which gave a 5x speedup and really made the whole project viable.

Thanks to Meta AI for releasing [LLaMA](https://arxiv.org/abs/2302.13971), a powerful LLM.

