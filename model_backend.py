import torch
import quant

from utils import find_layers
import transformers
from transformers import AutoTokenizer, TextStreamer, GenerationConfig, TextIteratorStreamer
import accelerate
from time import time

import utils.llama_accelerate_path as llama_accelerate_path
import json 
from config.load_config import *


def load_quant(model, checkpoint, wbits, groupsize=-1, fused_mlp=True, eval=True, warmup_autotune=True):
    from transformers import LlamaConfig, LlamaForCausalLM
    config = LlamaConfig.from_pretrained(model)

    def noop(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = noop
    torch.nn.init.uniform_ = noop
    torch.nn.init.normal_ = noop

    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = LlamaForCausalLM(config)
    torch.set_default_dtype(torch.float)
    if eval:
        model = model.eval()
    layers = find_layers(model)
    for name in ['lm_head']:
        if name in layers:
            del layers[name]
    quant.make_quant_linear(model, layers, wbits, groupsize)

    del layers

    print('Loading model ...')
    if checkpoint.endswith('.safetensors'):
        from safetensors.torch import load_file as safe_load
        model.load_state_dict(safe_load(checkpoint), strict=True)
    else:
        model.load_state_dict(torch.load(checkpoint), strict=False)

    if eval:
        quant.make_quant_attn(model)
        quant.make_quant_norm(model)
        if fused_mlp:
            quant.make_fused_mlp(model)
    if warmup_autotune:
        quant.autotune_warmup_linear(model, transpose=not (eval))
        if eval and fused_mlp:
            quant.autotune_warmup_fused(model)
    model.seqlen = 2048
    print('Done.')

    device_map = json.load(open(DEVICE_MAP, 'r'))
    model = accelerate.dispatch_model(model, device_map=device_map)     
    model = llama_accelerate_path.apply_to_model(model)
    
    print('Done.')

    return model

def generate(prompt, gen_config, model, tokenizer, streamer=None):    
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(torch.device('cuda:0'))
    time1 = time()
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            streamer=streamer,
           **gen_config
        )
    output = tokenizer.decode([el.item() for el in generated_ids[0]])

    token_count = len(generated_ids[0]) - len(input_ids[0])
    print(f"Tokens: {token_count}")
    print(f"Tokens per second: {token_count / (time() - time1)}")

    return output

class LlamaInferenceBackend:

    def __init__(self,
                 model_dir=MODEL_DIR,
                 checkpoints_path=CHECKPOINTS,
                 wbits=WBITS, groupsize=GROUPSIZE,
                 gen_config=GEN_CONFIG, 
                 device_map=DEVICE_MAP):

        self.model_dir = model_dir
        self.checkpoints_path = checkpoints_path
        self.wbits = wbits
        self.groupsize = groupsize
        self.gen_config = gen_config
        self.device_map = device_map

        self.model = load_quant(self.model_dir, self.checkpoints_path, self.wbits, self.groupsize, eval=True, warmup_autotune=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, use_fast=False)
        self.streamer = TextStreamer(self.tokenizer)
    
    def generate(self, prompt, generation_config=None):
        if generation_config is None:
            generation_config: dict = {}
            generation_config["generation_config"] = GenerationConfig._dict_from_json_file(self.gen_config)
            generation_config["generation_config"] = GenerationConfig.from_dict(generation_config["generation_config"])
        return generate(prompt, generation_config, self.model, self.tokenizer, self.streamer)
    

if __name__ == "__main__":

    backend = LlamaInferenceBackend()
    
    while True:

        user_input = input("Enter prompt: ")

        with open("prompts/translation.txt", "r") as f:
            prompt = f.read()
        prompt = prompt.format(user_input=user_input)
        backend.generate(prompt)





