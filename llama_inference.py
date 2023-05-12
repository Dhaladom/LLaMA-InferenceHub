import torch
import quant

from utils import find_layers, DEV
import transformers
from transformers import AutoTokenizer, TextStreamer, GenerationConfig
import accelerate
from time import time

from os import environ

import llama_accelerate_path
import json 


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


    device_map = json.load(open("/home/ddl/LLaMA-InferenceHub/config/device_map.json", 'r'))
    model = accelerate.dispatch_model(model, device_map=device_map)     
    model = llama_accelerate_path.apply_to_model(model)
    


    print('Done.')

    return model

def inference(model, tokenizer, input, gen_config, streamer=None):    

    input_ids = tokenizer.encode(input, return_tensors="pt").to(torch.device('cuda:0'))

    generation_config = gen_config
    time1 = time()
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            streamer=streamer,
           **generation_config
        )

    token_count = len(generated_ids[0]) - len(input_ids[0])

    output = tokenizer.decode([el.item() for el in generated_ids[0]])
    print(f"Tokens: {token_count}")
    print(f"Tokens per second: {token_count / (time() - time1)}")

    return output

if __name__ == '__main__':

    model_dir: str = "/home/ddl/models/Neko-Institute-of-Science_LLaMA-65B-4bit-128g"
    model_file_path: str = "/home/ddl/models/Neko-Institute-of-Science_LLaMA-65B-4bit-128g/gptq_model-4bit-128g.safetensors"
    wbits: int = 4
    groupsize: int = 128

    model:str = load_quant(model_dir, model_file_path, wbits, groupsize, eval=True, warmup_autotune=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    streamer = TextStreamer(tokenizer)

    while True:

        prompt = input("Enter a prompt: ")
        gen_config = GenerationConfig._dict_from_json_file("/home/ddl/LLaMA-InferenceHub/config/gen_default.json")
        output = inference(model, tokenizer, prompt, gen_config, streamer)
        print(output)



