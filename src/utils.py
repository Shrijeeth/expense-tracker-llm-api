import torch

from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer

model = None
tokenizer = None

async def load_configs(model_name_or_path):
    peft_config = PeftConfig.from_pretrained(model_name_or_path)
    bnb_config = BitsAndBytesConfig(
        load_in_8bit = False,
        load_in_4bit = True,
        llm_int8_threshold = 6.0,
        llm_int8_skip_modules = None,
        llm_int8_enable_fp32_cpu_offload = False,
        llm_int8_has_fp16_weight = False,
        bnb_4bit_quant_type = 'nf4',
        bnb_4bit_use_double_quant = True,
        bnb_4bit_compute_dtype = torch.float16,
    )
    return peft_config, bnb_config

async def load_model(model_name_or_path):
    global model, tokenizer
    peft_config, bnb_config = await load_configs(model_name_or_path)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
    if model is None:
        base_model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, quantization_config=bnb_config)
        model = PeftModel.from_pretrained(base_model, model_name_or_path, config=peft_config)
    return model, tokenizer

def download_llama_2():
    from huggingface_hub import snapshot_download

    model_name = "meta-llama/Llama-2-7b-chat-hf"
    snapshot_download(model_name)