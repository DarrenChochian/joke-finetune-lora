import os
from transformers import AutoTokenizer, pipeline, BitsAndBytesConfig, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch
import html
import re

base_model_name = "teknium/OpenHermes-2.5-Mistral-7B"


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    llm_int8_enable_fp32_cpu_offload=True
)


base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    quantization_config=bnb_config
)


adapter_path = os.path.abspath("../models/jokebot")

peft_config = PeftConfig.from_pretrained(
    adapter_path,
    local_files_only=True
)

model = PeftModel(base_model, peft_config)

tokenizer = AutoTokenizer.from_pretrained(
    base_model_name,
    use_fast=False
)
tokenizer.pad_token = tokenizer.eos_token

gen = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"
)

prompt = "Why don’t skeletons fight each other?"

res = gen(
    prompt,
    max_new_tokens=40,
    return_full_text=False,
    do_sample=True,
    temperature=0.7,
    top_k=40,
    top_p=0.9,
    repetition_penalty=1.1,
    eos_token_id=tokenizer.eos_token_id,
)

output = res[0]["generated_text"]
output = html.unescape(output)
output = re.sub(r"</s>", "", output)
output = re.sub(r"\s+", " ", output).strip()
output = re.split(r"[.?!]", output)[0].strip() + "."
output = re.sub(r"[^\w\s.,!?']", "", output)  # Remove unwanted symbols

print("🃏 Generated Joke:\n")
print(output)
