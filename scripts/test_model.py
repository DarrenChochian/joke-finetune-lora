from transformers import AutoTokenizer, pipeline, BitsAndBytesConfig, AutoModelForCausalLM
from peft import PeftModel
import torch
import html
import re

# 🔧 Model & Quant config
base_model_name = "teknium/OpenHermes-2.5-Mistral-7B"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    llm_int8_enable_fp32_cpu_offload=True
)

# 🧠 Load base + LoRA adapter
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    quantization_config=bnb_config
)
model = PeftModel.from_pretrained(base_model, "models/jokebot")

# 🧽 Tokenizer
tokenizer = AutoTokenizer.from_pretrained("models/jokebot", use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

# 🧪 Text generation pipeline
gen = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"
)

# 🎯 Safer prompt (like training)
prompt = "Why don’t skeletons fight each other?"

# 🚀 Generate
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


# 🧼 Clean output
output = res[0]["generated_text"]
output = html.unescape(output)
output = re.sub(r"</s>", "", output)
output = re.sub(r"\s+", " ", output).strip()
output = re.split(r"[.?!]", output)[0].strip() + "."
output = re.sub(r"[^\w\s.,!?']", "", output)  # Remove odd symbols


print("🃏 Generated Joke:\n")
print(output)
