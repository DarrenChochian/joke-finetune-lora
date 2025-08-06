from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import torch
import os

model_name = "teknium/OpenHermes-2.5-Mistral-7B"

data = load_dataset("json", data_files="data/jokes_10k.jsonl")["train"]

def format(example):
    return {"text": f"<s>{example['prompt']} {example['completion']}</s>"}

data = data.map(format)
data = data.remove_columns([col for col in data.column_names if col != "text"])  # remove extras

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    llm_int8_enable_fp32_cpu_offload=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

tokenized = data.map(tokenize, batched=True)

args = TrainingArguments(
    output_dir="models/jokebot",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    save_steps=5,
    save_total_limit=2,
    logging_steps=10,
    bf16=False,
    fp16=True,
    report_to="none",
    disable_tqdm=False
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

last_checkpoint = None
checkpoint_dir = "models/jokebot"
if os.path.isdir(checkpoint_dir):
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint-")]
    if checkpoints:
        last_checkpoint = os.path.join(checkpoint_dir, sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1])

trainer.train(resume_from_checkpoint=last_checkpoint)

model.save_pretrained("models/jokebot")
tokenizer.save_pretrained("models/jokebot")
