from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
import torch

model_name = "mistralai/Mistral-7B-Instruct-v0.1"

# Load dataset
data = load_dataset("json", data_files="data/jokes.jsonl")["train"]

# Prompt + response formatting
def format(example):
    return {
        "text": f"<s>[INST] {example['prompt']} [/INST] {example['response']}</s>"
    }

data = data.map(format)

# Load tokenizer and model in 4bit
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto"
)

# LoRA configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)

# Tokenize the text
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

tokenized = data.map(tokenize)

# Training configuration
args = TrainingArguments(
    output_dir="models/jokebot",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    save_strategy="epoch",
    logging_steps=10,
    bf16=False,
    fp16=True
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

trainer.train()

# Save model and tokenizer
model.save_pretrained("models/jokebot")
tokenizer.save_pretrained("models/jokebot")
