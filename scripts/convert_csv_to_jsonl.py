import csv
import json
import os
import html
import re

csv_file = "data/one-million-reddit-jokes.csv"
jsonl_file = "data/jokes_10k.jsonl"

def clean_text(text):
    text = html.unescape(text)  # Convert HTML entities (e.g., &amp;)
    text = re.sub(r"[\u200B-\u200D\uFEFF]", "", text)  # Remove zero-width chars
    text = re.sub(r"&[#a-zA-Z0-9]+;", "", text)        # Remove remaining HTML-like entities
    text = re.sub(r"\s+", " ", text)                   # Collapse whitespace
    text = text.strip()
    return text

def csv_to_jsonl(csv_path, jsonl_path, max_samples=10000):
    count = 0
    with open(csv_path, "r", encoding="utf-8") as csv_f, open(jsonl_path, "w", encoding="utf-8") as jsonl_f:
        reader = csv.DictReader(csv_f)
        for row in reader:
            if count >= max_samples:
                break

            prompt_raw = row.get("title", "").strip()
            completion_raw = row.get("selftext", "").strip()

            # Skip bad or removed samples
            if not prompt_raw or "[removed]" in completion_raw.lower() or "[deleted]" in completion_raw.lower():
                continue

            # Fallback to title-only if no body
            if not completion_raw:
                completion_raw = prompt_raw

            # Clean both prompt and completion
            prompt = clean_text(prompt_raw)
            completion = clean_text(completion_raw)

            # Skip too short or nonsense content
            if len(prompt) < 5 or len(completion) < 5:
                continue

            json_obj = {
                "prompt": prompt,
                "completion": completion
            }

            json_line = json.dumps(json_obj, ensure_ascii=False)
            jsonl_f.write(json_line + "\n")

            if count < 5:
                print(f"Sample {count+1}: {json_line}")

            count += 1

    print(f"✅ Converted {count} clean jokes to JSONL.")

if __name__ == "__main__":
    abs_path = os.path.abspath(jsonl_file)
    print(f"Writing output to: {abs_path}")
    csv_to_jsonl(csv_file, jsonl_file, max_samples=10000)
