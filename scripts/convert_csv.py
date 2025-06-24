import pandas as pd
import json
import os

# Load the dataset
df = pd.read_csv("data/reddit_jokes.csv")

# Drop rows without joke content
df = df.dropna(subset=["selftext"])

# Optional: Filter out very short or very long jokes
df = df[df["selftext"].str.len().between(20, 300)]

# Sample a smaller subset (adjust as needed)
df = df.sample(1000, random_state=42).reset_index(drop=True)

# Output path
output_path = "data/jokes.jsonl"
os.makedirs("data", exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    print(df["selftext"].iloc[0])  # print first joke

    for idx, joke in enumerate(df["selftext"]):
        first_word = joke.split()[0] if joke.split() else "this"
        prompt = f"Tell me a joke about {first_word}"
        response = joke.strip().replace("\n", " ")
        f.write(json.dumps({"prompt": prompt, "response": response}) + "\n")

print(f"✅ Saved {len(df)} jokes to {output_path}")
