import pandas as pd

# Load your CSV file (adjust the path if needed)
df = pd.read_csv("data/reddit_jokes.csv")

# Print the list of column names to the console
print(df.columns)
